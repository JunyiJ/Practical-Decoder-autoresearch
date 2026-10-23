"""Candidate isolation and provenance tests; these never train a model."""
from __future__ import annotations

import hashlib
import json
import subprocess
from pathlib import Path

import pytest

from src.research.protocol import digest
from src.research.team.workspace import (
    create_workspace,
    freeze_workspace,
    inspect_workspace,
    source_digest,
)


def git(root, *args):
    return subprocess.run(["git", "-C", str(root), *args], check=True, capture_output=True).stdout.decode().strip()


@pytest.fixture
def repository(tmp_path):
    root = tmp_path / "repo"
    root.mkdir()
    git(root, "init", "-b", "main")
    git(root, "config", "user.name", "Fixture")
    git(root, "config", "user.email", "fixture@example.invalid")
    files = {
        ".gitignore": ".venv/\ndata/\n__pycache__/\n*.pyc\n",
        "src/models/gpt.py": "VALUE = 1\n",
        "src/models/remove.py": "REMOVE = True\n",
        "src/attention/mha.py": "HEADS = 2\n",
        "src/config/defaults.yaml": "width: 64\n",
        "src/train/train.py": "# Frozen evaluator infrastructure\n",
        "src/research/worker.py": "# Frozen execution infrastructure\n",
        "tests/test_model.py": "def test_placeholder():\n    assert True\n",
        "requirements.txt": "torch>=2.4\n",
        "README.md": "Fixture repository\n",
    }
    for relative, content in files.items():
        target = root / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(content)
    git(root, "add", ".")
    git(root, "commit", "-m", "Fixture baseline")
    return root


def candidate(repository, tmp_path):
    return create_workspace(repository, tmp_path / "candidate", "codex/study/idea")


def test_create_keeps_source_branch_and_preserves_isolation(repository, tmp_path):
    main_commit = git(repository, "rev-parse", "HEAD")
    workspace = candidate(repository, tmp_path)
    root = Path(workspace["path"])
    assert workspace["base_commit"] == main_commit
    assert workspace["branch"] == "codex/study/idea"
    assert git(repository, "branch", "--show-current") == "main"
    (root / "src/models/gpt.py").write_text("VALUE = 2\n")
    assert (repository / "src/models/gpt.py").read_text() == "VALUE = 1\n"
    assert git(repository, "status", "--porcelain") == ""


@pytest.mark.parametrize("dirty_path", ["README.md", "new-file.txt"])
def test_create_requires_clean_committed_baseline(repository, tmp_path, dirty_path):
    (repository / dirty_path).write_text("uncommitted preparation\n")
    with pytest.raises(ValueError, match="clean committed baseline"):
        candidate(repository, tmp_path)
    assert not (tmp_path / "candidate").exists()
    assert git(repository, "branch", "--list", "codex/study/idea") == ""


def test_create_never_resets_existing_branch_or_overwrites_destination(repository, tmp_path):
    git(repository, "branch", "codex/study/idea")
    with pytest.raises(ValueError, match="existing branch"):
        candidate(repository, tmp_path)
    occupied = tmp_path / "occupied"
    occupied.mkdir()
    with pytest.raises(ValueError, match="new directory"):
        create_workspace(repository, occupied, "codex/study/second")
    with pytest.raises(ValueError, match="codex/<campaign>/<idea>"):
        create_workspace(repository, tmp_path / "other", "main")


def test_inspect_and_freeze_capture_all_edit_types_and_replayable_patch(repository, tmp_path):
    workspace = candidate(repository, tmp_path)
    root = Path(workspace["path"])
    (root / "src/models/gpt.py").write_text("VALUE = 2\n")  # Unstaged.
    (root / "src/config/defaults.yaml").write_text("width: 128\n")
    git(root, "add", "src/config/defaults.yaml")  # Staged.
    (root / "src/models/remove.py").unlink()  # Deleted.
    (root / "src/attention/mha.py").rename(root / "src/attention/new_attention.py")  # Rename.
    (root / "tests/test_new.py").write_text("def test_new():\n    assert 1 + 1 == 2\n")  # Untracked.
    (root / "tests/binary_fixture.bin").write_bytes(b"\xff small non-UTF-8 fixture\n")
    (root / "data").mkdir()
    (root / "data/private.txt").write_text("ignored dataset\n")

    inspected = inspect_workspace(root, workspace["base_commit"])
    changes = {change["path"]: change for change in inspected["changes"]}
    assert inspected["dirty"] is True
    assert changes["src/models/gpt.py"]["status"] == "modified"
    assert changes["src/models/remove.py"]["status"] == "deleted"
    assert changes["src/attention/new_attention.py"]["status"] == "added"
    assert changes["tests/test_new.py"]["status"] == "added"
    assert "data/private.txt" not in inspected["files"]
    assert "width: 128" in inspected["patch"]

    snapshot = tmp_path / "snapshot"
    frozen = freeze_workspace(root, snapshot, workspace["base_commit"])
    assert frozen["source_digest"] == source_digest(snapshot) == source_digest(root)
    assert not (snapshot / ".git").exists()
    assert not (snapshot / "data").exists()
    assert not (snapshot / "src/models/remove.py").exists()
    assert (snapshot / "src/models/gpt.py").read_text() == "VALUE = 2\n"
    assert (snapshot / "src/models/gpt.py").stat().st_mode & 0o222 == 0
    assert json.loads((snapshot / "candidate-manifest.json").read_text()) == frozen
    assert hashlib.sha256((snapshot / "candidate.patch").read_bytes()).hexdigest() == inspected["patch_sha256"]

    # A patch must retain untracked additions as well as Git's tracked changes.
    replay = create_workspace(repository, tmp_path / "replay", "codex/study/replay")
    git(Path(replay["path"]), "apply", str(snapshot / "candidate.patch"))
    assert inspect_workspace(Path(replay["path"]), replay["base_commit"])["files"] == inspected["files"]
    (root / "src/models/gpt.py").write_text("VALUE = 3\n")
    assert (snapshot / "src/models/gpt.py").read_text() == "VALUE = 2\n"


@pytest.mark.parametrize("path", ["src/train/train.py", "src/research/worker.py", "requirements.txt", "README.md"])
def test_inspect_rejects_protected_changes_even_if_committed(repository, tmp_path, path):
    workspace = candidate(repository, tmp_path)
    root = Path(workspace["path"])
    (root / path).write_text("changed protocol\n")
    git(root, "add", path)
    git(root, "commit", "-m", "Candidate attempts protected edit")
    with pytest.raises(ValueError, match="protected path"):
        inspect_workspace(root, workspace["base_commit"])


def test_allowlist_can_only_narrow_policy(repository, tmp_path):
    workspace = candidate(repository, tmp_path)
    root = Path(workspace["path"])
    with pytest.raises(ValueError, match="cannot permit protected"):
        inspect_workspace(root, workspace["base_commit"], allowed_paths=["src/train/"])
    with pytest.raises(ValueError, match="Unsafe repository path"):
        inspect_workspace(root, workspace["base_commit"], allowed_paths=["src/models/../../train/"])
    (root / "src/models/gpt.py").write_text("VALUE = 2\n")
    with pytest.raises(ValueError, match="protected path"):
        inspect_workspace(root, workspace["base_commit"], allowed_paths=["src/config/"])


def test_rejects_untracked_symlink_and_submodule(repository, tmp_path):
    workspace = candidate(repository, tmp_path)
    root = Path(workspace["path"])
    link = root / "src/models/linked.py"
    link.symlink_to(repository / "src/train/train.py")
    with pytest.raises(ValueError, match="Symlinks"):
        inspect_workspace(root, workspace["base_commit"])
    link.unlink()
    git(root, "update-index", "--add", "--cacheinfo", f"160000,{workspace['base_commit']},src/models/submodule")
    with pytest.raises(ValueError, match="submodules"):
        inspect_workspace(root, workspace["base_commit"])


def test_source_digest_matches_canonical_protocol_digest(repository):
    hashes = {path.relative_to(repository).as_posix(): hashlib.sha256(path.read_bytes()).hexdigest()
              for path in (repository / "src").rglob("*.py")}
    hashes["requirements.txt"] = hashlib.sha256((repository / "requirements.txt").read_bytes()).hexdigest()
    assert source_digest(repository) == digest(hashes)


def test_freeze_refuses_existing_or_nested_destinations(repository, tmp_path):
    workspace = candidate(repository, tmp_path)
    root = Path(workspace["path"])
    with pytest.raises(ValueError, match="new directory"):
        freeze_workspace(root, tmp_path, workspace["base_commit"])
    with pytest.raises(ValueError, match="outside the worktree"):
        freeze_workspace(root, root / "snapshot", workspace["base_commit"])
