"""Isolate candidate edits and freeze their exact source without committing.

The allowlist protects experiment infrastructure from accidental benchmark changes.
It is a research policy, not a security sandbox for untrusted Python code.
"""
from __future__ import annotations

import hashlib
import json
import re
import shutil
import stat
import subprocess
import tempfile
from pathlib import Path, PurePosixPath

from src.research.protocol import digest


ALLOWED_PATHS = ("src/models/", "src/attention/", "src/moe/", "src/config/", "tests/")


def _git(root: Path, *arguments: str, check: bool = True) -> subprocess.CompletedProcess:
    result = subprocess.run(["git", "-C", str(root), *arguments], capture_output=True)
    if check and result.returncode:
        detail = result.stderr.decode("utf-8", errors="replace").strip()
        raise RuntimeError(f"git {' '.join(arguments[:2])} failed: {detail}")
    return result


def _repository(path: Path) -> Path:
    root = Path(path).expanduser().resolve()
    actual = Path(_git(root, "rev-parse", "--show-toplevel").stdout.decode().strip()).resolve()
    if root != actual:
        raise ValueError(f"Expected the repository root, got {root}; root is {actual}")
    return root


def _commit(root: Path, ref: str) -> str:
    if not isinstance(ref, str) or not ref or ref.startswith("-"):
        raise ValueError("A valid committed Git ref is required")
    return _git(root, "rev-parse", "--verify", "--end-of-options", f"{ref}^{{commit}}").stdout.decode().strip()


def _safe_path(root: Path, relative: str) -> Path:
    parsed = PurePosixPath(relative)
    if not relative or parsed.is_absolute() or any(part in (".", "..", ".git") for part in parsed.parts):
        raise ValueError(f"Unsafe repository path: {relative!r}")
    path = root
    for part in parsed.parts:
        path = path / part
        if path.is_symlink():
            raise ValueError(f"Symlinks are not supported in candidate artifacts: {relative}")
    if not path.is_relative_to(root):
        raise ValueError(f"Path escapes the candidate: {relative}")
    return path


def _base_tree(root: Path, commit: str) -> dict:
    entries = {}
    for item in _git(root, "ls-tree", "-rz", "--full-tree", commit).stdout.split(b"\0"):
        if not item:
            continue
        header, raw_path = item.split(b"\t", 1)
        mode, kind, object_id = header.decode().split()
        relative = raw_path.decode("utf-8")
        _safe_path(root, relative)
        if kind != "blob" or mode not in ("100644", "100755"):
            raise ValueError(f"Symlinks and submodules are unsupported: {relative} ({mode})")
        content = _git(root, "cat-file", "blob", object_id).stdout
        entries[relative] = {"sha256": hashlib.sha256(content).hexdigest(), "mode": mode}
    return entries


def _working_files(root: Path) -> dict:
    if _git(root, "ls-files", "--unmerged").stdout:
        raise ValueError("Resolve Git conflicts before inspecting a candidate")
    for entry in _git(root, "ls-files", "--stage", "-z").stdout.split(b"\0"):
        if entry and entry.split(b" ", 1)[0] not in (b"100644", b"100755"):
            raise ValueError("Symlinks and submodules are unsupported in the candidate index")
    names = set(_git(root, "ls-files", "--cached", "--others", "--exclude-standard", "-z").stdout.split(b"\0"))
    files = {}
    for raw_name in sorted(names - {b""}):
        relative = raw_name.decode("utf-8")
        path = _safe_path(root, relative)
        if not path.exists():
            continue  # A tracked deletion is recorded against the base tree below.
        if not path.is_file():
            raise ValueError(f"Only regular files can be frozen: {relative}")
        info = path.stat()
        if not stat.S_ISREG(info.st_mode):
            raise ValueError(f"Only regular files can be frozen: {relative}")
        files[relative] = {"sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                           "mode": "100755" if info.st_mode & 0o111 else "100644"}
    return files


def source_digest(root: Path) -> str:
    """Match the harness's canonical hash of Python source and requirements."""
    root = Path(root).resolve()
    paths = sorted((root / "src").rglob("*.py"))
    if (root / "requirements.txt").exists():
        paths.append(root / "requirements.txt")
    hashes = {}
    for path in paths:
        relative = path.relative_to(root).as_posix()
        path = _safe_path(root, relative)
        if not path.is_file():
            raise ValueError(f"Expected regular source file: {relative}")
        hashes[relative] = hashlib.sha256(path.read_bytes()).hexdigest()
    return digest(hashes)


def create_workspace(repo: Path, destination: Path, branch: str, base_ref: str = "HEAD") -> dict:
    """Create a new branch/worktree only from a clean, committed repository."""
    root = _repository(repo)
    if _git(root, "status", "--porcelain=v1", "-z", "--untracked-files=all").stdout:
        raise ValueError("Candidate creation requires a clean committed baseline. Preserve or commit the current "
                         "changes first, including the prepared harness; this function never commits for you.")
    if not re.fullmatch(r"codex/[A-Za-z0-9][A-Za-z0-9._-]*/[A-Za-z0-9][A-Za-z0-9._-]*", branch):
        raise ValueError("Candidate branch must be codex/<campaign>/<idea>")
    _git(root, "check-ref-format", "--branch", branch)
    if _git(root, "show-ref", "--verify", "--quiet", f"refs/heads/{branch}", check=False).returncode == 0:
        raise ValueError(f"Refusing to reset an existing branch: {branch}")
    destination = Path(destination).expanduser().resolve()
    if destination.exists() or destination.is_relative_to(root):
        raise ValueError("Candidate destination must be a new directory outside the source repository")
    commit = _commit(root, base_ref)
    _base_tree(root, commit)  # Refuse symlink/submodule baselines before modifying Git state.
    destination.parent.mkdir(parents=True, exist_ok=True)
    _git(root, "worktree", "add", "-b", branch, str(destination), commit)
    return {"path": str(destination), "branch": branch, "base_commit": commit}


def _allowed(path: str, prefixes: tuple[str, ...]) -> bool:
    return any(path.startswith(prefix) if prefix.endswith("/") else path == prefix for prefix in prefixes)


def _patch(root: Path, commit: str) -> str:
    patch = _git(root, "diff", "--binary", "--no-ext-diff", "--no-textconv", commit, "--").stdout
    for name in _git(root, "ls-files", "--others", "--exclude-standard", "-z").stdout.split(b"\0"):
        if not name:
            continue
        result = _git(root, "diff", "--no-index", "--binary", "--no-ext-diff", "--no-textconv",
                      "--", "/dev/null", name.decode("utf-8"), check=False)
        if result.returncode not in (0, 1):
            raise RuntimeError(result.stderr.decode("utf-8", errors="replace"))
        patch += result.stdout
    # A small binary fixture can be classified as text by Git despite containing
    # non-UTF-8 bytes. Surrogate escaping preserves a byte-exact replayable patch.
    return patch.decode("utf-8", errors="surrogateescape")


def inspect_workspace(worktree: Path, base_commit: str, allowed_paths: list[str] | None = None) -> dict:
    """Inspect the working copy, including staged, unstaged, and untracked edits.

    An optional allowlist can narrow the built-in candidate policy, never expand it.
    Deletions and renames are represented as file changes against the committed base.
    """
    root = _repository(worktree)
    commit = _commit(root, base_commit)
    prefixes = tuple(ALLOWED_PATHS if allowed_paths is None else allowed_paths)
    for prefix in prefixes:
        if not isinstance(prefix, str) or not prefix or not _allowed(prefix, ALLOWED_PATHS):
            raise ValueError(f"Candidate allowlist cannot permit protected paths: {prefix!r}")
        _safe_path(root, prefix.rstrip("/"))
    before = _base_tree(root, commit)
    after = _working_files(root)
    changes = []
    for path in sorted(set(before) | set(after)):
        old, new = before.get(path), after.get(path)
        if old == new:
            continue
        if not _allowed(path, prefixes):
            raise ValueError(f"Candidate changed protected path: {path}")
        changes.append({"path": path, "status": "added" if old is None else "deleted" if new is None else "modified",
                        "before_sha256": old["sha256"] if old else None,
                        "after_sha256": new["sha256"] if new else None,
                        "before_mode": old["mode"] if old else None,
                        "after_mode": new["mode"] if new else None})
    patch = _patch(root, commit)
    return {"path": str(root), "branch": _git(root, "branch", "--show-current").stdout.decode().strip(),
            "base_commit": commit, "dirty": bool(changes), "changes": changes,
            "files": {name: meta["sha256"] for name, meta in after.items()},
            "source_digest": source_digest(root), "patch": patch,
            "patch_sha256": hashlib.sha256(patch.encode("utf-8", errors="surrogateescape")).hexdigest()}


def freeze_workspace(worktree: Path, destination: Path, base_commit: str) -> dict:
    """Copy a verified candidate to a plain snapshot with read-only source files.

    No Git metadata, ignored datasets, environments, or generated runs are copied.
    Read-only permissions discourage accidental edits; hashes establish provenance.
    """
    inspected = inspect_workspace(worktree, base_commit)
    root = Path(inspected["path"])
    if {"candidate-manifest.json", "candidate.patch"} & set(inspected["files"]):
        raise ValueError("Repository files conflict with reserved candidate artifact names")
    destination = Path(destination).expanduser().resolve()
    if destination.exists() or destination.is_relative_to(root):
        raise ValueError("Snapshot destination must be a new directory outside the worktree")
    destination.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=f".{destination.name}-freeze-", dir=destination.parent))
    try:
        for relative, expected in inspected["files"].items():
            source = _safe_path(root, relative)
            content = source.read_bytes()
            if hashlib.sha256(content).hexdigest() != expected:
                raise ValueError(f"Candidate changed while freezing: {relative}")
            target = staging / relative
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(content)
            target.chmod(0o555 if source.stat().st_mode & 0o111 else 0o444)
        current = inspect_workspace(root, base_commit)
        if current != inspected or source_digest(staging) != inspected["source_digest"]:
            raise ValueError("Candidate changed while freezing; retry after edits stop")
        manifest = {key: value for key, value in inspected.items() if key not in ("patch", "path")}
        manifest.update({"path": str(destination), "worktree": str(root),
                         "manifest_path": str(destination / "candidate-manifest.json"),
                         "patch_path": str(destination / "candidate.patch")})
        (staging / "candidate-manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
        (staging / "candidate.patch").write_bytes(inspected["patch"].encode("utf-8", errors="surrogateescape"))
        (staging / "candidate-manifest.json").chmod(0o444)
        (staging / "candidate.patch").chmod(0o444)
        staging.rename(destination)
        return manifest
    except BaseException:
        shutil.rmtree(staging, ignore_errors=True)
        raise
