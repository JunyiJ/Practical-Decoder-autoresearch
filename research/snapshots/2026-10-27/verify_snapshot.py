"""Verify archived metadata; optionally extract to a new directory. Never run research."""
import argparse
import hashlib
import json
from pathlib import Path, PurePosixPath
import tarfile


def digest(path):
    h = hashlib.sha256()
    with path.open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--extract', type=Path, help='New, nonexistent destination directory')
    args = parser.parse_args()
    folder = Path(__file__).resolve().parent
    manifest = json.loads((folder / 'metadata-manifest.json').read_text())
    archive = folder / manifest['archive']
    if archive.name != 'research-metadata.tar.gz' or digest(archive) != manifest['archive_sha256']:
        raise ValueError('Archive identity mismatch')
    expected = {x['path']: x for x in manifest['files']}
    if len(expected) != len(manifest['files']):
        raise ValueError('Duplicate manifest paths')
    seen = set()
    with tarfile.open(archive, 'r:gz') as tar:
        for member in tar:
            name = PurePosixPath(member.name)
            if not member.isfile() or name.is_absolute() or '..' in name.parts or member.name not in expected or member.name in seen:
                raise ValueError('Unexpected, unsafe or duplicate archive member: ' + member.name)
            item = expected[member.name]
            content = tar.extractfile(member).read()
            if len(content) != item['bytes'] or hashlib.sha256(content).hexdigest() != item['sha256']:
                raise ValueError('Member identity mismatch: ' + member.name)
            seen.add(member.name)
    if seen != set(expected):
        raise ValueError('Missing archive members')
    print(f'Verified {len(seen)} metadata files; {manifest["knowledge_entries"]} knowledge records.')
    if args.extract:
        destination = args.extract.expanduser().absolute()
        if destination.exists() or destination.is_symlink() or any(p.is_symlink() for p in destination.parents):
            raise ValueError('Use a new destination with no symlink parents')
        destination.mkdir(parents=True, exist_ok=False)
        with tarfile.open(archive, 'r:gz') as tar:
            for member in tar:
                path = destination / member.name
                path.parent.mkdir(parents=True, exist_ok=True)
                with path.open('xb') as output:
                    output.write(tar.extractfile(member).read())
        print('Extracted metadata to ' + str(destination))
    print('No training, evaluation, automation, model loading or controller execution occurred.')


if __name__ == '__main__':
    main()
