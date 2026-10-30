"""Read-only verification of the nearest release manifest; no network or robot."""
import hashlib
import json
from pathlib import Path


def main():
    root = next((p for p in Path(__file__).resolve().parents
                 if (p / 'PACKAGE_MANIFEST.json').is_file()), None)
    if root is None:
        raise SystemExit('No PACKAGE_MANIFEST.json above this script')
    manifest = json.loads((root / 'PACKAGE_MANIFEST.json').read_text(encoding='utf-8'))
    bad = []
    for entry in manifest['files']:
        path = (root / entry['path']).resolve()
        if not path.is_relative_to(root) or not path.is_file():
            bad.append(entry['path'])
        elif hashlib.sha256(path.read_bytes()).hexdigest() != entry['sha256']:
            bad.append(entry['path'])
    print(json.dumps({'checked': len(manifest['files']), 'mismatched_or_missing': bad}, indent=2))
    return 2 if bad else 0


if __name__ == '__main__':
    raise SystemExit(main())
