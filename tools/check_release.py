"""Fail closed on unreviewed release files or common credentials; never print values."""
import hashlib
import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
IGNORE = {'.git', '.venv', '.venv-research', '__pycache__', '.pytest_cache'}
PATTERNS = {
    'provider_key': re.compile(rb'sk-(?:or-v1-|ant-)?[A-Za-z0-9_-]{32,}'),
    'jwt': re.compile(rb'eyJ[A-Za-z0-9_-]{12,}\.[A-Za-z0-9_-]{12,}\.[A-Za-z0-9_-]{12,}'),
    'github_token': re.compile(rb'(?:gh[pousr]_[A-Za-z0-9]{30,}|github_pat_[A-Za-z0-9_]{40,})'),
    'google_api_key': re.compile(rb'AIza[A-Za-z0-9_-]{30,}'),
    'private_key': re.compile(rb'-----BEGIN (?:RSA |EC |OPENSSH )?PRIVATE KEY-----'),
}

def main():
    manifest = json.loads((ROOT/'RELEASE_MANIFEST.json').read_text())['sha256']
    actual = {}
    errors = []
    for path in sorted(ROOT.rglob('*')):
        rel = path.relative_to(ROOT)
        if any(part in IGNORE or part.startswith('.venv') for part in rel.parts):
            continue
        if path.is_symlink():
            errors.append(f'Symlink requires review: {rel}')
            continue
        if not path.is_file():
            continue
        if path.name == '.DS_Store':
            errors.append(f'Remove machine metadata: {rel}')
            continue
        data = path.read_bytes()
        for kind, pattern in PATTERNS.items():
            if pattern.search(data):
                errors.append(f'Credential pattern ({kind}): {rel}')
        if str(rel) == 'RELEASE_MANIFEST.json':
            continue
        actual[str(rel)] = hashlib.sha256(data).hexdigest()
        if str(rel) not in manifest:
            errors.append(f'Unreviewed file: {rel}')
        elif actual[str(rel)] != manifest[str(rel)]:
            errors.append(f'Changed file requires review: {rel}')
    errors.extend(f'Missing file: {p}' for p in sorted(set(manifest)-set(actual)))
    if errors:
        print('\n'.join(errors))
        return 1
    print(f'PASS: {len(actual)} reviewed files; hashes match; no common credential patterns detected.')
    print('Scope: release files only. Git history, ignored environments and chat history are not scanned.')
    return 0

if __name__ == '__main__':
    sys.exit(main())
