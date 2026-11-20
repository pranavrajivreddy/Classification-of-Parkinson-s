from pathlib import Path
import hashlib
import json
import platform
import subprocess
from datetime import datetime, timezone

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'paper_ready'
PRIVATE = OUT / 'private'
CACHE = OUT / 'cache'


def sha256(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, default=str, allow_nan=False) + '\n')


def fingerprint():
    return {str(p.relative_to(ROOT)): sha256(p) for p in
            sorted((ROOT / 'research').glob('*.py'))}


def git_revision():
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True, stderr=subprocess.DEVNULL).strip()
    except (OSError, subprocess.CalledProcessError):
        return "unversioned-release"


def manifest(extra=None):
    return dict(created_utc=datetime.now(timezone.utc).isoformat(),
                python=platform.python_version(), platform=platform.platform(),
                git_commit=git_revision(),
                source_hashes=fingerprint(), **(extra or {}))
