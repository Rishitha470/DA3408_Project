"""Helpers that tie every MLflow run to an exact code version and data version."""
import hashlib
import subprocess
from pathlib import Path

from src.common.config import ROOT


def _git(*args: str) -> str:
    try:
        return subprocess.check_output(["git", *args], cwd=ROOT, stderr=subprocess.DEVNULL,
                                       text=True).strip()
    except (OSError, subprocess.CalledProcessError):
        return ""


def git_commit() -> str:
    """Current Git commit hash, or 'unknown' outside a Git repository."""
    return _git("rev-parse", "HEAD") or "unknown"


def git_dirty() -> bool:
    """True if tracked files have uncommitted changes (the run is then not exactly reproducible)."""
    return bool(_git("status", "--porcelain", "--untracked-files=no"))


def file_md5(path: Path) -> str:
    h = hashlib.md5()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()
