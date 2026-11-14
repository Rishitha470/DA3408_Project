"""Encryption at rest for logged user inputs (health data is sensitive).

Each prediction's inputs are encrypted with Fernet (AES-128-CBC + HMAC-SHA256) before being written
to the prediction log that the drift monitor reads. The key comes from PREDICTION_LOG_KEY.
Generate one with:  python -c "from cryptography.fernet import Fernet; print(Fernet.generate_key().decode())"
"""
import json
import os
from pathlib import Path
from typing import Dict, Iterator, Optional

from cryptography.fernet import Fernet, InvalidToken


def get_cipher(key: Optional[str] = None) -> Optional[Fernet]:
    """Return a Fernet cipher from the given key or PREDICTION_LOG_KEY; None if no key is set."""
    key = key or os.environ.get("PREDICTION_LOG_KEY")
    return Fernet(key.encode()) if key else None


def append_encrypted(path: Path, record: Dict, cipher: Fernet) -> None:
    """Encrypt one JSON record and append it as a line."""
    path.parent.mkdir(parents=True, exist_ok=True)
    token = cipher.encrypt(json.dumps(record).encode("utf-8"))
    with open(path, "ab") as fh:
        fh.write(token + b"\n")


def read_encrypted(path: Path, cipher: Fernet, last_n: Optional[int] = None) -> Iterator[Dict]:
    """Yield decrypted records; lines that cannot be decrypted (wrong key, corruption) are skipped."""
    if not path.exists():
        return
    lines = path.read_bytes().splitlines()
    if last_n:
        lines = lines[-last_n:]
    for line in lines:
        try:
            yield json.loads(cipher.decrypt(line.strip()))
        except (InvalidToken, ValueError):
            continue
