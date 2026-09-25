from __future__ import annotations

import hashlib
import hmac
import os
from pathlib import Path


class ArtifactIntegrityError(RuntimeError):
    """Raised when a model artifact fails its integrity contract."""


def sha256_file(path: str | Path, chunk_size: int = 1024 * 1024) -> str:
    path = Path(path)
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


def checksum_path(path: str | Path) -> Path:
    path = Path(path)
    return Path(f"{path}.sha256")


def write_checksum(path: str | Path) -> Path:
    path = Path(path)
    sidecar = checksum_path(path)
    sidecar.parent.mkdir(parents=True, exist_ok=True)
    value = sha256_file(path)
    temporary = sidecar.with_name(f".{sidecar.name}.tmp")
    temporary.write_text(f"{value}\n", encoding="ascii")
    os.replace(temporary, sidecar)
    return sidecar


def verify_checksum(path: str | Path) -> str:
    path = Path(path)
    sidecar = checksum_path(path)
    if not sidecar.exists():
        raise ArtifactIntegrityError(
            f"Model checksum not found at {sidecar.as_posix()}. Retrain the model and retry."
        )
    expected = sidecar.read_text(encoding="ascii").strip().lower()
    if len(expected) != 64 or any(char not in "0123456789abcdef" for char in expected):
        raise ArtifactIntegrityError("Model checksum file is malformed. Retrain the model and retry.")
    actual = sha256_file(path)
    if not hmac.compare_digest(expected, actual):
        raise ArtifactIntegrityError("Model artifact checksum mismatch. Retrain the model and retry.")
    return actual
