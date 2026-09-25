from pathlib import Path

import pytest

from ml.artifacts import (
    ArtifactIntegrityError,
    checksum_path,
    verify_checksum,
    write_checksum,
)


def test_checksum_round_trip(tmp_path: Path):
    artifact = tmp_path / "model.joblib"
    artifact.write_bytes(b"trusted-model-bytes")

    sidecar = write_checksum(artifact)

    assert sidecar == checksum_path(artifact)
    assert verify_checksum(artifact) == sidecar.read_text(encoding="ascii").strip()


def test_checksum_detects_tampering(tmp_path: Path):
    artifact = tmp_path / "model.joblib"
    artifact.write_bytes(b"version-one")
    write_checksum(artifact)
    artifact.write_bytes(b"version-two")

    with pytest.raises(ArtifactIntegrityError, match="checksum mismatch"):
        verify_checksum(artifact)


def test_checksum_rejects_malformed_sidecar(tmp_path: Path):
    artifact = tmp_path / "model.joblib"
    artifact.write_bytes(b"model")
    checksum_path(artifact).write_text("not-a-sha256\n", encoding="ascii")

    with pytest.raises(ArtifactIntegrityError, match="malformed"):
        verify_checksum(artifact)
