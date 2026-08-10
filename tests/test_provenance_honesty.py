from __future__ import annotations

import pytest

from pdac_circuit.core.provenance import (
    GATED,
    POINTER,
    REAL,
    build_doc,
    make_artifact,
    verify_provenance,
)
from pdac_circuit.core.seeds import sha256_file

def test_pointer_must_not_carry_sha256():
    with pytest.raises(ValueError):
        make_artifact("x", "http://e", POINTER, sha256="deadbeef")

def test_gated_must_not_carry_sha256():
    with pytest.raises(ValueError):
        make_artifact("x", "http://e", GATED, sha256="deadbeef")

def test_pointer_artifact_is_clean():
    a=make_artifact("x", "http://example", POINTER)
    assert a["sha256"] is None
    assert a["dataClass"] == POINTER

def test_verify_real_rehash_roundtrip(tmp_path):
    f=tmp_path / "blob.txt"
    f.write_text("real bytes")
    sha=sha256_file(f)
    art=make_artifact("blob", "http://e", REAL, sha256=sha, local_path="blob.txt", n_bytes=10)
    doc=build_doc("demo", [art])
    res=verify_provenance(doc, tmp_path)
    assert res["ok"]
    assert res["real_rehashed"] == 1

def test_verify_detects_tamper(tmp_path):
    f=tmp_path / "blob.txt"
    f.write_text("real bytes")
    art=make_artifact("blob", "http://e", REAL, sha256="0" * 64, local_path="blob.txt")
    doc=build_doc("demo", [art])
    res=verify_provenance(doc, tmp_path)
    assert not res["ok"]
    assert "mismatch" in res["failures"][0]["reason"]

def test_not_materialized_is_honest_not_failure(tmp_path):
    art=make_artifact("absent", "http://e", REAL, sha256="a" * 64, local_path="missing.txt")
    doc=build_doc("demo", [art])
    res=verify_provenance(doc, tmp_path)
    assert res["ok"]
    assert res["not_materialized"] == 1
