from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Sequence

from .seeds import sha256_file

PROVENANCE_SCHEMA="pdac-circuit.provenance/1"

REAL="REAL"
REAL_REFERENCE="REAL-REFERENCE"
POINTER="POINTER"
GATED="GATED"
_ORDER={REAL: 0, REAL_REFERENCE: 1, POINTER: 2, GATED: 3}

HONESTY_NOTE=(
    "REAL = downloaded + sha256-hashed with url/license/retrievedAt. "
    "POINTER = open but resolved at ingest / not statically pulled this run. "
    "GATED = behind an auth/DUC wall, recorded as a pointer and never bypassed. "
    "sha256 is present only when actually hashed and is never fabricated."
)

def now_iso() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")

def reduce_data_class(classes: Sequence[str]) -> str:
    if not classes:
        return REAL
    return max(classes, key=lambda c: _ORDER.get(c, 99))

def make_artifact(
    name: str,
    url: str,
    status: str,
    *,
    sha256: str | None = None,
    n_bytes: int | None = None,
    local_path: str | None = None,
    license: str | None = None,
    note: str | None = None,
) -> dict:
    if status not in _ORDER:
        raise ValueError(f"unknown status {status!r}; must be one of {list(_ORDER)}")
    if status in (POINTER, GATED) and sha256 is not None:
        raise ValueError(f"{status} artifact {name!r} must NOT carry a sha256 (honesty contract)")
    art={
        "name": name,
        "url": url,
        "status": status,
        "dataClass": status,
        "sha256": sha256,
        "bytes": n_bytes,
        "localPath": local_path,
        "license": license,
        "retrievedAt": now_iso() if status in (REAL, REAL_REFERENCE) else None,
    }
    if note:
        art["note"]=note
    return art

def build_doc(corpus_id: str, artifacts: Sequence[dict], *, note: str | None = None) -> dict:
    real=sum(1 for a in artifacts if a["status"] in (REAL, REAL_REFERENCE))
    pointer=sum(1 for a in artifacts if a["status"] == POINTER)
    gated=sum(1 for a in artifacts if a["status"] == GATED)
    return {
        "schema": PROVENANCE_SCHEMA,
        "corpus": corpus_id,
        "generatedAt": now_iso(),
        "note": note or HONESTY_NOTE,
        "summary": {"real": real, "pointer": pointer, "gated": gated},
        "artifacts": list(artifacts),
    }

def write_manifest(manifests_dir: Path, corpus_id: str, doc: dict) -> Path:
    manifests_dir.mkdir(parents=True, exist_ok=True)
    out=manifests_dir / f"{corpus_id}.json"
    out.write_text(json.dumps(doc, indent=2, sort_keys=True), encoding="utf-8")
    return out

def verify_provenance(doc: dict, root: Path) -> dict:
    failures: list[dict]=[]
    real_rehashed=0
    not_materialized=0
    artifacts=doc.get("artifacts", [])

    for a in artifacts:
        name=a.get("name", "?")
        status=a.get("status")
        sha=a.get("sha256")
        if status in (POINTER, GATED):
            if sha is not None:
                failures.append({"name": name, "reason": f"{status} must not carry sha256"})
            continue
        if status in (REAL, REAL_REFERENCE):
            if sha is None:
                failures.append({"name": name, "reason": "REAL artifact missing sha256"})
                continue
            lp=a.get("localPath")
            if not lp:
                failures.append({"name": name, "reason": "REAL artifact missing localPath"})
                continue
            path=(root / lp) if not Path(lp).is_absolute() else Path(lp)
            if not path.exists():
                not_materialized += 1
                continue
            actual=sha256_file(path)
            if actual != sha:
                failures.append(
                    {"name": name, "reason": f"sha256 mismatch (recorded {sha[:12]}, got {actual[:12]})"}
                )
            else:
                real_rehashed += 1
        else:
            failures.append({"name": name, "reason": f"unknown status {status!r}"})

    return {
        "ok": len(failures) == 0,
        "checked": len(artifacts),
        "real_rehashed": real_rehashed,
        "not_materialized": not_materialized,
        "failures": failures,
    }
