from __future__ import annotations

from datetime import datetime
import json
from pathlib import Path, PurePosixPath
import re
import shutil
import tarfile
import uuid

from .streaming import sha256_file

BIGWIG_RE=re.compile(
    r"^(?P<gsm>GSM\d+)_(?P<label>.+)_(?P<assay>ATAC|H3K27ac|H3K4me1|FOXA1)"
    r"(?:_rep(?P<replicate>\d+))?\.bigWig$",
    flags=re.IGNORECASE,
)
RNA_RE=re.compile(r"^(?P<gsm>GSM\d+)_(?P<label>.+)_RNA-Seq_rpkm\.txt\.gz$", re.IGNORECASE)

def _safe_member_path(name: str) -> PurePosixPath:
    path=PurePosixPath(name)
    if path.is_absolute() or not path.parts or any(part in {"", ".", ".."} for part in path.parts):
        raise ValueError(f"unsafe tar member path {name!r}")
    return path

def _candidate(name: str) -> dict | None:
    basename=PurePosixPath(name).name
    match=BIGWIG_RE.fullmatch(basename)
    if match:
        return {
            "gsm": match.group("gsm"),
            "sample_label": match.group("label"),
            "assay": match.group("assay"),
            "replicate": int(match.group("replicate")) if match.group("replicate") else None,
            "state": "UNRESOLVED_REQUIRES_REGISTERED_METADATA",
        }
    match=RNA_RE.fullmatch(basename)
    if match:
        return {
            "gsm": match.group("gsm"),
            "sample_label": match.group("label"),
            "assay": "RNA_seq_RPKM",
            "replicate": None,
            "state": "UNRESOLVED_REQUIRES_REGISTERED_METADATA",
        }
    return None

def inspect_geo_archive(path: str | Path, out: str | Path | None = None) -> dict:
    path=Path(path)
    members, candidates = [], []
    total_bytes=0
    with tarfile.open(path, mode="r:*") as archive:
        for member in archive:
            safe=_safe_member_path(member.name)
            if member.issym() or member.islnk() or member.isdev():
                raise ValueError(f"unsupported special tar member {member.name!r}")
            if member.isdir():
                continue
            if not member.isfile():
                raise ValueError(f"unsupported tar member type {member.name!r}")
            total_bytes += member.size
            candidate=_candidate(member.name)
            record={
                "name": safe.as_posix(),
                "bytes": member.size,
                "candidate": candidate,
            }
            members.append(record)
            if candidate:
                candidates.append({"name": safe.as_posix(), **candidate})
    report={
        "schema": "pdac-circuit.geo-archive-inventory/1",
        "created_at": datetime.now().astimezone().isoformat(),
        "archive": str(path),
        "archive_sha256": sha256_file(path),
        "members": members,
        "member_count": len(members),
        "unpacked_bytes": total_bytes,
        "track_candidates": candidates,
        "state_policy": "No state is inferred from N/P/T/M or other filename tokens.",
    }
    if out:
        destination=Path(out)
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    return report

def extract_geo_archive(
    path: str | Path,
    destination: str | Path,
    *,
    max_unpacked_gb: float = 25.0,
) -> dict:
    path=Path(path)
    destination=Path(destination)
    if destination.exists():
        raise FileExistsError(f"archive destination already exists: {destination}")
    inventory=inspect_geo_archive(path)
    maximum=int(max_unpacked_gb * 1024**3)
    if maximum <= 0 or inventory["unpacked_bytes"] > maximum:
        raise RuntimeError(
            f"archive expands to {inventory['unpacked_bytes'] / 1024**3:.2f} GiB; "
            f"limit is {max_unpacked_gb:.2f} GiB"
        )
    temporary=destination.with_name(f"{destination.name}.partial-{uuid.uuid4().hex}")
    temporary.mkdir(parents=True)
    extracted=[]
    with tarfile.open(path, mode="r:*") as archive:
        for member in archive:
            if not member.isfile():
                continue
            safe=_safe_member_path(member.name)
            output=temporary.joinpath(*safe.parts)
            output.parent.mkdir(parents=True, exist_ok=True)
            source=archive.extractfile(member)
            if source is None:
                raise OSError(f"cannot read tar member {member.name!r}")
            with source, output.open("wb") as handle:
                shutil.copyfileobj(source, handle, length=8 * 1024 * 1024)
            if output.stat().st_size != member.size:
                raise OSError(f"short extraction for {member.name!r}")
            extracted.append(
                {
                    "name": safe.as_posix(),
                    "bytes": member.size,
                    "sha256": sha256_file(output),
                }
            )
    manifest={
        "schema": "pdac-circuit.geo-archive-extraction/1",
        "archive_sha256": inventory["archive_sha256"],
        "unpacked_bytes": inventory["unpacked_bytes"],
        "files": extracted,
    }
    (temporary / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8"
    )
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary.replace(destination)
    manifest["destination"]=str(destination)
    return manifest
