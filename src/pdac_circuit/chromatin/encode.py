from __future__ import annotations

from dataclasses import asdict
from concurrent.futures import ThreadPoolExecutor, as_completed
import json
from pathlib import Path
import re
import time
from typing import Iterable
import uuid

from .streaming import TrackSpec, sha256_file

ENCODE="https://www.encodeproject.org"
ACCESSION_RE=re.compile(r"^(ENCFF[A-Z0-9]+)")

def _get_json(path: str) -> dict:
    import requests

    last_error=None
    for attempt in range(5):
        try:
            response=requests.get(
                f"{ENCODE}{path}",
                params={"format": "json"},
                headers={
                    "Accept": "application/json",
                    "User-Agent": "pdac-circuit/0.2 (RUO)",
                },
                timeout=90,
            )
            if response.status_code == 429 or response.status_code >= 500:
                response.raise_for_status()
            response.raise_for_status()
            return response.json()
        except requests.RequestException as exc:
            last_error=exc
            if attempt == 4:
                break
            time.sleep(min(8.0, 0.5 * (2**attempt)))
    raise RuntimeError(f"ENCODE API failed after retries for {path}: {last_error}")

def _atomic_json(path: Path, payload) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary=path.parent / f".{path.name}.partial-{uuid.uuid4().hex}"
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    temporary.replace(path)

def _label(value) -> str:
    if isinstance(value, dict):
        return str(value.get("label") or value.get("term_name") or value.get("@id") or "")
    if isinstance(value, str):
        return value.rstrip("/").split("/")[-1]
    return ""

def _accession(path_or_id: str) -> str:
    return path_or_id.rstrip("/").split("/")[-1]

def resolve_encode_accession(accession: str) -> dict:
    file_record=_get_json(f"/files/{accession}/")
    dataset_path=file_record.get("dataset")
    if not dataset_path:
        raise ValueError(f"ENCODE file {accession} has no dataset/experiment")
    experiment=_get_json(dataset_path)
    target=_label(experiment.get("target"))
    biosample=_label(experiment.get("biosample_ontology"))
    audit=file_record.get("audit", {})
    errors=audit.get("ERROR", []) if isinstance(audit, dict) else []
    warnings=audit.get("WARNING", []) if isinstance(audit, dict) else []
    return {
        "accession": accession,
        "dataset": _accession(dataset_path),
        "assay": str(experiment.get("assay_title") or file_record.get("assay_title") or ""),
        "biosample": biosample,
        "target": target,
        "output_type": str(file_record.get("output_type") or ""),
        "file_format": str(file_record.get("file_format") or ""),
        "biological_replicates": list(file_record.get("biological_replicates") or []),
        "technical_replicates": list(file_record.get("technical_replicates") or []),
        "released": str(file_record.get("date_released") or "")[:10],
        "status": str(file_record.get("status") or ""),
        "audit_errors": len(errors),
        "audit_warnings": len(warnings),
    }

def assay_vector(metadata: dict) -> tuple[float, ...]:

    vector=[0.0] * 12
    assay=metadata.get("assay", "").lower()
    target=metadata.get("target", "").upper()
    output=metadata.get("output_type", "").lower()
    if "atac" in assay or "dnase" in assay:
        vector[0]=1.0
    target_slots={
        "H3K27AC": 1,
        "H3K4ME1": 2,
        "H3K4ME3": 3,
        "H3K27ME3": 4,
        "H3K9ME3": 5,
    }
    if target in target_slots:
        vector[target_slots[target]] = 1.0
    if "tf chip" in assay or target == "CTCF":
        vector[6]=1.0
    if "rna" in assay:
        if "minus" in output or "reverse" in output:
            vector[8]=1.0
        elif "plus" in output or "forward" in output:
            vector[7]=1.0
        else:
            vector[7]=vector[8] = 0.5
    if "rampage" in assay or "cage" in assay:
        vector[9]=1.0
        if "minus" in output or "reverse" in output:
            vector[8]=1.0
        elif "plus" in output or "forward" in output:
            vector[7]=1.0
    if "wgbs" in assay or "methyl" in assay:
        vector[10]=1.0
    vector[11]=1.0 if metadata.get("status") == "released" and not metadata.get("audit_errors") else 0.5
    if not any(vector[:11]):
        raise ValueError(
            f"cannot map assay={metadata.get('assay')!r}, target={metadata.get('target')!r}, "
            f"output={metadata.get('output_type')!r}"
        )
    return tuple(vector)

def _healthy_selection_reason(metadata: dict, policy: dict) -> str | None:
    if metadata.get("status") != policy["required_status"]:
        return f"status is {metadata.get('status')!r}, not released"
    assay=str(metadata.get("assay") or "")
    output=str(metadata.get("output_type") or "")
    allowed_outputs=policy["canonical_outputs"].get(assay)
    if not allowed_outputs:
        if assay == "WGBS":
            return "CpG coverage is not a methylation-state target"
        return f"assay {assay!r} has no frozen condition semantics"
    if output not in allowed_outputs:
        return f"noncanonical output {output!r} for {assay}"
    if assay == "Histone ChIP-seq" and metadata.get("target") not in set(
        policy["supported_histone_targets"]
    ):
        return f"unsupported histone target {metadata.get('target')!r}"
    return None

def healthy_state_vector() -> tuple[float, ...]:
    state=[0.0] * 18
    state[0]=1.0
    state[11]=1.0
    state[15]=1.0
    state[16]=1.0
    return tuple(state)

def _artifacts(manifest_path: Path) -> Iterable[dict]:
    payload=json.loads(manifest_path.read_text(encoding="utf-8"))
    for artifact in payload.get("artifacts", []):
        match=ACCESSION_RE.match(str(artifact.get("name", "")))
        if match:
            yield {**artifact, "accession": match.group(1)}

def build_encode_track_specs(
    project_root: str | Path,
    *,
    refresh: bool = False,
    limit: int | None = None,
    metadata_workers: int = 8,
) -> dict:
    root=Path(project_root)
    manifest_path=root / "data" / "manifests" / "encode-bulk.heavy.json"
    policy_path=root / "configs" / "encode-healthy-selection-policy.json"
    cache_path=root / "data" / "metadata" / "encode_tracks.json"
    specs_dir=root / "data" / "track_specs" / "encode_healthy_pancreas"
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    specs_dir.mkdir(parents=True, exist_ok=True)
    policy=json.loads(policy_path.read_text(encoding="utf-8"))
    if policy.get("schema") != "pdac-circuit.encode-healthy-selection-policy/1":
        raise ValueError("unsupported ENCODE healthy selection policy")
    cache=(
        json.loads(cache_path.read_text(encoding="utf-8"))
        if cache_path.exists() and not refresh
        else {}
    )
    artifacts=[artifact for artifact in _artifacts(manifest_path) if artifact["name"].endswith(".bigWig")]
    if limit is not None:
        artifacts=artifacts[:limit]
    if metadata_workers < 1 or metadata_workers > 16:
        raise ValueError("metadata_workers must be in [1, 16]")
    failures, written, excluded = [], [], []
    resolution_failures={}
    missing_accessions=sorted(
        {
            artifact["accession"]
            for artifact in artifacts
            if refresh or artifact["accession"] not in cache
        }
    )
    if missing_accessions:
        completed_since_flush=0
        with ThreadPoolExecutor(max_workers=metadata_workers) as executor:
            futures={
                executor.submit(resolve_encode_accession, accession): accession
                for accession in missing_accessions
            }
            for future in as_completed(futures):
                accession=futures[future]
                try:
                    cache[accession]=future.result()
                    completed_since_flush += 1
                    if completed_since_flush >= 16:
                        _atomic_json(cache_path, cache)
                        completed_since_flush=0
                except Exception as exc:
                    resolution_failures[accession]=(
                        f"{type(exc).__name__}: {exc}"
                    )
        _atomic_json(cache_path, cache)
    for artifact in artifacts:
        accession=artifact["accession"]
        if accession in resolution_failures:
            failures.append(
                {"accession": accession, "error": resolution_failures[accession]}
            )
            continue
        try:
            metadata=cache[accession]
            exclusion_reason=_healthy_selection_reason(metadata, policy)
            if exclusion_reason:
                excluded.append(
                    {
                        "accession": accession,
                        "dataset": metadata.get("dataset"),
                        "assay": metadata.get("assay"),
                        "target": metadata.get("target"),
                        "output_type": metadata.get("output_type"),
                        "reason": exclusion_reason,
                    }
                )
                continue
            local_path=root / str(artifact["localPath"])
            spec=TrackSpec(
                accession=accession,
                path=str(local_path),
                assay_features=assay_vector(metadata),
                state_features=healthy_state_vector(),
                perturbation_features=(0.0,) * 22,
                sample_group=metadata["dataset"],
                study="ENCODE_HEALTHY_PANCREAS",
                released=metadata.get("released") or "",
                disease=False,
                source_sha256=artifact.get("sha256"),
                genome="hg38",
                organism="Homo sapiens",
                biological_state="healthy_pancreas",
            )
            spec.validate()
            path=specs_dir / f"{accession}.json"
            _atomic_json(path, asdict(spec))
            written.append(
                {
                    "accession": accession,
                    "spec": str(path.relative_to(root)),
                    "metadata": metadata,
                    "source_url": artifact.get("url"),
                }
            )
        except Exception as exc:
            failures.append(
                {"accession": accession, "error": f"{type(exc).__name__}: {exc}"}
            )
    _atomic_json(cache_path, cache)
    index={
        "schema": "pdac-circuit.encode-track-specs/1",
        "manifest": str(manifest_path.relative_to(root)),
        "selection_policy": str(policy_path.relative_to(root)),
        "selection_policy_sha256": sha256_file(policy_path),
        "candidate_tracks": len(artifacts),
        "written": written,
        "excluded": excluded,
        "failures": failures,
        "biological_groups": len({row["metadata"]["dataset"] for row in written}),
    }
    _atomic_json(specs_dir / "index.json", index)
    return index
