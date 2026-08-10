from __future__ import annotations

import json
from pathlib import Path
from typing import Iterator
import uuid

import numpy as np

from .streaming import sha256_file

def _row_count(paths: list[Path]) -> int:
    total=0
    for path in paths:
        with np.load(path, allow_pickle=False) as shard:
            total += len(shard["start"])
    return total

def _compile_contract(paths: list[Path]) -> dict:
    contracts=set()
    manifests=[]
    for parent in sorted({path.parent for path in paths}):
        manifest_path=parent / "manifest.json"
        if not manifest_path.is_file():
            raise FileNotFoundError(f"compiled source manifest missing: {manifest_path}")
        manifest=json.loads(manifest_path.read_text(encoding="utf-8"))
        contract=(
            manifest.get("sequence_length"),
            manifest.get("bin_size"),
            manifest.get("negative_keep_probability"),
        )
        if (
            manifest.get("schema") != "pdac-circuit.chromatin-shards/3"
            or not isinstance(contract[0], int)
            or not isinstance(contract[1], int)
            or not isinstance(contract[2], (int, float))
        ):
            raise ValueError(f"invalid compiled source manifest {manifest_path}")
        contracts.add(contract)
        manifests.append(
            {"path": str(manifest_path), "sha256": sha256_file(manifest_path)}
        )
    if len(contracts) != 1:
        raise ValueError(f"paired sources mix compile contracts: {sorted(contracts)}")
    sequence_length, bin_size, negative_keep_probability=next(iter(contracts))
    return {
        "sequence_length": sequence_length,
        "bin_size": bin_size,
        "negative_keep_probability": float(negative_keep_probability),
        "source_manifests": manifests,
    }

def _coordinate_key(chrom: str, start: int, end: int):
    label=chrom.removeprefix("chr")
    if label.isdigit():
        chromosome_order=(0, int(label))
    elif label in {"X", "Y", "M", "MT"}:
        chromosome_order=(0, {"X": 23, "Y": 24, "M": 25, "MT": 25}[label])
    else:
        chromosome_order=(1, label)
    return chromosome_order, start, end

def _iter_rows(paths: list[Path]) -> Iterator[dict]:
    previous=None
    for path in paths:
        with np.load(path, allow_pickle=False) as shard:
            for index in range(len(shard["start"])):
                coordinate=(
                    str(shard["chrom"][index]),
                    int(shard["start"][index]),
                    int(shard["end"][index]),
                )
                key=_coordinate_key(*coordinate)
                if previous is not None and key < previous:
                    raise ValueError(f"shards are not coordinate-sorted near {path}")
                previous=key
                row={"_key": key, "_coordinate": coordinate, "_source": str(path)}
                for field in shard.files:
                    row[field]=shard[field][index].copy()
                yield row

def compose_paired_shards(
    reference_paths: list[str | Path],
    treatment_paths: list[str | Path],
    output_dir: str | Path,
    *,
    mode: str,
    windows_per_shard: int = 64,
    minimum_overlap_fraction: float = 0.80,
    pair_id: str | None = None,
) -> dict:

    if mode not in {"state", "perturbation"}:
        raise ValueError("pair mode must be state or perturbation")
    if windows_per_shard < 1:
        raise ValueError("windows_per_shard must be positive")
    if not 0 < minimum_overlap_fraction <= 1:
        raise ValueError("minimum_overlap_fraction must be in (0, 1]")
    reference_paths=sorted(Path(path) for path in reference_paths)
    treatment_paths=sorted(Path(path) for path in treatment_paths)
    if not reference_paths or not treatment_paths:
        raise ValueError("both reference and treatment shard lists are required")
    reference_contract=_compile_contract(reference_paths)
    treatment_contract=_compile_contract(treatment_paths)
    contract_keys=("sequence_length", "bin_size", "negative_keep_probability")
    if any(reference_contract[key] != treatment_contract[key] for key in contract_keys):
        raise ValueError("reference and treatment compile contracts differ")
    destination=Path(output_dir)
    if destination.exists():
        raise FileExistsError(f"paired destination already exists: {destination}")
    temporary=destination.with_name(f"{destination.name}.partial-{uuid.uuid4().hex}")
    temporary.mkdir(parents=True)
    reference_total=_row_count(reference_paths)
    treatment_total=_row_count(treatment_paths)
    reference_iterator=iter(_iter_rows(reference_paths))
    treatment_iterator=iter(_iter_rows(treatment_paths))
    reference=next(reference_iterator, None)
    treatment=next(treatment_iterator, None)
    conditioning_dimensions={
        key: int(np.asarray(treatment[key]).size)
        for key in ("assay_features", "state_features", "perturbation_features")
    } if treatment is not None else {}
    batch, shard_records=[], []
    matches=reference_only=treatment_only=0

    def flush() -> None:
        nonlocal batch
        if not batch:
            return
        path=temporary / f"shard-{len(shard_records):06d}.npz"
        payload={}
        for field in (
            "example_id",
            "accession",
            "sample_group",
            "study",
            "genome",
            "organism",
            "pair_group",
            "pair_relation",
            "split",
            "chrom",
            "start",
            "end",
            "target",
            "valid",
            "assay_features",
            "state_features",
            "perturbation_features",
            "disease_mask",
            "healthy_mask",
            "paired_delta",
            "pair_mask",
            "perturbation_delta",
            "perturbation_mask",
        ):
            values=[row[field] for row in batch if field in row]
            if len(values) == len(batch):
                payload[field]=np.asarray(values)
        np.savez_compressed(path, **payload)
        shard_records.append(
            {
                "path": path.name,
                "examples": len(batch),
                "bytes": path.stat().st_size,
                "sha256": sha256_file(path),
            }
        )
        batch=[]

    while reference is not None and treatment is not None:
        if reference["_key"] < treatment["_key"]:
            reference_only += 1
            reference=next(reference_iterator, None)
            continue
        if treatment["_key"] < reference["_key"]:
            treatment_only += 1
            treatment=next(treatment_iterator, None)
            continue
        matches += 1
        reference_target=np.asarray(reference["target"], dtype=np.float32)
        treatment_target=np.asarray(treatment["target"], dtype=np.float32)
        reference_genome=str(reference.get("genome", "hg38"))
        treatment_genome=str(treatment.get("genome", "hg38"))
        if reference_genome != treatment_genome:
            raise ValueError(
                f"cannot pair {reference_genome} and {treatment_genome} at "
                f"{reference['_coordinate']}"
            )
        reference_pair_group=str(reference.get("pair_group", ""))
        treatment_pair_group=str(treatment.get("pair_group", ""))
        if not reference_pair_group or reference_pair_group != treatment_pair_group:
            raise ValueError(
                f"pair_group mismatch at {reference['_coordinate']}: "
                f"{reference_pair_group!r} != {treatment_pair_group!r}"
            )
        expected_relations=(
            ("state_reference", "state_treatment")
            if mode == "state"
            else ("control", "intervention")
        )
        observed_relations=(
            str(reference.get("pair_relation", "")),
            str(treatment.get("pair_relation", "")),
        )
        if observed_relations != expected_relations:
            raise ValueError(
                f"pair relations at {reference['_coordinate']} are {observed_relations}; "
                f"require {expected_relations}"
            )
        if reference_target.shape != treatment_target.shape:
            raise ValueError(f"paired target shapes differ at {reference['_coordinate']}")
        reference_valid=np.asarray(reference["valid"], dtype=bool)
        treatment_valid=np.asarray(treatment["valid"], dtype=bool)
        pair_mask=reference_valid & treatment_valid
        chrom, start, end=treatment["_coordinate"]
        accession=str(treatment.get("accession", "treatment"))
        row={
            "example_id": f"{accession}:{chrom}:{start}:{end}",
            "accession": accession,
            "sample_group": str(treatment.get("sample_group", accession)),
            "study": str(treatment.get("study", "paired_study")),
            "genome": treatment_genome,
            "organism": str(treatment.get("organism", "Homo sapiens")),
            "pair_group": treatment_pair_group,
            "pair_relation": "state_treatment" if mode == "state" else "intervention",
            "split": str(treatment.get("split", "train")),
            "chrom": chrom,
            "start": start,
            "end": end,
            "target": treatment_target.astype(np.float16),
            "valid": treatment_valid.astype(np.uint8),
            "assay_features": np.asarray(treatment["assay_features"], dtype=np.float16),
            "state_features": np.asarray(treatment["state_features"], dtype=np.float16),
            "perturbation_features": np.asarray(
                treatment["perturbation_features"], dtype=np.float16
            ),
            "disease_mask": np.uint8(treatment["disease_mask"]),
            "healthy_mask": np.uint8(0),
        }
        delta=(treatment_target - reference_target).astype(np.float16)
        if mode == "state":
            row["paired_delta"]=delta
            row["pair_mask"]=pair_mask.astype(np.uint8)
        else:
            row["perturbation_delta"]=delta
            row["perturbation_mask"]=pair_mask.astype(np.uint8)
        batch.append(row)
        if len(batch) >= windows_per_shard:
            flush()
        reference=next(reference_iterator, None)
        treatment=next(treatment_iterator, None)
    reference_only += int(reference is not None) + sum(1 for _ in reference_iterator)
    treatment_only += int(treatment is not None) + sum(1 for _ in treatment_iterator)
    flush()
    denominator=max(1, min(reference_total, treatment_total))
    overlap_fraction=matches / denominator
    manifest={
        "schema": "pdac-circuit.paired-chromatin-shards/1",
        "pair_id": pair_id,
        "mode": mode,
        "valid": overlap_fraction >= minimum_overlap_fraction and matches > 0,
        "reference_examples": reference_total,
        "treatment_examples": treatment_total,
        "matched_examples": matches,
        "reference_only": reference_only,
        "treatment_only": treatment_only,
        "overlap_fraction": overlap_fraction,
        "minimum_overlap_fraction": minimum_overlap_fraction,
        "sequence_length": reference_contract["sequence_length"],
        "bin_size": reference_contract["bin_size"],
        "negative_keep_probability": reference_contract[
            "negative_keep_probability"
        ],
        "conditioning_dimensions": conditioning_dimensions,
        "source_manifests": {
            "reference": reference_contract["source_manifests"],
            "treatment": treatment_contract["source_manifests"],
        },
        "reference_shards": [
            {"path": str(path), "sha256": sha256_file(path)} for path in reference_paths
        ],
        "treatment_shards": [
            {"path": str(path), "sha256": sha256_file(path)} for path in treatment_paths
        ],
        "shards": shard_records,
    }
    (temporary / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8"
    )
    if not manifest["valid"]:
        raise RuntimeError(
            f"paired overlap {overlap_fraction:.3f} is below {minimum_overlap_fraction:.3f}; "
            f"invalid partial output retained at {temporary}"
        )
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary.replace(destination)
    manifest["output_dir"]=str(destination)
    return manifest
