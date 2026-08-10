from __future__ import annotations

import hashlib
import json
from pathlib import Path
import uuid

import numpy as np

from .benchmark import PredictionBundle, save_prediction_bundle
from .provenance import write_prediction_manifest
from .streaming import sha256_file

RAW_REQUIRED={"model", "example_id", "prediction", "metadata"}
TRUTH_REQUIRED={"example_id", "target", "group", "split"}

def _example_id_sha256(values) -> str:
    digest=hashlib.sha256()
    for value in sorted(str(item) for item in values):
        digest.update(value.encode("utf-8"))
        digest.update(b"\n")
    return digest.hexdigest()

def freeze_evaluation_windows(
    shard_paths: list[str | Path],
    windows_out: str | Path,
    conditions_out: str | Path,
    *,
    split: str,
    output_bins: int = 896,
    genomes: set[str] | frozenset[str] | None = None,
    context_length: int | None = None,
    chrom_sizes: dict[str, dict[str, int]] | None = None,
    example_ids: set[str] | frozenset[str] | None = None,
    max_examples_per_condition_group: int | None = None,
    sampling_seed: int = 20_260_715,
) -> dict:

    required={
        "example_id",
        "genome",
        "chrom",
        "start",
        "end",
        "split",
        "sample_group",
        "assay_features",
        "state_features",
        "perturbation_features",
    }
    if context_length is not None and context_length < 1:
        raise ValueError("context_length must be positive")
    allowed_ids=frozenset(example_ids) if example_ids is not None else None
    if allowed_ids is not None and not allowed_ids:
        raise ValueError("example_ids cannot be empty")
    if max_examples_per_condition_group is not None:
        if max_examples_per_condition_group < 1:
            raise ValueError("max_examples_per_condition_group must be positive")
        if allowed_ids is not None:
            raise ValueError("cannot combine an exact example-ID cohort with cohort sampling")
    records=[]
    sources=[]
    dropped_edge_windows=0
    for value in sorted(Path(path) for path in shard_paths):
        sources.append({"path": str(value), "sha256": sha256_file(value)})
        with np.load(value, allow_pickle=False) as shard:
            missing=sorted(required - set(shard.files))
            if missing:
                raise ValueError(f"{value.name} cannot freeze baseline windows; missing {missing}")
            for index in range(len(shard["example_id"])):
                if str(shard["split"][index]) != split:
                    continue
                if genomes is not None and str(shard["genome"][index]) not in genomes:
                    continue
                example_id=str(shard["example_id"][index])
                if allowed_ids is not None and example_id not in allowed_ids:
                    continue
                genome=str(shard["genome"][index])
                chrom=str(shard["chrom"][index])
                source_start=int(shard["start"][index])
                source_end=int(shard["end"][index])
                if source_end <= source_start:
                    raise ValueError(f"invalid source interval for {example_id}")
                target_length=context_length or source_end - source_start
                if ((source_start + source_end) - target_length) % 2:
                    raise ValueError(
                        f"cannot center an even-base context exactly on {example_id}"
                    )
                start=(source_start + source_end - target_length) // 2
                end=start + target_length
                if target_length != source_end - source_start:
                    if chrom_sizes is None or genome not in chrom_sizes:
                        raise ValueError(
                            f"context expansion requires chromosome sizes for {genome}"
                        )
                    if chrom not in chrom_sizes[genome]:
                        raise ValueError(f"chromosome sizes for {genome} lack {chrom}")
                    if start < 0 or end > int(chrom_sizes[genome][chrom]):
                        dropped_edge_windows += 1
                        continue
                records.append(
                    {
                        "example_id": example_id,
                        "genome": genome,
                        "chrom": chrom,
                        "start": start,
                        "end": end,
                        "source_start": source_start,
                        "source_end": source_end,
                        "group": str(shard["sample_group"][index]),
                        "assay_features": shard["assay_features"][index].astype(np.float32),
                        "state_features": shard["state_features"][index].astype(np.float32),
                        "perturbation_features": shard["perturbation_features"][index].astype(
                            np.float32
                        ),
                    }
                )
    if not records:
        raise RuntimeError(f"no shard examples are labeled {split!r}")
    eligible_records=len(records)
    sampling_strata=0
    if max_examples_per_condition_group is not None:
        strata: dict[str, list[dict]]={}
        for record in records:
            condition_digest=hashlib.sha256()
            for key in (
                "assay_features",
                "state_features",
                "perturbation_features",
            ):
                condition_digest.update(
                    np.asarray(record[key], dtype=np.float32).tobytes()
                )
            stratum=f"{record['group']}::{condition_digest.hexdigest()}"
            strata.setdefault(stratum, []).append(record)
        selected=[]
        for stratum, members in sorted(strata.items()):
            members.sort(
                key=lambda record: hashlib.sha256(
                    f"{sampling_seed}:{record['example_id']}".encode()
                ).hexdigest()
            )
            selected.extend(members[:max_examples_per_condition_group])
        records=selected
        sampling_strata=len(strata)
    records.sort(key=lambda record: record["example_id"])
    example_ids=[record["example_id"] for record in records]
    if len(set(example_ids)) != len(example_ids):
        raise ValueError("evaluation windows contain duplicate example IDs")
    if allowed_ids is not None and set(example_ids) != set(allowed_ids):
        missing=sorted(set(allowed_ids) - set(example_ids))
        raise ValueError(
            f"frozen evaluation cohort omitted {len(missing)} requested IDs; "
            f"first missing IDs: {missing[:5]}"
        )
    lengths={record["end"] - record["start"] for record in records}
    if len(lengths) != 1 or min(lengths) <= 0:
        raise ValueError(f"evaluation windows do not share one positive context length: {lengths}")
    dimensions={
        key: {record[key].shape for record in records}
        for key in ("assay_features", "state_features", "perturbation_features")
    }
    if any(len(shapes) != 1 for shapes in dimensions.values()):
        raise ValueError(f"condition dimensions differ across evaluation windows: {dimensions}")
    if output_bins < 1:
        raise ValueError("output_bins must be positive")

    windows_payload={
        "schema": "pdac-circuit.evaluation-windows/1",
        "split": split,
        "genome_filter": sorted(genomes) if genomes is not None else None,
        "sequence_length": next(iter(lengths)),
        "source_sequence_lengths": sorted(
            {record["source_end"] - record["source_start"] for record in records}
        ),
        "context_expanded": any(
            record["start"] != record["source_start"]
            or record["end"] != record["source_end"]
            for record in records
        ),
        "dropped_edge_windows": dropped_edge_windows,
        "cohort_filter_applied": allowed_ids is not None,
        "label_free_sampling": (
            {
                "method": "sha256_rank_within_group_and_exact_condition_vector",
                "seed": int(sampling_seed),
                "max_examples_per_condition_group": max_examples_per_condition_group,
                "eligible_examples": eligible_records,
                "selected_examples": len(records),
                "strata": sampling_strata,
                "signal_access": False,
            }
            if max_examples_per_condition_group is not None
            else None
        ),
        "output_bins": output_bins,
        "example_id_sha256": _example_id_sha256(example_ids),
        "source_shards": sources,
        "examples": [
            {
                key: record[key]
                for key in (
                    "example_id",
                    "genome",
                    "chrom",
                    "start",
                    "end",
                    "source_start",
                    "source_end",
                )
            }
            for record in records
        ],
    }
    windows_path=Path(windows_out)
    windows_path.parent.mkdir(parents=True, exist_ok=True)
    windows_temp=windows_path.parent / f".{windows_path.name}.partial-{uuid.uuid4().hex}"
    windows_temp.write_text(
        json.dumps(windows_payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    windows_temp.replace(windows_path)
    windows_sha=sha256_file(windows_path)

    conditions_path=Path(conditions_out)
    conditions_path.parent.mkdir(parents=True, exist_ok=True)
    conditions_temp=(
        conditions_path.parent
        / f".{conditions_path.stem}.partial-{uuid.uuid4().hex}.npz"
    )
    metadata={
        "schema": "pdac-circuit.baseline-conditions/1",
        "split": split,
        "windows_sha256": windows_sha,
        "contains_targets": False,
        "candidate_feature_access": False,
        "sequence_length": next(iter(lengths)),
        "cohort_filter_applied": allowed_ids is not None,
        "label_free_sampling": windows_payload["label_free_sampling"],
        "example_id_sha256": windows_payload["example_id_sha256"],
    }
    np.savez_compressed(
        conditions_temp,
        example_id=np.asarray(example_ids),
        group=np.asarray([record["group"] for record in records]),
        split=np.repeat(split, len(records)),
        assay_features=np.stack([record["assay_features"] for record in records]),
        state_features=np.stack([record["state_features"] for record in records]),
        perturbation_features=np.stack(
            [record["perturbation_features"] for record in records]
        ),
        metadata=np.asarray(json.dumps(metadata, sort_keys=True)),
    )
    conditions_temp.replace(conditions_path)
    return {
        "windows": str(windows_path),
        "conditions": str(conditions_path),
        "examples": len(records),
        "genomes": sorted({record["genome"] for record in records}),
        "sequence_length": next(iter(lengths)),
        "output_bins": output_bins,
        "dropped_edge_windows": dropped_edge_windows,
        "cohort_filter_applied": allowed_ids is not None,
        "label_free_sampling": windows_payload["label_free_sampling"],
        "windows_sha256": windows_sha,
        "conditions_sha256": sha256_file(conditions_path),
        "contains_targets": False,
        "example_id_sha256": windows_payload["example_id_sha256"],
    }

def save_raw_predictions(
    path: str | Path,
    *,
    model: str,
    example_id: np.ndarray,
    prediction: np.ndarray,
    metadata: dict,
    lower: np.ndarray | None = None,
    upper: np.ndarray | None = None,
) -> None:
    if len(example_id) == 0 or len(example_id) != len(prediction):
        raise ValueError("raw prediction IDs/predictions are empty or misaligned")
    if len(set(example_id.astype(str))) != len(example_id):
        raise ValueError("raw example IDs must be unique")
    if not np.isfinite(prediction).all():
        raise ValueError("raw predictions must be finite")
    if (lower is None) != (upper is None):
        raise ValueError("raw lower/upper intervals must be supplied together")
    if lower is not None and (lower.shape != prediction.shape or upper.shape != prediction.shape):
        raise ValueError("raw interval shapes differ from predictions")
    metadata=dict(metadata)
    cohort_sha=_example_id_sha256(np.asarray(example_id).astype(str))
    if metadata.get("example_id_sha256") not in {None, cohort_sha}:
        raise ValueError("raw prediction metadata has a mismatched example-ID hash")
    metadata["example_id_sha256"]=cohort_sha
    payload={
        "model": np.asarray(model),
        "example_id": np.asarray(example_id),
        "prediction": np.asarray(prediction),
        "metadata": np.asarray(json.dumps(metadata, sort_keys=True)),
    }
    if lower is not None:
        payload["lower"]=np.asarray(lower)
        payload["upper"]=np.asarray(upper)
    destination=Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary=destination.parent / f".{destination.stem}.partial-{uuid.uuid4().hex}.npz"
    np.savez_compressed(temporary, **payload)
    temporary.replace(destination)

def merge_raw_predictions(
    paths: list[str | Path], out: str | Path, *, command: str
) -> dict:

    if len(paths) < 1:
        raise ValueError("at least one raw prediction file is required")
    rows=[]
    sources=[]
    identity=None
    model_name=None
    trailing_shape=None
    for path in sorted(Path(value) for value in paths):
        payload=_load_npz(path, RAW_REQUIRED)
        metadata=json.loads(str(payload["metadata"].item()))
        if metadata.get("example_id_sha256") != _example_id_sha256(
            payload["example_id"].astype(str)
        ):
            raise ValueError(f"{path.name} label-free cohort hash drifted")
        current_model=str(payload["model"].item())
        current_identity=tuple(
            str(metadata.get(key, ""))
            for key in ("model_version", "weights_sha256", "track_mapping_sha256")
        )
        if not all(current_identity):
            raise ValueError(f"{path.name} lacks frozen baseline identity metadata")
        if identity is None:
            identity=current_identity
            model_name=current_model
            trailing_shape=payload["prediction"].shape[1:]
        elif current_identity != identity or current_model != model_name:
            raise ValueError("raw prediction sources use different model/weight/mapping identities")
        if payload["prediction"].shape[1:] != trailing_shape:
            raise ValueError("raw prediction sources have different profile shapes")
        if len(payload["example_id"]) != len(payload["prediction"]):
            raise ValueError(f"{path.name} has misaligned example IDs and predictions")
        for index, example_id in enumerate(payload["example_id"].astype(str)):
            rows.append((example_id, payload["prediction"][index].astype(np.float32)))
        sources.append(
            {
                "path": str(path),
                "sha256": sha256_file(path),
                "target_rule": metadata.get("target_rule"),
            }
        )
    rows.sort(key=lambda row: row[0])
    example_ids=np.asarray([row[0] for row in rows])
    if len(set(example_ids)) != len(example_ids):
        raise ValueError("raw prediction sources contain duplicate example IDs")
    metadata={
        "schema": "pdac-circuit.raw-predictions/1",
        "model": model_name,
        "model_version": identity[0],
        "weights_sha256": identity[1],
        "track_mapping_sha256": identity[2],
        "sources": sources,
        "target_rules": sorted(
            {str(source["target_rule"]) for source in sources if source["target_rule"]}
        ),
        "command": command,
    }
    metadata["example_id_sha256"]=_example_id_sha256(example_ids)
    save_raw_predictions(
        out,
        model=model_name,
        example_id=example_ids,
        prediction=np.stack([row[1] for row in rows]),
        metadata=metadata,
    )
    return {
        "out": str(out),
        "examples": len(rows),
        "shape": [len(rows), *trailing_shape],
        "sources": len(sources),
        "sha256": sha256_file(out),
        **metadata,
    }

def contrast_raw_predictions(
    reference_path: str | Path,
    treatment_path: str | Path,
    out: str | Path,
    *,
    mode: str,
    command: str,
) -> dict:

    if mode not in {"state", "perturbation"}:
        raise ValueError("raw contrast mode must be state or perturbation")
    reference=_load_npz(reference_path, RAW_REQUIRED)
    treatment=_load_npz(treatment_path, RAW_REQUIRED)
    reference_metadata=json.loads(str(reference["metadata"].item()))
    treatment_metadata=json.loads(str(treatment["metadata"].item()))
    reference_model=str(reference["model"].item())
    treatment_model=str(treatment["model"].item())
    if reference_model != treatment_model:
        raise ValueError("raw contrast inputs use different models")
    identity_fields=(
        "model_version",
        "weights_sha256",
        "track_mapping_sha256",
        "seed",
        "crop_bins",
        "reverse_complement",
        "local_tiling",
        "label_free_cohort",
    )
    drift=[
        field
        for field in identity_fields
        if reference_metadata.get(field) != treatment_metadata.get(field)
    ]
    if drift:
        raise ValueError(f"raw contrast identity differs for fields: {drift}")
    ablation_field=(
        "ablate_intervention_residual" if mode == "perturbation" else "ablate_state_residual"
    )
    if (
        reference_metadata.get(ablation_field) is not True
        or treatment_metadata.get(ablation_field) is not False
    ):
        raise ValueError(
            f"{mode} contrast requires reference {ablation_field}=true and treatment=false"
        )
    reference_ids=reference["example_id"].astype(str)
    treatment_ids=treatment["example_id"].astype(str)
    if (
        len(set(reference_ids)) != len(reference_ids)
        or len(set(treatment_ids)) != len(treatment_ids)
        or set(reference_ids) != set(treatment_ids)
    ):
        raise ValueError("raw contrast inputs require identical unique example IDs")
    treatment_index={value: index for index, value in enumerate(treatment_ids)}
    order=np.asarray([treatment_index[value] for value in reference_ids])
    reference_prediction=reference["prediction"].astype(np.float32)
    treatment_prediction=treatment["prediction"][order].astype(np.float32)
    if reference_prediction.shape != treatment_prediction.shape:
        raise ValueError("raw contrast prediction shapes differ")
    prediction=treatment_prediction - reference_prediction
    metadata={
        **{
            field: treatment_metadata.get(field)
            for field in identity_fields
            if field in treatment_metadata
        },
        "schema": "pdac-circuit.raw-predictions/1",
        "model": reference_model,
        "component": f"{mode}_delta",
        "ablate_state_residual": False,
        "ablate_intervention_residual": False,
        "contrast": {
            "mode": mode,
            "operation": "treatment_minus_reference",
            "reference_raw": str(reference_path),
            "reference_sha256": sha256_file(reference_path),
            "treatment_raw": str(treatment_path),
            "treatment_sha256": sha256_file(treatment_path),
            "reference_ablation": ablation_field,
            "exact_example_ids": True,
        },
        "command": command,
    }
    save_raw_predictions(
        out,
        model=reference_model,
        example_id=reference_ids,
        prediction=prediction,
        metadata=metadata,
    )
    return {
        "out": str(out),
        "model": reference_model,
        "mode": mode,
        "examples": len(reference_ids),
        "shape": list(prediction.shape),
        "sha256": sha256_file(out),
    }

def ensemble_seed_raw_predictions(
    paths: list[str | Path],
    out: str | Path,
    *,
    registered_seeds: list[int] | tuple[int, ...],
    command: str,
) -> dict:

    expected_seeds=tuple(int(seed) for seed in registered_seeds)
    if len(expected_seeds) < 3 or len(set(expected_seeds)) != len(expected_seeds):
        raise ValueError("seed ensemble requires at least three unique registered seeds")
    if len(paths) != len(expected_seeds):
        raise ValueError(
            f"seed ensemble received {len(paths)} files; require {len(expected_seeds)}"
        )
    identity=None
    model_name=None
    reference_ids=None
    reference_shape=None
    components={}
    aligned_predictions=[]
    identity_fields=(
        "track_mapping_sha256",
        "component",
        "crop_bins",
        "reverse_complement",
        "ablate_state_residual",
        "ablate_intervention_residual",
        "local_tiling",
        "label_free_cohort",
    )
    for path in sorted(Path(value) for value in paths):
        payload=_load_npz(path, RAW_REQUIRED)
        if "lower" in payload or "upper" in payload:
            raise ValueError("seed predictions must be ensembled before conformal intervals")
        metadata=json.loads(str(payload["metadata"].item()))
        try:
            seed=int(metadata["seed"])
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError(f"{path.name} lacks an integer campaign seed") from exc
        if seed not in expected_seeds:
            raise ValueError(f"{path.name} seed {seed} is not preregistered")
        if seed in components:
            raise ValueError(f"duplicate seed {seed} in ensemble inputs")
        current_model=str(payload["model"].item())
        current_identity=tuple(
            json.dumps(metadata.get(field), sort_keys=True, separators=(",", ":"))
            for field in identity_fields
        )
        weights_sha256=str(metadata.get("weights_sha256", ""))
        model_version=str(metadata.get("model_version", ""))
        if (
            len(weights_sha256) != 64
            or any(char not in "0123456789abcdef" for char in weights_sha256.lower())
            or not model_version
        ):
            raise ValueError(f"{path.name} lacks a frozen candidate weight identity")
        ids=payload["example_id"].astype(str)
        if len(ids) == 0 or len(set(ids)) != len(ids):
            raise ValueError(f"{path.name} has empty or duplicate example IDs")
        prediction=payload["prediction"].astype(np.float64)
        if len(prediction) != len(ids) or not np.isfinite(prediction).all():
            raise ValueError(f"{path.name} predictions are misaligned or non-finite")
        if identity is None:
            identity=current_identity
            model_name=current_model
            reference_ids=ids.copy()
            reference_shape=prediction.shape[1:]
        elif current_identity != identity or current_model != model_name:
            raise ValueError("seed predictions differ in model, cohort, mapping, or inference policy")
        if prediction.shape[1:] != reference_shape or set(ids) != set(reference_ids):
            raise ValueError("seed predictions do not share exact example IDs and output shape")
        index={value: index for index, value in enumerate(ids)}
        order=np.asarray([index[value] for value in reference_ids])
        aligned_predictions.append((seed, prediction[order]))
        components[seed]={
            "seed": seed,
            "raw_path": str(path),
            "raw_sha256": sha256_file(path),
            "weights_sha256": weights_sha256,
            "model_version": model_version,
        }
    if set(components) != set(expected_seeds):
        raise ValueError(
            f"seed ensemble observed {sorted(components)}; require {sorted(expected_seeds)}"
        )
    ordered_components=[components[seed] for seed in expected_seeds]
    identity_components=[
        {
            key: component[key]
            for key in ("seed", "raw_sha256", "weights_sha256", "model_version")
        }
        for component in ordered_components
    ]
    ensemble_identity=hashlib.sha256(
        json.dumps(identity_components, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    seed_ensemble={
        "schema": "pdac-circuit.seed-ensemble/1",
        "registered_seeds": list(expected_seeds),
        "aggregation": "arithmetic_mean",
        "exact_example_ids": True,
        "components": ordered_components,
    }
    metadata={
        "schema": "pdac-circuit.raw-predictions/1",
        "model": model_name,
        "model_version": f"three-seed-ensemble-sha256:{ensemble_identity}",
        "weights_sha256": ensemble_identity,
        "track_mapping_sha256": json.loads(identity[0]),
        "component": json.loads(identity[1]),
        "crop_bins": json.loads(identity[2]),
        "reverse_complement": json.loads(identity[3]),
        "ablate_state_residual": json.loads(identity[4]),
        "ablate_intervention_residual": json.loads(identity[5]),
        "local_tiling": json.loads(identity[6]),
        "label_free_cohort": json.loads(identity[7]),
        "seed_ensemble": seed_ensemble,
        "command": command,
    }
    prediction_by_seed={seed: prediction for seed, prediction in aligned_predictions}
    mean_prediction=np.mean(
        np.stack([prediction_by_seed[seed] for seed in expected_seeds]), axis=0
    ).astype(np.float32)
    save_raw_predictions(
        out,
        model=model_name,
        example_id=reference_ids,
        prediction=mean_prediction,
        metadata=metadata,
    )
    return {
        "out": str(out),
        "model": model_name,
        "examples": len(reference_ids),
        "shape": list(mean_prediction.shape),
        "registered_seeds": list(expected_seeds),
        "weights_sha256": ensemble_identity,
        "sha256": sha256_file(out),
    }

def _load_npz(path: str | Path, required: set[str]) -> dict:
    with np.load(path, allow_pickle=False) as archive:
        missing=sorted(required - set(archive.files))
        if missing:
            raise ValueError(f"{Path(path).name} missing keys: {missing}")
        return {key: archive[key].copy() for key in archive.files}

def freeze_profile_truth(
    shard_paths: list[str | Path],
    out: str | Path,
    *,
    split: str,
    crop_bins: int | None = 896,
    genomes: set[str] | frozenset[str] | None = None,
    example_ids: set[str] | frozenset[str] | None = None,
    target_field: str = "target",
    mask_field: str = "valid",
) -> dict:

    field_pairs={
        "target": "valid",
        "paired_delta": "pair_mask",
        "perturbation_delta": "perturbation_mask",
    }
    if field_pairs.get(target_field) != mask_field:
        raise ValueError(
            f"truth field {target_field!r} requires mask {field_pairs.get(target_field)!r}"
        )
    allowed_ids=frozenset(example_ids) if example_ids is not None else None
    if allowed_ids is not None and not allowed_ids:
        raise ValueError("example_ids cannot be empty")
    records=[]
    sources=[]
    for value in sorted(Path(path) for path in shard_paths):
        sources.append({"path": str(value), "sha256": sha256_file(value)})
        with np.load(value, allow_pickle=False) as shard:
            for index in range(len(shard["start"])):
                chrom=str(shard["chrom"][index])
                row_split=(
                    str(shard["split"][index])
                    if "split" in shard.files
                    else "validation"
                    if chrom in {"chr6", "chr7"}
                    else "locus_test"
                    if chrom in {"chr8", "chr9"}
                    else "train"
                )
                if row_split != split:
                    continue
                row_genome=(
                    str(shard["genome"][index])
                    if "genome" in shard.files
                    else "hg38"
                )
                if genomes is not None and row_genome not in genomes:
                    continue
                accession=(
                    str(shard["accession"][index])
                    if "accession" in shard.files
                    else value.parent.name
                )
                start, end=int(shard["start"][index]), int(shard["end"][index])
                example_id=(
                    str(shard["example_id"][index])
                    if "example_id" in shard.files
                    else f"{accession}:{chrom}:{start}:{end}"
                )
                if allowed_ids is not None and example_id not in allowed_ids:
                    continue
                if target_field not in shard.files or mask_field not in shard.files:
                    raise ValueError(
                        f"{value} lacks truth fields {target_field!r}/{mask_field!r}"
                    )
                target=shard[target_field][index].astype(np.float32)
                valid=shard[mask_field][index].astype(bool)
                if crop_bins is not None:
                    if crop_bins < 1 or crop_bins > target.shape[-1]:
                        raise ValueError(
                            f"crop_bins={crop_bins} is invalid for {target.shape[-1]} bins"
                        )
                    trim=(target.shape[-1] - crop_bins) // 2
                    target=target[trim : trim + crop_bins]
                    valid=valid[trim : trim + crop_bins]
                records.append(
                    (
                        example_id,
                        target,
                        valid,
                        str(shard["sample_group"][index])
                        if "sample_group" in shard.files
                        else accession,
                    )
                )
    if not records:
        raise RuntimeError(f"no shard examples are labeled {split!r}")
    records.sort(key=lambda record: record[0])
    example_ids=np.asarray([record[0] for record in records])
    if len(set(example_ids)) != len(example_ids):
        raise ValueError("profile truth contains duplicate example IDs")
    if allowed_ids is not None and set(example_ids.astype(str)) != set(allowed_ids):
        missing=sorted(set(allowed_ids) - set(example_ids.astype(str)))
        raise ValueError(
            f"profile truth omitted {len(missing)} requested IDs; first missing IDs: {missing[:5]}"
        )
    target=np.stack([record[1] for record in records])
    mask=np.stack([record[2] for record in records])
    examples_without_valid_bins=int((~mask.any(axis=1)).sum())
    metadata={
        "schema": "pdac-circuit.profile-truth/1",
        "split": split,
        "crop_bins": crop_bins,
        "genome_filter": sorted(genomes) if genomes is not None else None,
        "cohort_filter_applied": allowed_ids is not None,
        "target_field": target_field,
        "mask_field": mask_field,
        "source_shards": sources,
    }
    destination=Path(out)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary=destination.parent / f".{destination.stem}.partial-{uuid.uuid4().hex}.npz"
    np.savez_compressed(
        temporary,
        example_id=example_ids,
        target=target,
        mask=mask.astype(np.uint8),
        group=np.asarray([record[3] for record in records]),
        split=np.repeat(split, len(records)),
        metadata=np.asarray(json.dumps(metadata, sort_keys=True)),
    )
    temporary.replace(destination)
    return {
        "out": str(destination),
        "examples": len(records),
        "shape": list(target.shape),
        "valid_fraction": float(mask.mean()),
        "examples_without_valid_bins": examples_without_valid_bins,
        "sha256": sha256_file(destination),
    }

def make_state_invariant_raw(
    truth_path: str | Path,
    out: str | Path,
    *,
    model: str,
    model_version: str,
    weights_sha256: str,
    track_mapping_sha256: str,
    command: str,
) -> dict:

    truth=_load_npz(truth_path, TRUTH_REQUIRED)
    prediction=np.zeros_like(truth["target"], dtype=np.float32)
    metadata={
        "schema": "pdac-circuit.raw-predictions/1",
        "model": model,
        "model_version": model_version,
        "weights_sha256": weights_sha256,
        "track_mapping_sha256": track_mapping_sha256,
        "state_invariant": True,
        "construction": "zero disease/subtype/perturbation contrast for identical DNA sequence",
        "command": command,
    }
    save_raw_predictions(
        out,
        model=model,
        example_id=truth["example_id"],
        prediction=prediction,
        metadata=metadata,
    )
    return metadata

def conformalize_raw_predictions(
    calibration_raw_path: str | Path,
    calibration_truth_path: str | Path,
    target_raw_path: str | Path,
    out: str | Path,
    *,
    nominal: float = 0.90,
    command: str,
) -> dict:

    if not 0 < nominal < 1:
        raise ValueError("nominal conformal coverage must be in (0, 1)")
    calibration_raw=_load_npz(calibration_raw_path, RAW_REQUIRED)
    calibration_truth=_load_npz(calibration_truth_path, TRUTH_REQUIRED)
    target_raw=_load_npz(target_raw_path, RAW_REQUIRED)
    calibration_metadata=json.loads(str(calibration_raw["metadata"].item()))
    target_metadata=json.loads(str(target_raw["metadata"].item()))
    for field in ("model", "model_version", "weights_sha256", "track_mapping_sha256"):
        calibration_value=(
            str(calibration_raw["model"].item())
            if field == "model"
            else str(calibration_metadata.get(field, ""))
        )
        target_value=(
            str(target_raw["model"].item())
            if field == "model"
            else str(target_metadata.get(field, ""))
        )
        if calibration_value != target_value:
            raise ValueError(f"calibration/target raw {field} differs")
    raw_ids=calibration_raw["example_id"].astype(str)
    truth_ids=calibration_truth["example_id"].astype(str)
    if set(raw_ids) != set(truth_ids) or len(raw_ids) != len(truth_ids):
        raise ValueError("calibration raw/truth ID sets differ")
    raw_index={value: index for index, value in enumerate(raw_ids)}
    order=np.asarray([raw_index[value] for value in truth_ids])
    prediction=calibration_raw["prediction"][order]
    if prediction.shape != calibration_truth["target"].shape:
        raise ValueError("calibration prediction/target shapes differ")
    errors=np.abs(prediction - calibration_truth["target"])
    groups=calibration_truth["group"].astype(str)
    unique_groups=sorted(set(groups))
    if len(unique_groups) < 3:
        raise ValueError("group-block conformal calibration requires at least 3 groups")
    valid=(
        calibration_truth["mask"].astype(bool)
        if "mask" in calibration_truth
        else np.ones_like(errors, dtype=bool)
    )
    if valid.shape != errors.shape:
        raise ValueError("calibration validity mask shape differs from targets")
    scores=[]
    for group in unique_groups:
        group_valid=valid[groups == group]
        if not group_valid.any():
            raise ValueError(f"calibration group {group!r} has no valid targets")
        scores.append(float(errors[groups == group][group_valid].max()))
    scores=np.asarray(scores)
    rank=min(len(scores), max(1, int(np.ceil((len(scores) + 1) * nominal))))
    radius=float(np.sort(scores)[rank - 1])
    target_prediction=target_raw["prediction"].astype(np.float64)
    lower=target_prediction - radius
    if target_metadata.get("component") in {"mean", "baseline"}:
        lower=np.clip(lower, 0.0, None)
    upper=target_prediction + radius
    previous_command=target_metadata.get("command")
    target_metadata["prediction_command"]=previous_command
    target_metadata["command"]=command
    target_metadata["conformal"]={
        "method": "group_block_max_absolute_split_conformal",
        "nominal": nominal,
        "independent_groups": len(unique_groups),
        "order_statistic_rank": rank,
        "radius": radius,
        "calibration_truth_sha256": sha256_file(calibration_truth_path),
        "calibration_raw_sha256": sha256_file(calibration_raw_path),
    }
    save_raw_predictions(
        out,
        model=str(target_raw["model"].item()),
        example_id=target_raw["example_id"],
        prediction=target_raw["prediction"],
        lower=lower,
        upper=upper,
        metadata=target_metadata,
    )
    return target_metadata["conformal"]

def assemble_prediction_bundle(
    raw_path: str | Path,
    truth_path: str | Path,
    out: str | Path,
    provenance_out: str | Path,
    *,
    training_use: str,
    claim_surface_contract_path: str | Path | None = None,
) -> dict:
    raw=_load_npz(raw_path, RAW_REQUIRED)
    truth=_load_npz(truth_path, TRUTH_REQUIRED)
    raw_ids=raw["example_id"].astype(str)
    truth_ids=truth["example_id"].astype(str)
    if len(set(raw_ids)) != len(raw_ids) or len(set(truth_ids)) != len(truth_ids):
        raise ValueError("raw/truth example IDs must be unique")
    if set(raw_ids) != set(truth_ids):
        missing_prediction=sorted(set(truth_ids) - set(raw_ids))
        unexpected_prediction=sorted(set(raw_ids) - set(truth_ids))
        raise ValueError(
            "raw/truth ID sets differ; selective omission is forbidden: "
            f"missing={len(missing_prediction)}, unexpected={len(unexpected_prediction)}"
        )
    raw_index={value: index for index, value in enumerate(raw_ids)}
    order=np.asarray([raw_index[value] for value in truth_ids])
    prediction=raw["prediction"][order]
    if prediction.shape != truth["target"].shape:
        raise ValueError(
            f"prediction shape {prediction.shape} differs from truth {truth['target'].shape}"
        )
    model=str(raw["model"].item())
    metadata=json.loads(str(raw["metadata"].item()))
    if metadata.get("model") != model:
        raise ValueError("raw metadata model differs from raw model")
    bundle=PredictionBundle(
        model=model,
        example_id=truth["example_id"],
        target=truth["target"],
        prediction=prediction,
        group=truth["group"],
        split=truth["split"],
        lower=raw["lower"][order] if "lower" in raw else None,
        upper=raw["upper"][order] if "upper" in raw else None,
        mask=truth["mask"].astype(bool) if "mask" in truth else None,
    )
    save_prediction_bundle(bundle, out)
    write_prediction_manifest(
        provenance_out,
        model=model,
        model_version=str(metadata["model_version"]),
        prediction_bundle_path=out,
        weights_sha256=str(metadata["weights_sha256"]),
        track_mapping_sha256=str(metadata["track_mapping_sha256"]),
        data_snapshot_sha256=sha256_file(truth_path),
        command=str(metadata["command"]),
        training_use=training_use,
        source_url=metadata.get("source_url"),
        seed=metadata.get("seed"),
        seed_ensemble=metadata.get("seed_ensemble"),
        raw_prediction_sha256=sha256_file(raw_path),
        claim_surface_contract_sha256=(
            sha256_file(claim_surface_contract_path)
            if claim_surface_contract_path is not None
            else None
        ),
    )
    return {
        "model": model,
        "examples": len(bundle.example_id),
        "shape": list(bundle.prediction.shape),
        "bundle_sha256": sha256_file(out),
        "provenance": str(provenance_out),
    }
