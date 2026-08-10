from __future__ import annotations

from dataclasses import asdict,fields,replace
import glob
import json
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[3]

def _filter_shards_by_replicate_quality(
    shard_paths: list[Path],minimum: float | None
) -> list[Path]:
    if minimum is None:
        return shard_paths
    if not 0.0 <= minimum <= 1.0:
        raise ValueError("minimum replicate quality must be in [0, 1]")
    decisions: dict[Path,bool] = {}
    selected = []
    for shard in shard_paths:
        parent = shard.parent
        if parent not in decisions:
            manifest_path = parent / "manifest.json"
            if not manifest_path.is_file():
                raise FileNotFoundError(
                    f"cannot quality-filter shard without manifest: {shard}"
                )
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            assay = manifest.get("track",{}).get("assay_features",[])
            if len(assay) != 12:
                raise ValueError(f"invalid assay vector in {manifest_path}")
            decisions[parent] = float(assay[-1]) >= minimum
        if decisions[parent]:
            selected.append(shard)
    if not selected:
        raise ValueError("replicate-quality filter removed every training shard")
    return selected

def _compiled_geometry(shard_paths: list[Path]) -> tuple[int,int]:
    geometries = set()
    for parent in {path.parent for path in shard_paths}:
        manifest_path = parent / "manifest.json"
        if not manifest_path.is_file():
            raise FileNotFoundError(f"compiled shard manifest missing: {manifest_path}")
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        geometries.add((manifest.get("sequence_length"),manifest.get("bin_size")))
    if len(geometries) != 1:
        raise ValueError(f"compiled shards mix incompatible geometries: {geometries}")
    sequence_length,bin_size = next(iter(geometries))
    if not isinstance(sequence_length,int) or not isinstance(bin_size,int):
        raise ValueError("compiled shard geometry is incomplete")
    return sequence_length,bin_size

def _validate_checkpoint_selection_scope(
    shard_paths: list[Path],validation_studies: list[str]
) -> dict | None:

    if not validation_studies:
        return None
    import numpy as np

    requested = frozenset(study.upper() for study in validation_studies)
    found: set[str] = set()
    profiles = 0
    shards = 0
    examples = 0
    shards_by_parent: dict[Path,list[Path]] = {}
    for path in shard_paths:
        shards_by_parent.setdefault(path.parent,[]).append(path)
    for parent,parent_shards in sorted(shards_by_parent.items()):
        manifest_path = parent / "manifest.json"
        if not manifest_path.is_file():
            raise FileNotFoundError(
                f"checkpoint-selection shard lacks manifest: {manifest_path}"
            )
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        track = manifest.get("track",{})
        study = str(track.get("study") or "").upper()
        if study not in requested:
            continue
        if track.get("split_role") != "validation_study":
            raise ValueError(
                f"checkpoint-selection study {study} must have split_role=validation_study "
                f"in {manifest_path}"
            )
        found.add(study)
        profiles += 1
        for shard_path in sorted(parent_shards):
            with np.load(shard_path,allow_pickle=False) as shard:
                if "study" not in shard.files or "split" not in shard.files:
                    raise ValueError(
                        f"checkpoint-selection shard lacks study/split arrays: {shard_path}"
                    )
                shard_studies = {str(value).upper() for value in shard["study"]}
                shard_splits = {str(value) for value in shard["split"]}
                if shard_studies != {study} or shard_splits != {"validation"}:
                    raise ValueError(
                        f"checkpoint-selection shard {shard_path} has studies "
                        f"{sorted(shard_studies)} and splits {sorted(shard_splits)}; expected "
                        f"only {study}/validation"
                    )
                examples += len(shard["split"])
                shards += 1
    missing = sorted(requested - found)
    if missing:
        raise ValueError(f"checkpoint-selection studies are absent from compiled shards: {missing}")
    if not profiles or not shards or not examples:
        raise ValueError("checkpoint-selection cohort contains no compiled validation examples")
    return {
        "schema": "pdac-circuit.checkpoint-selection-scope/1",
        "studies": sorted(found),
        "profiles": profiles,
        "shards": shards,
        "examples": examples,
        "split": "validation",
        "excluded_from_gradients": True,
    }

def _validate_stage_supervision(
    shard_paths: list[Path],stage: str,*,project_root: Path = PROJECT_ROOT
) -> dict | None:
    if stage != "signed_intervention_residual":
        return None
    import numpy as np
    from .streaming import sha256_file

    common = Path(str(Path(shard_paths[0]).parent))
    for path in shard_paths[1:]:
        while common not in path.parents and common != path.parent:
            if common.parent == common:
                raise ValueError("cannot locate one intervention-pair collection root")
            common = common.parent
    collection_root = common
    while not (collection_root / "_COMPLETE.json").is_file():
        if collection_root.parent == collection_root:
            raise ValueError(
                "signed intervention training requires a paired collection completion marker"
            )
        collection_root = collection_root.parent
    completion_path = collection_root / "_COMPLETE.json"
    completion = json.loads(completion_path.read_text(encoding="utf-8"))
    if (
        completion.get("schema")
        != "pdac-circuit.paired-collection-completion/1"
        or completion.get("successful") is not True
        or completion.get("registered_pairs") != completion.get("verified_pairs")
        or completion.get("registered_pairs",0) < 1
        or not all(
            row.get("valid") is True and row.get("source_shards_verified") is True
            for row in completion.get("pairs",[])
        )
    ):
        raise ValueError("signed intervention paired collection is incomplete or invalid")
    project_root = project_root.resolve()
    registered_manifests = {
        str((project_root / row["output"] / "manifest.json").resolve()): row[
            "manifest_sha256"
        ]
        for row in completion.get("pairs",[])
    }
    parents = sorted({path.parent.resolve() for path in shard_paths})
    selected_manifests = {str(parent / "manifest.json") for parent in parents}
    if selected_manifests != set(registered_manifests):
        raise ValueError(
            "signed intervention training must include every registered pair directory"
        )
    for parent in parents:
        manifest_path = parent / "manifest.json"
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if (
            manifest.get("schema")
            != "pdac-circuit.paired-chromatin-shards/1"
            or manifest.get("mode") != "perturbation"
            or manifest.get("valid") is not True
            or registered_manifests.get(str(manifest_path)) != sha256_file(manifest_path)
        ):
            raise ValueError(f"invalid signed intervention pair manifest {manifest_path}")
        expected_shards = {
            (parent / str(row["path"])).resolve() for row in manifest.get("shards",[])
        }
        selected_shards = {
            path.resolve() for path in shard_paths if path.parent.resolve() == parent
        }
        if not expected_shards or selected_shards != expected_shards:
            raise ValueError(
                f"signed intervention shard selection is incomplete for {parent}"
            )
        representative = next(path for path in shard_paths if path.parent.resolve() == parent)
        with np.load(representative,allow_pickle=False) as shard:
            required = {"perturbation_delta","perturbation_mask"}
            missing = sorted(required - set(shard.files))
            if missing:
                raise ValueError(
                    f"signed intervention shard {representative} lacks {missing}"
                )
    return {
        "schema": "pdac-circuit.signed-intervention-supervision/1",
        "completion_marker": str(completion_path),
        "completion_sha256": sha256_file(completion_path),
        "registered_pairs": completion["registered_pairs"],
        "selected_pair_directories": len(parents),
        "required_arrays": ["perturbation_delta","perturbation_mask"],
    }

def _maybe_tile_stream(stream,model_cfg,source_sequence_length: int,source_bin_size: int):
    if source_bin_size != model_cfg.bin_size:
        raise ValueError("compiled shard bin size does not match model config")
    if source_sequence_length == model_cfg.sequence_length:
        return stream,None
    if (
        model_cfg.architecture != "direct_conditional_cnn"
        or source_sequence_length < model_cfg.sequence_length
        or source_sequence_length % model_cfg.sequence_length
    ):
        raise ValueError(
            "compiled shard sequence length does not match model and cannot be tiled"
        )
    from .streaming import LocalTiledChromatinStream

    return (
        LocalTiledChromatinStream(
            stream,tile_bp=model_cfg.sequence_length,bin_size=model_cfg.bin_size
        ),
        {
            "source_sequence_length": source_sequence_length,
            "tile_sequence_length": model_cfg.sequence_length,
            "tiles_per_source_window": source_sequence_length
            // model_cfg.sequence_length,
        },
    )

def run_inventory(*,out: str | None = None) -> int:
    from .inventory import build_inventory

    report = build_inventory(PROJECT_ROOT)
    rendered = json.dumps(report,indent=2,sort_keys=True)
    print(rendered)
    if out:
        path = Path(out)
        path.parent.mkdir(parents=True,exist_ok=True)
        path.write_text(rendered + "\n",encoding="utf-8")
    return 0

def run_encode_specs(
    *,refresh: bool = False,limit: int | None = None,metadata_workers: int = 8
) -> int:
    from .encode import build_encode_track_specs

    report = build_encode_track_specs(
        PROJECT_ROOT,
        refresh=refresh,
        limit=limit,
        metadata_workers=metadata_workers,
    )
    print(json.dumps(report,indent=2,sort_keys=True))
    return 0 if not report["failures"] else 2

def run_enformer_target_map(
    *,
    refresh: bool = False,
    out: str | None = None,
    policy: str | None = None,
    metadata_cache: str | None = None,
) -> int:
    from .baselines import build_enformer_target_map

    report = build_enformer_target_map(
        PROJECT_ROOT,
        refresh=refresh,
        out=out,
        policy_path=policy,
        metadata_path=metadata_cache,
    )
    print(json.dumps(report,indent=2,sort_keys=True))
    return 0

def run_borzoi_target_map(*,refresh: bool = False,out: str | None = None) -> int:
    from .baselines import build_borzoi_target_map

    report = build_borzoi_target_map(PROJECT_ROOT,refresh=refresh,out=out)
    print(
        json.dumps(
            {
                key: report[key]
                for key in (
                    "schema",
                    "created_at",
                    "source_commit",
                    "target_metadata_sha256",
                    "policy_sha256",
                    "target_count",
                    "input_bp",
                    "native_output_bins",
                    "comparison_bins",
                )
            }
            | {
                "rules": [
                    {"name": rule["name"],"targets": len(rule["target_indices"])}
                    for rule in report["rules"]
                ],
                "strand_pair_indices": len(report["strand_pair_index"]),
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0

def run_plan_campaign(
    *,
    campaign_path: str,
    profile_config: str,
    out: str,
) -> int:
    from .campaign import build_campaign_plan

    report = build_campaign_plan(PROJECT_ROOT,campaign_path,profile_config,out)
    print(json.dumps(report,indent=2,sort_keys=True))
    return 0

def run_zero_baseline(
    *,
    truth_path: str,
    out: str,
    model: str,
    model_version: str,
    weights_sha256: str,
    track_mapping_sha256: str,
) -> int:
    from .evaluation import make_state_invariant_raw

    report = make_state_invariant_raw(
        truth_path,
        out,
        model=model,
        model_version=model_version,
        weights_sha256=weights_sha256,
        track_mapping_sha256=track_mapping_sha256,
        command=" ".join(sys.argv),
    )
    print(json.dumps(report,indent=2,sort_keys=True))
    return 0

def run_assemble_bundle(
    *,
    raw_path: str,
    truth_path: str,
    out: str,
    provenance_out: str | None,
    training_use: str,
    claim_surface_contract: str | None = None,
) -> int:
    from .evaluation import assemble_prediction_bundle

    provenance_out = provenance_out or str(Path(out).with_suffix(".provenance.json"))
    report = assemble_prediction_bundle(
        raw_path,
        truth_path,
        out,
        provenance_out,
        training_use=training_use,
        claim_surface_contract_path=claim_surface_contract,
    )
    print(json.dumps(report,indent=2,sort_keys=True))
    return 0

def run_contrast_raw(
    *,reference_raw: str,treatment_raw: str,out: str,mode: str
) -> int:
    from .evaluation import contrast_raw_predictions

    report = contrast_raw_predictions(
        reference_raw,
        treatment_raw,
        out,
        mode=mode,
        command=" ".join(sys.argv),
    )
    print(json.dumps(report,indent=2,sort_keys=True))
    return 0

def run_conformalize(
    *,
    calibration_raw_path: str,
    calibration_truth_path: str,
    target_raw_path: str,
    out: str,
    nominal: float = 0.90,
) -> int:
    from .evaluation import conformalize_raw_predictions

    report = conformalize_raw_predictions(
        calibration_raw_path,
        calibration_truth_path,
        target_raw_path,
        out,
        nominal=nominal,
        command=" ".join(sys.argv),
    )
    print(json.dumps(report,indent=2,sort_keys=True))
    return 0

def run_freeze_profile_truth(
    *,
    shards_glob: str | list[str],
    split: str,
    out: str,
    crop_bins: int | None = 896,
    genomes: list[str] | None = None,
    example_ids_from: str | None = None,
    target_field: str = "target",
    mask_field: str = "valid",
) -> int:
    from .evaluation import freeze_profile_truth

    patterns = [shards_glob] if isinstance(shards_glob,str) else list(shards_glob)
    paths = sorted(
        {
            Path(value)
            for pattern in patterns
            for value in glob.glob(pattern,recursive=True)
            if Path(value).suffix == ".npz"
        }
    )
    if not paths:
        raise FileNotFoundError(f"no .npz shards matched {patterns!r}")
    report = freeze_profile_truth(
        paths,
        out,
        split=split,
        crop_bins=crop_bins,
        genomes=set(genomes) if genomes else None,
        target_field=target_field,
        mask_field=mask_field,
        example_ids=(
            _load_label_free_example_ids(example_ids_from)
            if example_ids_from
            else None
        ),
    )
    print(json.dumps(report,indent=2,sort_keys=True))
    return 0

def run_freeze_evaluation_windows(
    *,
    shards_glob: str | list[str],
    split: str,
    windows_out: str,
    conditions_out: str,
    output_bins: int = 896,
    genomes: list[str] | None = None,
    context_length: int | None = None,
    example_ids_from: str | None = None,
    max_examples_per_condition_group: int | None = None,
    sampling_seed: int = 20_260_715,
) -> int:
    from .evaluation import freeze_evaluation_windows
    from .streaming import IndexedFasta

    patterns = [shards_glob] if isinstance(shards_glob,str) else list(shards_glob)
    paths = sorted(
        {
            Path(value)
            for pattern in patterns
            for value in glob.glob(pattern,recursive=True)
            if Path(value).suffix == ".npz"
        }
    )
    if not paths:
        raise FileNotFoundError(f"no .npz shards matched {patterns!r}")
    chrom_sizes = None
    if context_length is not None:
        chrom_sizes = {}
        for genome,reference in _default_reference_paths().items():
            if (not genomes or genome in genomes) and reference.exists():
                reader = IndexedFasta(reference)
                reader.assert_genome(genome)
                chrom_sizes[genome] = reader.chrom_sizes()
    report = freeze_evaluation_windows(
        paths,
        windows_out,
        conditions_out,
        split=split,
        output_bins=output_bins,
        genomes=set(genomes) if genomes else None,
        context_length=context_length,
        chrom_sizes=chrom_sizes,
        example_ids=(
            _load_label_free_example_ids(example_ids_from)
            if example_ids_from
            else None
        ),
        max_examples_per_condition_group=max_examples_per_condition_group,
        sampling_seed=sampling_seed,
    )
    print(json.dumps(report,indent=2,sort_keys=True))
    return 0

def run_adapter_train(
    *,
    config_path: str,
    train_raw: str,
    train_truth: str,
    train_conditions: str,
    validation_raw: str,
    validation_truth: str,
    validation_conditions: str,
    out: str,
    epochs: int = 30,
    batch_size: int = 16,
    learning_rate: float = 2e-4,
    weight_decay: float = 1e-4,
    seed: int = 20_260_620,
    device: str = "cpu",
) -> int:
    from .adapter_pipeline import train_enformer_adapter

    report = train_enformer_adapter(
        config_path,
        train_raw=train_raw,
        train_truth=train_truth,
        train_conditions=train_conditions,
        validation_raw=validation_raw,
        validation_truth=validation_truth,
        validation_conditions=validation_conditions,
        out=out,
        epochs=epochs,
        batch_size=batch_size,
        learning_rate=learning_rate,
        weight_decay=weight_decay,
        seed=seed,
        device=device,
    )
    print(json.dumps(report,indent=2,sort_keys=True))
    return 0

def run_adapter_predict(
    *,
    config_path: str,
    checkpoint_path: str,
    raw_path: str,
    conditions_path: str,
    out: str,
    batch_size: int = 32,
    device: str = "cpu",
    ablate_intervention_residual: bool = False,
) -> int:
    from .adapter_pipeline import predict_enformer_adapter

    report = predict_enformer_adapter(
        config_path,
        checkpoint_path=checkpoint_path,
        raw_path=raw_path,
        conditions_path=conditions_path,
        out=out,
        batch_size=batch_size,
        device=device,
        ablate_intervention_residual=ablate_intervention_residual,
    )
    print(json.dumps(report,indent=2,sort_keys=True))
    return 0

def run_merge_raw_predictions(*,inputs_glob: str,out: str) -> int:
    from .evaluation import merge_raw_predictions

    paths = sorted(
        path
        for path in (Path(value) for value in glob.glob(inputs_glob,recursive=True))
        if path.suffix == ".npz"
    )
    if not paths:
        raise FileNotFoundError(f"no .npz raw predictions matched {inputs_glob!r}")
    report = merge_raw_predictions(paths,out,command=" ".join(sys.argv))
    print(json.dumps(report,indent=2,sort_keys=True))
    return 0

def run_ensemble_seed_predictions(
    *,inputs_glob: str,campaign_path: str,out: str
) -> int:
    from .evaluation import ensemble_seed_raw_predictions

    paths = sorted(
        path
        for path in (Path(value) for value in glob.glob(inputs_glob,recursive=True))
        if path.suffix == ".npz"
    )
    if not paths:
        raise FileNotFoundError(f"no .npz seed predictions matched {inputs_glob!r}")
    campaign = json.loads(Path(campaign_path).read_text(encoding="utf-8"))
    report = ensemble_seed_raw_predictions(
        paths,
        out,
        registered_seeds=campaign.get("seeds",[]),
        command=" ".join(sys.argv),
    )
    print(json.dumps(report,indent=2,sort_keys=True))
    return 0

def run_audit_compiled_splits(*,shards_glob: str,out: str | None = None) -> int:
    from .compile_plan import audit_compiled_splits

    paths = sorted(
        path
        for path in (Path(value) for value in glob.glob(shards_glob,recursive=True))
        if path.suffix == ".npz"
    )
    if not paths:
        raise FileNotFoundError(f"no .npz shards matched {shards_glob!r}")
    report = audit_compiled_splits(paths)
    rendered = json.dumps(report,indent=2,sort_keys=True)
    if out:
        destination = Path(out)
        destination.parent.mkdir(parents=True,exist_ok=True)
        destination.write_text(rendered + "\n",encoding="utf-8")
    print(rendered)
    return 0 if report["ok"] else 2

def run_study_plan(
    *,accession: str,out: str | None = None,skip_size_probe: bool = False
) -> int:
    from .geo import build_geo_plan

    report = build_geo_plan(
        PROJECT_ROOT,accession,out=out,probe_sizes=not skip_size_probe
    )
    print(json.dumps(report,indent=2,sort_keys=True))
    return 0

def run_fetch_study(
    *,
    plan_path: str,
    output_root: str,
    allow_protected_study: bool = False,
    protected_seal: str | None = None,
    protected_release: str | None = None,
    max_total_gb: float = 25.0,
) -> int:
    from .geo import fetch_geo_plan

    report = fetch_geo_plan(
        plan_path,
        output_root,
        allow_protected_study=allow_protected_study,
        protected_seal_path=protected_seal,
        protected_release_path=protected_release,
        project_root=PROJECT_ROOT,
        max_total_gb=max_total_gb,
    )
    print(json.dumps(report,indent=2,sort_keys=True))
    return 0

def run_inspect_geo_archive(*,archive: str,out: str | None = None) -> int:
    from .geo_archive import inspect_geo_archive

    report = inspect_geo_archive(archive,out)
    print(json.dumps(report,indent=2,sort_keys=True))
    return 0

def run_extract_geo_archive(
    *,archive: str,output: str,max_unpacked_gb: float = 25.0
) -> int:
    from .geo_archive import extract_geo_archive

    report = extract_geo_archive(
        archive,output,max_unpacked_gb=max_unpacked_gb
    )
    print(json.dumps(report,indent=2,sort_keys=True))
    return 0

def run_geo_metadata(
    *,
    accession: str,
    refresh: bool = False,
    allow_protected_metadata: bool = False,
    protected_release: str | None = None,
) -> int:
    from .geo_metadata import fetch_geo_soft_metadata

    report = fetch_geo_soft_metadata(
        PROJECT_ROOT,
        accession,
        refresh=refresh,
        allow_protected_metadata=allow_protected_metadata,
        protected_release_path=protected_release,
    )
    summary = {
        key: report[key]
        for key in (
            "schema",
            "accession",
            "source_sha256",
            "protected",
            "samples_total",
            "samples_resolved",
            "errors",
            "registry_path",
        )
    }
    print(json.dumps(summary,indent=2,sort_keys=True))
    return 0 if not report["errors"] else 2

def run_seal_protected_studies(
    *,campaign: str,registry: str,assets: str,out: str
) -> int:
    from .protected import freeze_protected_study_seal

    report = freeze_protected_study_seal(
        PROJECT_ROOT,
        campaign_path=campaign,
        registry_path=registry,
        assets_path=assets,
        out=out,
    )
    print(json.dumps(report,indent=2,sort_keys=True))
    return 0

def run_authorize_protected_metadata(
    *,seal: str,checkpoints: list[str],out: str
) -> int:
    from .protected import authorize_protected_metadata_release

    report = authorize_protected_metadata_release(
        PROJECT_ROOT,
        seal_path=seal,
        checkpoint_paths=checkpoints,
        out=out,
    )
    print(json.dumps(report,indent=2,sort_keys=True))
    return 0

def run_geo_track_specs(
    *,
    accession: str,
    extracted_dir: str,
    metadata_path: str | None = None,
    evaluation_only: bool = False,
    protected_release: str | None = None,
) -> int:
    from .geo_metadata import build_geo_track_specs

    report = build_geo_track_specs(
        PROJECT_ROOT,
        accession,
        extracted_dir,
        metadata_path=metadata_path,
        evaluation_only=evaluation_only,
        protected_release_path=protected_release,
    )
    print(json.dumps(report,indent=2,sort_keys=True))
    return 0 if not report["failures"] else 2

def run_intervention_pair_plan(*,track_index: str,out: str | None = None) -> int:
    from .pair_registry import build_intervention_pair_plan

    report = build_intervention_pair_plan(PROJECT_ROOT,track_index,out=out)
    print(json.dumps(report,indent=2,sort_keys=True))
    return 0 if not report["unresolved"] else 2

def run_external_perturbation_pair_plan(
    *,
    track_indexes: list[str],
    contract: str,
    out: str,
    merged_index_out: str,
) -> int:
    from .pair_registry import build_external_perturbation_pair_plan

    report = build_external_perturbation_pair_plan(
        PROJECT_ROOT,
        track_indexes,
        contract_path=contract,
        out=out,
        merged_index_out=merged_index_out,
    )
    print(json.dumps(report,indent=2,sort_keys=True))
    return 0 if not report["unresolved"] else 2

def _default_reference_paths() -> dict[str,Path]:
    return {
        "hg38": PROJECT_ROOT / "data" / "raw" / "hg38-ref" / "hg38.fa",
        "hg19": PROJECT_ROOT / "data" / "raw" / "hg19-ref" / "hg19.fa",
        "mm10": PROJECT_ROOT / "data" / "raw" / "mm10-ref" / "mm10.fa",
        "mm9": PROJECT_ROOT / "data" / "raw" / "mm9-ref" / "mm9.fa",
    }

def _load_label_free_example_ids(path: str | Path) -> frozenset[str]:

    import numpy as np

    source = Path(path)
    if source.suffix.lower() == ".json":
        payload = json.loads(source.read_text(encoding="utf-8"))
        if payload.get("schema") != "pdac-circuit.evaluation-windows/1":
            raise ValueError("example-ID JSON must be a label-free evaluation-window artifact")
        values = [str(example["example_id"]) for example in payload.get("examples",[])]
    elif source.suffix.lower() == ".npz":
        with np.load(source,allow_pickle=False) as archive:
            if "example_id" not in archive.files or "metadata" not in archive.files:
                raise ValueError("example-ID NPZ lacks example_id/metadata")
            metadata = json.loads(str(archive["metadata"].item()))
            if (
                metadata.get("schema") != "pdac-circuit.baseline-conditions/1"
                or metadata.get("contains_targets") is not False
            ):
                raise ValueError("example-ID NPZ must be a label-free conditions artifact")
            values = archive["example_id"].astype(str).tolist()
    else:
        raise ValueError("example-ID cohort must be evaluation-window JSON or conditions NPZ")
    if not values or len(set(values)) != len(values):
        raise ValueError("example-ID cohort must be nonempty and unique")
    return frozenset(values)

def run_fetch_reference(*,genome: str,discard_compressed: bool = False) -> int:
    from .reference import materialize_reference

    report = materialize_reference(
        PROJECT_ROOT,
        genome,
        keep_compressed=not discard_compressed,
    )
    print(json.dumps(report,indent=2,sort_keys=True))
    return 0

def _device_would_use_cuda(requested: str) -> bool:
    if requested == "cpu":
        return False
    try:
        import torch

        return torch.cuda.is_available() and (requested == "auto" or requested.startswith("cuda"))
    except Exception:
        return False

def _vram_gate(requested: str,minimum_free_gb: float,allow_low_vram: bool) -> None:
    from .trainer import available_cuda_memory_gb

    if not _device_would_use_cuda(requested):
        return
    free = available_cuda_memory_gb()
    if free is not None and free < minimum_free_gb and not allow_low_vram:
        raise RuntimeError(
            f"only {free:.2f} GB CUDA memory is free; require {minimum_free_gb:.2f} GB. "
            "Use --device cpu, wait for the live workload, or explicitly pass --allow-low-vram."
        )

def run_model_info(
    *,
    config_path: str,
    forward_check: bool = False,
    device: str = "cpu",
    minimum_free_gb: float = 8.0,
    allow_low_vram: bool = False,
) -> int:
    from .config import load_chromatin_config
    from .model import build_chromatin_model
    from .trainer import parameter_report

    model_cfg,_,payload = load_chromatin_config(config_path)
    model = build_chromatin_model(model_cfg)
    report = {
        "profile": payload.get("profile"),
        "sequence_length": model_cfg.sequence_length,
        "bin_size": model_cfg.bin_size,
        "n_bins": model_cfg.n_bins,
        **parameter_report(model),
    }
    if forward_check:
        import numpy as np
        import torch

        from .streaming import IndexedFasta,one_hot_sequence

        _vram_gate(device,minimum_free_gb,allow_low_vram)
        actual_device = torch.device(
            ("cuda" if torch.cuda.is_available() else "cpu")
            if device == "auto"
            else device
        )
        fasta = IndexedFasta(PROJECT_ROOT / "data" / "raw" / "hg38-ref" / "hg38.fa")
        sequence = one_hot_sequence(fasta.fetch("chr1",0,model_cfg.sequence_length))[None]
        assay = np.zeros((1,model_cfg.assay_features),dtype=np.float32)
        state = np.zeros((1,model_cfg.state_features),dtype=np.float32)
        perturbation = np.zeros((1,model_cfg.perturbation_features),dtype=np.float32)
        assay[:,0] = 1.0
        state[:,0] = 1.0
        model = model.to(actual_device).eval()
        with torch.no_grad():
            output = model(
                torch.from_numpy(sequence).to(actual_device),
                torch.from_numpy(assay).to(actual_device),
                torch.from_numpy(state).to(actual_device),
                torch.from_numpy(perturbation).to(actual_device),
                torch.zeros(1,device=actual_device),
            )
        report["forward_shape"] = list(output.mean.shape)
        report["forward_finite"] = bool(torch.isfinite(output.mean).all().cpu())
        if actual_device.type == "cuda":
            report["peak_cuda_gb"] = round(torch.cuda.max_memory_allocated() / 1024**3,3)
    print(json.dumps(report,indent=2,sort_keys=True))
    return 0

def _load_track_spec(path: str):
    from .streaming import TrackSpec

    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    allowed = {field.name for field in fields(TrackSpec)}
    unknown = sorted(set(payload) - allowed)
    if unknown:
        raise ValueError(f"unknown TrackSpec fields: {unknown}")
    for key in ("assay_features","state_features","perturbation_features"):
        payload[key] = tuple(payload[key])
    return TrackSpec(**payload)

def run_compile(
    *,
    config_path: str,
    track_spec_path: str,
    output_dir: str,
    fasta_path: str | None = None,
    stride: int | None = None,
    max_windows: int | None = None,
    windows_per_shard: int = 64,
    negative_keep_probability: float = 0.05,
    emit_report: bool = True,
) -> int:
    from .config import load_chromatin_config
    from .streaming import IndexedFasta,compile_bigwig_track,genome_windows

    model_cfg,train_cfg,_ = load_chromatin_config(config_path)
    track = _load_track_spec(track_spec_path)
    if len(track.assay_features) != model_cfg.assay_features:
        raise ValueError("track assay feature count does not match model config")
    if len(track.state_features) != model_cfg.state_features:
        raise ValueError("track state feature count does not match model config")
    if len(track.perturbation_features) != model_cfg.perturbation_features:
        raise ValueError("track perturbation feature count does not match model config")
    fasta_path = fasta_path or str(_default_reference_paths()[track.genome])
    fasta = IndexedFasta(fasta_path)
    fasta.assert_genome(track.genome)
    windows = genome_windows(
        fasta.chrom_sizes(),
        sequence_length=model_cfg.sequence_length,
        stride=stride,
        max_windows=max_windows,
        seed=train_cfg.seed,
    )
    report = compile_bigwig_track(
        track,
        windows,
        output_dir,
        bin_size=model_cfg.bin_size,
        windows_per_shard=windows_per_shard,
        negative_keep_probability=negative_keep_probability,
        seed=train_cfg.seed,
        sequence_length=model_cfg.sequence_length,
    )
    if emit_report:
        print(json.dumps(report,indent=2,sort_keys=True))
    return 0

def run_compile_index(
    *,
    config_path: str,
    track_index_path: str,
    output_dir: str,
    max_tracks: int | None = None,
    stride: int | None = None,
    max_windows: int | None = None,
    windows_per_shard: int = 64,
    negative_keep_probability: float = 0.05,
) -> int:
    from datetime import datetime
    import uuid

    from .config import load_chromatin_config
    from .compile_plan import verify_compiled_track
    from .streaming import sha256_file

    index_path = Path(track_index_path)
    config = Path(config_path)
    destination_root = Path(output_dir)
    completion_path = destination_root / "_COMPLETE.json"
    index = json.loads(index_path.read_text(encoding="utf-8"))
    if index.get("schema") not in {
        "pdac-circuit.geo-track-specs/1",
        "pdac-circuit.encode-track-specs/1",
    }:
        raise ValueError("unsupported TrackSpec index schema")
    if index.get("failures"):
        raise ValueError("TrackSpec index has unresolved failures; refusing partial compilation")
    rows = list(index.get("written",[]))
    if not rows:
        raise ValueError("TrackSpec index contains no runnable tracks")
    full_index_run = max_tracks is None
    if full_index_run and completion_path.exists():
        completion_path.unlink()
    if max_tracks is not None:
        if max_tracks < 1:
            raise ValueError("max_tracks must be positive")
        rows = rows[:max_tracks]
    completed,skipped,failures = [],[],[]
    for row in rows:
        spec_path = PROJECT_ROOT / row["spec"]
        spec = _load_track_spec(str(spec_path))
        destination = Path(output_dir) / spec.accession
        if destination.exists():
            verification = verify_compiled_track(destination,spec_path,config_path)
            if verification["valid"]:
                skipped.append({"accession": spec.accession,**verification})
                continue
            failures.append(
                {
                    "accession": spec.accession,
                    "error": "existing compiled output failed verification",
                    "details": verification["failures"],
                }
            )
            continue
        try:
            run_compile(
                config_path=config_path,
                track_spec_path=str(spec_path),
                output_dir=output_dir,
                stride=stride,
                max_windows=max_windows,
                windows_per_shard=windows_per_shard,
                negative_keep_probability=negative_keep_probability,
                emit_report=False,
            )
            verification = verify_compiled_track(destination,spec_path,config_path)
            if not verification["valid"]:
                raise RuntimeError(f"post-compile verification failed: {verification['failures']}")
            completed.append({"accession": spec.accession,**verification})
        except Exception as exc:
            failures.append(
                {"accession": spec.accession,"error": f"{type(exc).__name__}: {exc}"}
            )
    successful = not failures and len(completed) + len(skipped) == len(rows)
    completion_marker = None
    if full_index_run and successful:
        model,_,_ = load_chromatin_config(config)
        verified = sorted(
            completed + skipped,key=lambda row: str(row["accession"])
        )
        completion = {
            "schema": "pdac-circuit.compiled-collection-completion/1",
            "created_at": datetime.now().astimezone().isoformat(),
            "successful": True,
            "track_index": track_index_path,
            "track_index_sha256": sha256_file(index_path),
            "config": config_path,
            "config_sha256": sha256_file(config),
            "sequence_length": model.sequence_length,
            "bin_size": model.bin_size,
            "registered_tracks": len(index["written"]),
            "verified_tracks": len(verified),
            "examples": sum(int(row.get("examples") or 0) for row in verified),
            "tracks": verified,
        }
        completion_path.parent.mkdir(parents=True,exist_ok=True)
        temporary = completion_path.parent / (
            f".{completion_path.name}.partial-{uuid.uuid4().hex}"
        )
        temporary.write_text(
            json.dumps(completion,indent=2,sort_keys=True) + "\n",
            encoding="utf-8",
        )
        temporary.replace(completion_path)
        completion_marker = str(completion_path)
    report = {
        "schema": "pdac-circuit.compile-index-run/1",
        "track_index": track_index_path,
        "requested": len(rows),
        "completed": completed,
        "skipped_verified": skipped,
        "failures": failures,
        "output_dir": output_dir,
        "successful": successful,
        "completion_marker": completion_marker,
    }
    print(json.dumps(report,indent=2,sort_keys=True))
    return 0 if successful else 2

def run_pair_shards(
    *,
    reference_glob: str,
    treatment_glob: str,
    output_dir: str,
    mode: str,
    windows_per_shard: int = 64,
    minimum_overlap_fraction: float = 0.80,
) -> int:
    from .pairing import compose_paired_shards

    reference = sorted(
        path
        for path in (Path(value) for value in glob.glob(reference_glob,recursive=True))
        if path.suffix == ".npz"
    )
    treatment = sorted(
        path
        for path in (Path(value) for value in glob.glob(treatment_glob,recursive=True))
        if path.suffix == ".npz"
    )
    report = compose_paired_shards(
        reference,
        treatment,
        output_dir,
        mode=mode,
        windows_per_shard=windows_per_shard,
        minimum_overlap_fraction=minimum_overlap_fraction,
    )
    print(json.dumps(report,indent=2,sort_keys=True))
    return 0

def run_materialize_intervention_pairs(
    *,
    pair_plan: str,
    compiled_root: str,
    output_root: str,
    windows_per_shard: int = 64,
    minimum_overlap_fraction: float = 0.80,
) -> int:
    from .pair_registry import materialize_intervention_pair_plan

    report = materialize_intervention_pair_plan(
        PROJECT_ROOT,
        pair_plan,
        compiled_root,
        output_root,
        windows_per_shard=windows_per_shard,
        minimum_overlap_fraction=minimum_overlap_fraction,
    )
    print(json.dumps(report,indent=2,sort_keys=True))
    return 0

def run_train(
    *,
    config_path: str,
    shards_glob: str | list[str],
    checkpoint_dir: str,
    fasta_path: str | None = None,
    device: str | None = None,
    resume: bool = True,
    minimum_free_gb: float = 8.0,
    allow_low_vram: bool = False,
    stage: str = "human_state_adaptation",
    initialize_from: str | None = None,
    seed: int | None = None,
    minimum_replicate_quality: float | None = None,
    validation_study: str | list[str] | None = None,
) -> int:
    import torch

    from .config import load_chromatin_config
    from .curriculum import apply_training_stage
    from .model import build_chromatin_model
    from .streaming import (
        ChromatinShardStream,
        as_torch_iterable,
        collate_chromatin,
        shard_collection_fingerprint,
    )
    from .trainer import MemoryBoundedTrainer,load_weights_for_initialization

    model_cfg,train_cfg,_ = load_chromatin_config(config_path)
    if seed is not None:
        if seed < 0:
            raise ValueError("seed must be non-negative")
        train_cfg = replace(train_cfg,seed=seed)
    if device:
        train_cfg = replace(train_cfg,device=device)
        train_cfg.validate()
    validation_studies = (
        [validation_study] if isinstance(validation_study,str) else list(validation_study or [])
    )
    validation_studies = [str(study).upper() for study in validation_studies]
    if len(set(validation_studies)) != len(validation_studies):
        raise ValueError("validation studies must be unique")
    if validation_studies and stage != "human_state_adaptation":
        raise ValueError(
            "study-restricted checkpoint selection is registered only for human_state_adaptation"
        )
    _vram_gate(train_cfg.device,minimum_free_gb,allow_low_vram)
    shard_globs = [shards_glob] if isinstance(shards_glob,str) else list(shards_glob)
    if not shard_globs or any(not pattern for pattern in shard_globs):
        raise ValueError("at least one nonempty shard glob is required")
    if len(set(shard_globs)) != len(shard_globs):
        raise ValueError("duplicate shard globs are not permitted")
    shard_paths = sorted(
        {
            Path(path).resolve()
            for pattern in shard_globs
            for path in glob.glob(pattern,recursive=True)
            if Path(path).suffix == ".npz"
        }
    )
    if not shard_paths:
        raise FileNotFoundError(f"no .npz shards matched {shard_globs!r}")
    matched_shards = len(shard_paths)
    shard_paths = _filter_shards_by_replicate_quality(
        shard_paths,minimum_replicate_quality
    )
    stage_supervision = _validate_stage_supervision(shard_paths,stage)
    checkpoint_selection_scope = _validate_checkpoint_selection_scope(
        shard_paths,validation_studies
    )
    data_fingerprint = shard_collection_fingerprint(shard_paths)
    source_sequence_length,source_bin_size = _compiled_geometry(shard_paths)
    fasta_path = (
        {"hg38": Path(fasta_path)} if fasta_path else _default_reference_paths()
    )
    stream = ChromatinShardStream(
        shard_paths,
        fasta_path,
        seed=train_cfg.seed,
        include_splits={"train"},
        exclude_studies=set(validation_studies),
        validation_only_studies=set(validation_studies),
        conditioning_dimensions={
            "assay_features": model_cfg.assay_features,
            "state_features": model_cfg.state_features,
            "perturbation_features": model_cfg.perturbation_features,
        },
    )
    validation_stream = ChromatinShardStream(
        shard_paths,
        fasta_path,
        shuffle=False,
        seed=train_cfg.seed,
        include_splits={"validation"},
        include_studies=set(validation_studies) if validation_studies else None,
        validation_only_studies=set(validation_studies),
        conditioning_dimensions={
            "assay_features": model_cfg.assay_features,
            "state_features": model_cfg.state_features,
            "perturbation_features": model_cfg.perturbation_features,
        },
    )
    stream,tiling = _maybe_tile_stream(
        stream,model_cfg,source_sequence_length,source_bin_size
    )
    validation_stream,validation_tiling = _maybe_tile_stream(
        validation_stream,model_cfg,source_sequence_length,source_bin_size
    )
    if tiling != validation_tiling:
        raise RuntimeError("training and validation tiling contracts differ")
    loader_kwargs = {
        "batch_size": train_cfg.micro_batch_size,
        "num_workers": train_cfg.num_workers,
        "collate_fn": collate_chromatin,
        "pin_memory": _device_would_use_cuda(train_cfg.device),
    }
    if train_cfg.num_workers:
        loader_kwargs["prefetch_factor"] = train_cfg.prefetch_factor
    loader = torch.utils.data.DataLoader(as_torch_iterable(stream),**loader_kwargs)
    validation_loader = torch.utils.data.DataLoader(
        as_torch_iterable(validation_stream),
        batch_size=1,
        num_workers=0,
        collate_fn=collate_chromatin,
        pin_memory=_device_would_use_cuda(train_cfg.device),
    )
    model = build_chromatin_model(model_cfg)
    stage_report = apply_training_stage(model,stage)
    checkpoint_path = Path(checkpoint_dir) / "latest.pt"
    initialization_provenance = None
    if initialize_from:
        if checkpoint_path.exists():
            raise ValueError(
                "--initialize-from requires a new checkpoint directory; latest.pt already exists"
            )
        initialization_provenance = load_weights_for_initialization(
            model,
            model_cfg,
            initialize_from,
            device=torch.device("cpu"),
        )
        apply_training_stage(model,stage)
    trainer = MemoryBoundedTrainer(
        model,
        model_cfg,
        train_cfg,
        training_stage=stage,
        data_fingerprint=data_fingerprint,
        initialization_provenance=initialization_provenance,
    )
    report = trainer.fit(
        loader,
        checkpoint_dir,
        resume=resume,
        validation_loader=validation_loader,
    )
    report["training_stage"] = stage_report
    report["initialized_from"] = initialization_provenance
    report["data_fingerprint"] = data_fingerprint
    report["replicate_quality_filter"] = {
        "minimum": minimum_replicate_quality,
        "matched_shards": matched_shards,
        "selected_shards": len(shard_paths),
    }
    report["local_tiling"] = tiling
    report["stage_supervision"] = stage_supervision
    report["checkpoint_selection"] = {
        "split": "validation",
        "studies": validation_studies or "all_stage_validation_groups",
        "validation_studies_excluded_from_gradients": bool(validation_studies),
        "preflight_scope": checkpoint_selection_scope,
    }
    print(json.dumps(report,indent=2,sort_keys=True))
    return 0

def run_predict(
    *,
    config_path: str,
    checkpoint_path: str,
    shards_glob: str,
    out: str,
    track_mapping_path: str,
    fasta_path: str | None = None,
    device: str = "auto",
    component: str = "mean",
    crop_bins: int | None = 896,
    reverse_complement: bool = True,
    minimum_free_gb: float = 8.0,
    allow_low_vram: bool = False,
    example_ids_from: str | None = None,
    ablate_state_residual: bool = False,
    ablate_intervention_residual: bool = False,
    seed: int | None = None,
) -> int:
    import numpy as np
    import torch

    from .config import load_chromatin_config
    from .evaluation import save_raw_predictions
    from .model import build_chromatin_model
    from .streaming import (
        ChromatinShardStream,
        as_torch_iterable,
        collate_chromatin,
        sha256_file,
    )
    from .trainer import load_weights_for_inference

    allowed_components = {
        "mean",
        "baseline",
        "state_residual",
        "perturbation_residual",
        "residual",
        "circuit_factors",
        "intervention_factors",
        "intervention_axis_potentials",
    }
    if component not in allowed_components:
        raise ValueError(f"unsupported component {component!r}")
    _vram_gate(device,minimum_free_gb,allow_low_vram)
    model_cfg,train_cfg,_ = load_chromatin_config(config_path)
    if seed is not None:
        if seed < 0:
            raise ValueError("seed must be non-negative")
        train_cfg = replace(train_cfg,seed=seed)
        train_cfg.validate()
    actual_device = torch.device(
        ("cuda" if torch.cuda.is_available() else "cpu")
        if device == "auto"
        else device
    )
    model = build_chromatin_model(model_cfg)
    checkpoint = load_weights_for_inference(
        model,
        model_cfg,
        train_cfg,
        checkpoint_path,
        device=actual_device,
    )
    shard_paths = sorted(Path(path) for path in glob.glob(shards_glob,recursive=True))
    shard_paths = [path for path in shard_paths if path.suffix == ".npz"]
    if not shard_paths:
        raise FileNotFoundError(f"no .npz shards matched {shards_glob!r}")
    source_sequence_length,source_bin_size = _compiled_geometry(shard_paths)
    fasta_path = (
        {"hg38": Path(fasta_path)} if fasta_path else _default_reference_paths()
    )
    cohort_ids = (
        _load_label_free_example_ids(example_ids_from) if example_ids_from else None
    )
    stream = ChromatinShardStream(
        shard_paths,
        fasta_path,
        shuffle=False,
        seed=train_cfg.seed,
        include_example_ids=cohort_ids,
        include_targets=False,
        conditioning_dimensions={
            "assay_features": model_cfg.assay_features,
            "state_features": model_cfg.state_features,
            "perturbation_features": model_cfg.perturbation_features,
        },
    )
    stream,tiling = _maybe_tile_stream(
        stream,model_cfg,source_sequence_length,source_bin_size
    )
    loader = torch.utils.data.DataLoader(
        as_torch_iterable(stream),
        batch_size=1,
        num_workers=0,
        collate_fn=collate_chromatin,
        pin_memory=actual_device.type == "cuda",
    )
    predictions,example_ids = [],[]
    tiled_predictions: dict[str,list[tuple[int,np.ndarray]]] = {}
    tiled_parent_order: list[str] = []
    profile_components = {
        "mean",
        "baseline",
        "state_residual",
        "perturbation_residual",
        "residual",
    }
    if tiling is not None and component not in profile_components:
        raise ValueError("local tiled inference supports profile components only")
    with torch.no_grad():
        for batch in loader:
            inputs = (
                batch["sequence"].float().to(actual_device),
                batch["assay_features"].float().to(actual_device),
                batch["state_features"].float().to(actual_device),
                batch["perturbation_features"].float().to(actual_device),
                batch["disease_mask"].float().to(actual_device),
            )
            output = (
                model.predict_reverse_complement_ensemble(
                    *inputs,
                    ablate_state_residual=ablate_state_residual,
                    ablate_intervention_residual=ablate_intervention_residual,
                )
                if reverse_complement
                else model(
                    *inputs,
                    ablate_state_residual=ablate_state_residual,
                    ablate_intervention_residual=ablate_intervention_residual,
                )
            )
            value = getattr(output,component)
            if tiling is None and component in profile_components and crop_bins is not None:
                if crop_bins < 1 or crop_bins > value.shape[-1]:
                    raise ValueError(
                        f"crop_bins={crop_bins} is invalid for {value.shape[-1]} bins"
                    )
                trim = (value.shape[-1] - crop_bins) // 2
                value = value[...,trim : trim + crop_bins]
            array = value.detach().float().cpu().numpy()
            if tiling is None:
                predictions.append(array)
                example_ids.extend(batch["example_id"])
            else:
                if len(batch.get("parent_example_id",[])) != 1:
                    raise ValueError("local tiled inference requires batch size one")
                parent = str(batch["parent_example_id"][0])
                tile_index = int(batch["tile_index"][0])
                if parent not in tiled_predictions:
                    tiled_predictions[parent] = []
                    tiled_parent_order.append(parent)
                tiled_predictions[parent].append((tile_index,array))
    if tiling is not None:
        expected_tiles = int(tiling["tiles_per_source_window"])
        for parent in tiled_parent_order:
            rows = sorted(tiled_predictions[parent],key=lambda row: row[0])
            if [index for index,_ in rows] != list(range(expected_tiles)):
                raise ValueError(f"incomplete local tiling for {parent}")
            full = np.concatenate([array for _,array in rows],axis=-1)
            if crop_bins is not None:
                if crop_bins < 1 or crop_bins > full.shape[-1]:
                    raise ValueError(
                        f"crop_bins={crop_bins} is invalid for reconstructed "
                        f"{full.shape[-1]} bins"
                    )
                trim = (full.shape[-1] - crop_bins) // 2
                full = full[...,trim : trim + crop_bins]
            predictions.append(full)
            example_ids.append(parent)
    if not predictions:
        raise RuntimeError("candidate inference cohort produced no examples")
    if cohort_ids is not None and set(example_ids) != set(cohort_ids):
        missing = sorted(set(cohort_ids) - set(example_ids))
        extra = sorted(set(example_ids) - set(cohort_ids))
        raise ValueError(
            f"candidate predictions do not match the frozen cohort: "
            f"missing={missing[:5]}, extra={extra[:5]}"
        )
    if len(set(example_ids)) != len(example_ids):
        raise ValueError("candidate prediction example IDs are not unique")
    prediction = np.concatenate(predictions,axis=0)
    weights_sha = sha256_file(checkpoint_path)
    metadata = {
        "schema": "pdac-circuit.raw-predictions/1",
        "model": model.model_name,
        "model_version": f"checkpoint-sha256:{weights_sha}",
        "weights_sha256": weights_sha,
        "track_mapping_sha256": sha256_file(track_mapping_path),
        "component": component,
        "crop_bins": crop_bins,
        "reverse_complement": reverse_complement,
        "checkpoint": checkpoint,
        "signed_perturbation_features": model_cfg.signed_perturbation_features,
        "ablate_state_residual": ablate_state_residual,
        "ablate_intervention_residual": ablate_intervention_residual,
        "local_tiling": tiling,
        "seed": train_cfg.seed,
        "label_free_cohort": (
            {
                "path": str(example_ids_from),
                "sha256": sha256_file(example_ids_from),
                "examples": len(cohort_ids),
            }
            if example_ids_from and cohort_ids is not None
            else None
        ),
        "command": " ".join(sys.argv),
    }
    save_raw_predictions(
        out,
        model=model.model_name,
        example_id=np.asarray(example_ids),
        prediction=prediction,
        metadata=metadata,
    )
    print(
        json.dumps(
            {
                "out": out,
                "examples": len(example_ids),
                "shape": list(prediction.shape),
                **metadata,
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0

def run_circuit_audit(
    *,inputs_glob: str,registry_path: str,out: str
) -> int:
    from .circuit_audit import (
        audit_circuit_stability,
        write_circuit_stability_audit,
    )

    paths = sorted(Path(path) for path in glob.glob(inputs_glob,recursive=True))
    paths = [path for path in paths if path.suffix == ".npz"]
    if not paths:
        raise FileNotFoundError(f"no factor bundles matched {inputs_glob!r}")
    registry = json.loads(Path(registry_path).read_text(encoding="utf-8"))
    report = audit_circuit_stability(paths,registry)
    write_circuit_stability_audit(out,report)
    print(json.dumps({"out": out,**report},indent=2,sort_keys=True))
    return 0

def run_benchmark(
    *,
    candidate_root: str,
    baseline_root: str,
    registry_path: str,
    out: str,
    bootstrap: int = 10_000,
    candidate_seed_roots: list[str] | None = None,
    comparison_role: str = "headline_enformer",
) -> int:
    import numpy as np

    from .benchmark import (
        BenchmarkRule,
        claim_report,
        compare_axis,
        interval_calibration,
        load_prediction_bundle,
        validate_registered_axis_groups,
        write_claim_report,
    )
    from .provenance import validate_prediction_manifest

    candidate_root_path = Path(candidate_root)
    baseline_root_path = Path(baseline_root)
    registry = json.loads(Path(registry_path).read_text(encoding="utf-8"))
    claim_surface_sha256 = None
    claim_surface_path = registry.get("protected_study_policy",{}).get(
        "claim_surface_contract"
    )
    if claim_surface_path:
        from .claim_surfaces import validate_claim_surface_contract

        claim_surface_sha256 = validate_claim_surface_contract(
            PROJECT_ROOT,
            registry_path=registry_path,
            contract_path=claim_surface_path,
        )["sha256"]
    seed_policy = registry.get("candidate_seed_policy")
    registered_seeds = (
        [int(seed) for seed in seed_policy.get("registered_seeds",[])]
        if seed_policy
        else []
    )
    rules,results = [],[]
    candidates,baselines,provenance_by_axis = {},{},{}
    rule_fields = {field.name for field in fields(BenchmarkRule)}
    for raw in registry["rules"]:
        rule = BenchmarkRule(**{key: value for key,value in raw.items() if key in rule_fields})
        rules.append(rule)
        candidate_path = candidate_root_path / f"{rule.axis}.npz"
        baseline_path = baseline_root_path / f"{rule.axis}.npz"
        candidate_manifest_path = candidate_root_path / f"{rule.axis}.provenance.json"
        baseline_manifest_path = baseline_root_path / f"{rule.axis}.provenance.json"
        candidate = load_prediction_bundle(candidate_path)
        baseline = load_prediction_bundle(baseline_path)
        for bundle in (candidate,baseline):
            validate_registered_axis_groups(
                bundle,
                split=raw["split"],
                allowed_groups=raw.get("allowed_groups"),
                exact_groups_required=bool(raw.get("exact_groups_required",False)),
            )
        candidates[rule.axis] = candidate
        baselines[rule.axis] = baseline
        candidate_provenance = validate_prediction_manifest(
            candidate_manifest_path,candidate_path,expected_model=candidate.model
        )
        baseline_provenance = validate_prediction_manifest(
            baseline_manifest_path,baseline_path,expected_model=baseline.model
        )
        ensemble = candidate_provenance.get("manifest",{}).get("seed_ensemble")
        source_contract_ok = bool(
            claim_surface_sha256 is None
            or (
                candidate_provenance.get("manifest",{}).get(
                    "claim_surface_contract_sha256"
                )
                == claim_surface_sha256
                and baseline_provenance.get("manifest",{}).get(
                    "claim_surface_contract_sha256"
                )
                == claim_surface_sha256
            )
        )
        ensemble_ok = True
        if seed_policy:
            ensemble_ok = bool(
                isinstance(ensemble,dict)
                and ensemble.get("registered_seeds") == registered_seeds
                and ensemble.get("aggregation") == seed_policy.get("aggregation")
                and ensemble.get("exact_example_ids") is True
                and [row.get("seed") for row in ensemble.get("components",[])]
                == registered_seeds
            )
        provenance_by_axis[rule.axis] = {
            "ok": candidate_provenance["ok"]
            and baseline_provenance["ok"]
            and ensemble_ok
            and source_contract_ok,
            "candidate": candidate_provenance,
            "baseline": baseline_provenance,
            "registered_seed_ensemble_ok": ensemble_ok,
            "claim_surface_contract_ok": source_contract_ok,
        }
        results.append(
            compare_axis(
                candidate,
                baseline,
                rule,
                split=raw["split"],
                bootstrap=bootstrap,
                seed=20_260_620,
            )
        )

    candidate_models = {bundle.model for bundle in candidates.values()}
    baseline_models = {bundle.model for bundle in baselines.values()}
    if len(candidate_models) != 1:
        raise ValueError(f"candidate model differs across axes: {sorted(candidate_models)}")
    if len(baseline_models) != 1:
        raise ValueError(f"baseline model differs across axes: {sorted(baseline_models)}")
    model_identity_policy = registry.get("comparison_model_policy",{})
    role_policy = model_identity_policy.get(comparison_role)
    if model_identity_policy:
        if not isinstance(role_policy,dict):
            raise ValueError(f"comparison role {comparison_role!r} is not registered")
        candidate_model = next(iter(candidate_models))
        baseline_model = next(iter(baseline_models))
        if (
            candidate_model != role_policy.get("candidate_model")
            or baseline_model != role_policy.get("baseline_model")
        ):
            raise ValueError(
                f"comparison role {comparison_role!r} requires "
                f"{role_policy.get('candidate_model')!r} versus "
                f"{role_policy.get('baseline_model')!r}; received "
                f"{candidate_model!r} versus {baseline_model!r}"
            )

    calibration_axis = registry["calibration"].get("axis")
    if not calibration_axis:
        primary_split = registry["split_policy"]["primary_surface"]
        calibration_axis = next(
            (raw["axis"] for raw in registry["rules"] if raw["split"] == primary_split),
            None,
        )
    if calibration_axis not in candidates:
        raise ValueError(f"calibration axis {calibration_axis!r} has no prediction bundle")
    calibration = interval_calibration(
        candidates[calibration_axis],
        split=registry["split_policy"]["primary_surface"],
        nominal=float(registry["calibration"]["nominal"]),
        minimum_groups=int(registry["calibration"].get("minimum_groups",5)),
        max_width_iqr_multiplier=float(
            registry["calibration"].get("max_width_iqr_multiplier",4.0)
        ),
    )
    calibration["axis"] = calibration_axis
    seed_robustness = {
        "ok": True,
        "required": False,
        "reason": "candidate seed robustness is not registered",
    }
    if seed_policy:
        seed_robustness = {
            "ok": False,
            "required": True,
            "registered_seeds": registered_seeds,
            "aggregation": seed_policy.get("aggregation"),
            "individual_axis_policy": seed_policy.get("individual_axis_policy"),
            "seed_results": {},
            "failures": [],
        }
        roots = [Path(value) for value in (candidate_seed_roots or [])]
        if len(roots) != len(registered_seeds):
            seed_robustness["failures"].append(
                f"received {len(roots)} candidate seed roots; require {len(registered_seeds)}"
            )
        else:
            observed_seeds = set()
            required_axes = {rule.axis for rule in rules if rule.required_for_claim}
            for root in roots:
                root_seed = None
                axis_rows = []
                root_failures = []
                for raw,rule in zip(registry["rules"],rules,strict=True):
                    axis = rule.axis
                    bundle_path = root / f"{axis}.npz"
                    manifest_path = root / f"{axis}.provenance.json"
                    bundle = load_prediction_bundle(bundle_path)
                    validate_registered_axis_groups(
                        bundle,
                        split=raw["split"],
                        allowed_groups=raw.get("allowed_groups"),
                        exact_groups_required=bool(raw.get("exact_groups_required",False)),
                    )
                    manifest_report = validate_prediction_manifest(
                        manifest_path,bundle_path,expected_model=bundle.model
                    )
                    manifest = manifest_report.get("manifest",{})
                    if (
                        claim_surface_sha256 is not None
                        and manifest.get("claim_surface_contract_sha256")
                        != claim_surface_sha256
                    ):
                        root_failures.append(
                            f"axis {axis} claim-surface contract hash drifted"
                        )
                    seed = manifest.get("seed")
                    if not isinstance(seed,int):
                        root_failures.append(f"{axis} lacks an integer seed")
                    elif root_seed is None:
                        root_seed = seed
                    elif seed != root_seed:
                        root_failures.append(
                            f"axis {axis} seed {seed} differs from root seed {root_seed}"
                        )
                    if not manifest_report["ok"]:
                        root_failures.append(
                            f"axis {axis} provenance failed: {manifest_report['failures']}"
                        )
                    ensemble_manifest = provenance_by_axis[axis]["candidate"].get(
                        "manifest",{}
                    ).get("seed_ensemble",{})
                    ensemble_weights = {
                        row.get("seed"): row.get("weights_sha256")
                        for row in ensemble_manifest.get("components",[])
                    }
                    ensemble_raw = {
                        row.get("seed"): row.get("raw_sha256")
                        for row in ensemble_manifest.get("components",[])
                    }
                    if isinstance(seed,int) and ensemble_weights.get(seed) != manifest.get(
                        "weights_sha256"
                    ):
                        root_failures.append(
                            f"axis {axis} seed weight is absent from the frozen ensemble"
                        )
                    if isinstance(seed,int) and ensemble_raw.get(seed) != manifest.get(
                        "raw_prediction_sha256"
                    ):
                        root_failures.append(
                            f"axis {axis} seed raw prediction is absent from the frozen ensemble"
                        )
                    result = compare_axis(
                        bundle,
                        baselines[axis],
                        rule,
                        split=raw["split"],
                        bootstrap=bootstrap,
                        seed=20_260_620,
                    )
                    row = asdict(result)
                    row["strictly_positive_delta"] = bool(
                        np.isfinite(result.delta) and result.delta > 0
                    )
                    axis_rows.append(row)
                    if axis in required_axes and not row["strictly_positive_delta"]:
                        root_failures.append(
                            f"required axis {axis} delta is not strictly positive"
                        )
                if root_seed is None:
                    seed_robustness["failures"].append(
                        f"candidate root {root} has no recoverable seed"
                    )
                    continue
                if root_seed in observed_seeds:
                    root_failures.append(f"duplicate candidate seed root {root_seed}")
                observed_seeds.add(root_seed)
                seed_robustness["seed_results"][str(root_seed)] = {
                    "root": str(root),
                    "axes": axis_rows,
                    "failures": root_failures,
                    "ok": not root_failures,
                }
            if observed_seeds != set(registered_seeds):
                seed_robustness["failures"].append(
                    f"observed seeds {sorted(observed_seeds)}; require {registered_seeds}"
                )
            for seed,row in seed_robustness["seed_results"].items():
                if not row["ok"]:
                    seed_robustness["failures"].append(
                        f"seed {seed} failed individual-axis robustness"
                    )
        seed_robustness["ok"] = not seed_robustness["failures"]
        seed_robustness["reason"] = (
            "all registered seeds improve every required axis"
            if seed_robustness["ok"]
            else "; ".join(seed_robustness["failures"])
        )
    provenance = {
        "ok": all(item["ok"] for item in provenance_by_axis.values()),
        "axes": provenance_by_axis,
        "registered_seed_ensemble_required": bool(seed_policy),
    }
    report = claim_report(
        next(iter(candidate_models)),
        next(iter(baseline_models)),
        results,
        calibration,
        rules,
        provenance=provenance,
        seed_robustness=seed_robustness,
        comparison_role=comparison_role,
        model_identity_policy=role_policy,
    )
    write_claim_report(report,out)
    print(json.dumps(report,indent=2,sort_keys=True))
    return 0 if report["verdict"] == "BEATS_BASELINE" else 2

def run_benchmark_suite(
    *,headline: str,adapter: str,borzoi: str,registry_path: str,out: str
) -> int:
    from .benchmark import claim_suite_report

    registry = json.loads(Path(registry_path).read_text(encoding="utf-8"))
    reports = {
        "headline_enformer": json.loads(Path(headline).read_text(encoding="utf-8")),
        "diagnostic_enformer_adapter": json.loads(
            Path(adapter).read_text(encoding="utf-8")
        ),
        "secondary_borzoi": json.loads(Path(borzoi).read_text(encoding="utf-8")),
    }
    report = claim_suite_report(registry,reports)
    destination = Path(out)
    destination.parent.mkdir(parents=True,exist_ok=True)
    destination.write_text(
        json.dumps(report,indent=2,sort_keys=True) + "\n",encoding="utf-8"
    )
    print(json.dumps(report,indent=2,sort_keys=True))
    return 0 if report["verdict"].startswith("BEATS_ENFORMER") else 2
