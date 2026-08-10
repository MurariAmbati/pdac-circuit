from __future__ import annotations

from dataclasses import replace
import hashlib
import io
import json
from pathlib import Path
import shutil
import tarfile

import numpy as np
import pytest

from pdac_circuit.chromatin.benchmark import (
    BenchmarkRule,
    PredictionBundle,
    align_bundles,
    claim_suite_report,
    compare_axis,
    load_prediction_bundle,
    save_prediction_bundle,
    validate_registered_axis_groups,
)
from pdac_circuit.chromatin.baselines import (
    resolve_borzoi_target_map,
    resolve_enformer_target_map,
)
from pdac_circuit.chromatin.baseline_adapter import (
    EnformerAdapterConfig,
    EnformerStateAdapter,
    adapter_parameter_report,
)
from pdac_circuit.chromatin.adapter_pipeline import (
    predict_enformer_adapter,
    train_enformer_adapter,
)
from pdac_circuit.chromatin.cli import (
    _filter_shards_by_replicate_quality,
    _validate_checkpoint_selection_scope,
    _load_label_free_example_ids,
    _validate_stage_supervision,
    run_benchmark,
    run_compile_index,
)
from pdac_circuit.chromatin.campaign import _glob_summary, build_campaign_plan
from pdac_circuit.chromatin.circuit_audit import (
    audit_circuit_stability,
    write_circuit_stability_audit,
)
from pdac_circuit.chromatin.config import ChromatinModelConfig, ChromatinTrainConfig
from pdac_circuit.chromatin.compile_plan import audit_compiled_splits, verify_compiled_track
from pdac_circuit.chromatin.curriculum import apply_training_stage
from pdac_circuit.chromatin.encode import assay_vector, build_encode_track_specs
from pdac_circuit.chromatin.evaluation import (
    assemble_prediction_bundle,
    conformalize_raw_predictions,
    contrast_raw_predictions,
    ensemble_seed_raw_predictions,
    freeze_evaluation_windows,
    freeze_profile_truth,
    make_state_invariant_raw,
    merge_raw_predictions,
    save_raw_predictions,
)
from pdac_circuit.chromatin.geo import (
    _official_ncbi_ftp_fallback,
    _supplementary_links,
    fetch_geo_plan,
    geo_series_bucket,
    supplementary_url,
)
from pdac_circuit.chromatin.geo_archive import inspect_geo_archive
from pdac_circuit.chromatin.geo_metadata import (
    canonical_genome,
    canonical_state,
    driver_perturbation_vector,
    fetch_geo_soft_metadata,
    parse_geo_soft,
    perturbation_control_family,
    resolve_sample_metadata,
    signed_perturbation_vector,
)
from pdac_circuit.chromatin.losses import correlation_loss, total_chromatin_loss
from pdac_circuit.chromatin.human_cohort import validate_human_cohort_contract
from pdac_circuit.chromatin.claim_surfaces import validate_claim_surface_contract
from pdac_circuit.chromatin.locking import exclusive_artifact_lock
from pdac_circuit.chromatin.model import PDACircuitFormer, build_chromatin_model
from pdac_circuit.chromatin.pairing import compose_paired_shards
from pdac_circuit.chromatin.pair_registry import (
    build_external_perturbation_pair_plan,
    build_intervention_pair_plan,
    materialize_intervention_pair_plan,
    verify_paired_output,
)
from pdac_circuit.chromatin.provenance import write_prediction_manifest
from pdac_circuit.chromatin.protected import (
    authorize_protected_metadata_release,
    validate_protected_study_seal,
)
from pdac_circuit.chromatin.splits import SplitPolicy, assign_split, audit_split_records
from pdac_circuit.chromatin.streaming import (
    ChromatinShardStream,
    IndexedFasta,
    LocalTiledChromatinStream,
    TrackSpec,
    Window,
    genome_windows,
    one_hot_sequence,
    sha256_file,
    split_for_window,
)
from pdac_circuit.chromatin.trainer import (
    MemoryBoundedTrainer,
    load_weights_for_initialization,
)

def _tiny_config() -> ChromatinModelConfig:
    return ChromatinModelConfig(
        sequence_length=1024,
        bin_size=32,
        base_channels=8,
        d_model=48,
        n_layers=3,
        landmark_tokens=8,
        attention_heads=4,
        assay_features=3,
        state_features=6,
        perturbation_features=2,
        signed_perturbation_features=2,
        circuit_factors=5,
        gradient_checkpointing=False,
    )

def test_chromatin_config_rejects_non_power_of_two_bins():
    cfg=ChromatinModelConfig(bin_size=96)
    with pytest.raises(ValueError, match="power of two"):
        cfg.validate()
    ChromatinTrainConfig().validate()
    with pytest.raises(ValueError, match="loss weights"):
        ChromatinTrainConfig(loss_state_graph=-0.1).validate()
    with pytest.raises(ValueError, match="architecture"):
        replace(_tiny_config(), architecture="transformer").validate()
    with pytest.raises(ValueError, match="even integer"):
        replace(_tiny_config(), landmark_tokens=7).validate()
    with pytest.raises(ValueError, match="divisible by half"):
        replace(_tiny_config(), landmark_tokens=12).validate()
    with pytest.raises(ValueError, match="landmark_routing"):
        replace(_tiny_config(), landmark_routing="learned_router").validate()

def test_dual_statistic_landmarks_preserve_narrow_regulatory_peaks():
    import torch

    cfg=_tiny_config()
    model=PDACircuitFormer(cfg)
    mixer=next(iter(model.mixers.values()))
    x=torch.zeros(1, cfg.d_model, cfg.n_bins)
    x[0, 0, 0]=10.0

    landmarks=mixer._content_routed_landmarks(x)

    assert landmarks.shape == (1, cfg.landmark_tokens, cfg.d_model)
    assert landmarks[0, 1, 0] > 5 * landmarks[0, 0, 0]

    constant=torch.full_like(x, 3.0)
    constant_landmarks=mixer._content_routed_landmarks(constant)
    assert torch.allclose(constant_landmarks[:, 0::2], constant_landmarks[:, 1::2])

    mean_model=PDACircuitFormer(replace(cfg, landmark_routing="mean_only"))
    mean_mixer=next(iter(mean_model.mixers.values()))
    mean_landmarks=mean_mixer._content_routed_landmarks(x)
    assert torch.equal(mean_landmarks[:, 0::2], mean_landmarks[:, 1::2])
    assert sum(p.numel() for p in mean_model.parameters()) == sum(
        p.numel() for p in model.parameters()
    )

    regions=cfg.landmark_tokens // 2
    bias=mixer._relative_landmark_bias(
        cfg.n_bins, regions, device=x.device, dtype=x.dtype
    )
    assert bias.shape == (cfg.attention_heads, cfg.n_bins, cfg.landmark_tokens)
    span=cfg.n_bins // regions
    assert torch.allclose(bias[:, 0, 0], bias[:, span, 2])
    assert bias[0, 0, 0] > bias[0, 0, -1]

def test_campaign_plan_materializes_all_seed_stage_dependencies(tmp_path):
    (tmp_path / "configs").mkdir()
    (tmp_path / "configs" / "profile.json").write_text("{}", encoding="utf-8")
    (tmp_path / "configs" / "ablation.json").write_text("{}", encoding="utf-8")
    for stage in (
        "healthy_prior",
        "progression_state_residual",
        "signed_intervention_residual",
        "human_state_adaptation",
    ):
        directory=tmp_path / "data" / stage
        directory.mkdir(parents=True)
        np.savez_compressed(directory / "shard.npz", value=np.asarray([1]))
        (directory / "manifest.json").write_text(
            json.dumps(
                {
                    "sequence_length": 196608,
                    "negative_keep_probability": 1.0,
                    "track": {
                        "assay_features": [0.0] * 12,
                        "state_features": [0.0] * 18,
                        "perturbation_features": [0.0] * 22,
                    },
                }
            ),
            encoding="utf-8",
        )
        index_path=directory / "index.json"
        index_path.write_text(
            json.dumps(
                {
                    "schema": "pdac-circuit.geo-track-specs/1",
                    "written": [{}],
                    "failures": [],
                }
            ),
            encoding="utf-8",
        )
        (directory / "_COMPLETE.json").write_text(
            json.dumps(
                {
                    "schema": "pdac-circuit.compiled-collection-completion/1",
                    "successful": True,
                    "track_index": str(index_path.relative_to(tmp_path)),
                    "track_index_sha256": sha256_file(index_path),
                    "sequence_length": 196608,
                    "bin_size": 128,
                    "registered_tracks": 1,
                    "verified_tracks": 1,
                    "tracks": [{"source_sha256_verified": True}],
                }
            ),
            encoding="utf-8",
        )
    shutil.copytree(
        tmp_path / "data" / "human_state_adaptation",
        tmp_path / "data" / "human_validation",
    )
    campaign={
        "schema": "pdac-circuit.chromatin-campaign/1",
        "seeds": [11, 22, 33],
        "profiles": [
            {
                "config": "configs/profile.json",
                "hardware": "test",
                "minimum_free_vram_gb": 10,
            }
        ],
        "ablation_profiles": [
            {
                "config": "configs/ablation.json",
                "hardware": "test",
                "minimum_free_vram_gb": 10,
                "reuse_healthy_from": "configs/profile.json",
            }
        ],
        "selection": {"maximum_tuning_uses_of_test_surfaces": 0},
        "curriculum": [{"stage": 5, "name": "freeze_and_one_shot_test"}],
        "execution": {
            "checkpoint_root": "models/campaign",
            "require_complete_markers": True,
            "stage_compile_contracts": {
                stage: {"negative_keep_probability": 1.0}
                for stage in (
                    "healthy_prior",
                    "progression_state_residual",
                    "signed_intervention_residual",
                    "human_state_adaptation",
                )
            },
            "stage_validation_studies": {
                "healthy_prior": None,
                "progression_state_residual": None,
                "signed_intervention_residual": None,
                "human_state_adaptation": ["GSE272463"],
            },
            "stage_data": {
                stage: (
                    [
                        "data/human_state_adaptation/**/*.npz",
                        "data/human_validation/**/*.npz",
                    ]
                    if stage == "human_state_adaptation"
                    else f"data/{stage}/**/*.npz"
                )
                for stage in (
                    "healthy_prior",
                    "progression_state_residual",
                    "signed_intervention_residual",
                    "human_state_adaptation",
                )
            },
            "data_binding": "hash shards",
            "resume_policy": "same stage",
            "parallelism": "one process",
        },
    }
    campaign_path=tmp_path / "configs" / "campaign.json"
    campaign_path.write_text(json.dumps(campaign), encoding="utf-8")
    out=tmp_path / "plan.json"
    report=build_campaign_plan(
        tmp_path, "configs/campaign.json", "configs/profile.json", out
    )
    payload=json.loads(out.read_text(encoding="utf-8"))
    assert report["nodes"] == 12
    assert len(payload["plan_sha256"]) == 64
    assert payload["nodes"][0]["depends_on"] == []
    assert payload["nodes"][1]["depends_on"] == [payload["nodes"][0]["id"]]
    assert "--seed" in payload["nodes"][0]["argv"]
    assert payload["nodes"][0]["minimum_free_vram_gb"] == 10.0
    vram_index=payload["nodes"][0]["argv"].index("--min-free-vram-gb")
    assert payload["nodes"][0]["argv"][vram_index : vram_index + 2] == [
        "--min-free-vram-gb",
        "10.0",
    ]
    assert payload["nodes"][0]["selected_checkpoint"].endswith("best.pt")
    assert payload["nodes"][0]["resume_checkpoint"].endswith("latest.pt")
    assert payload["nodes"][0]["selected_checkpoint"] in payload["nodes"][1]["argv"]
    human_node=next(
        node for node in payload["nodes"] if node["stage"] == "human_state_adaptation"
    )
    assert human_node["argv"].count("--shards") == 2
    validation_index=human_node["argv"].index("--validation-study")
    assert human_node["argv"][validation_index : validation_index + 2] == [
        "--validation-study",
        "GSE272463",
    ]
    assert human_node["checkpoint_validation_studies"] == ["GSE272463"]
    assert human_node["data"]["source_count"] == 2
    assert payload["nodes"][0]["runnable_now"] is True
    assert payload["nodes"][1]["runnable_now"] is False
    (tmp_path / "data" / "healthy_prior" / "_COMPLETE.json").unlink()
    incomplete=tmp_path / "incomplete-plan.json"
    build_campaign_plan(
        tmp_path, "configs/campaign.json", "configs/profile.json", incomplete
    )
    incomplete_payload=json.loads(incomplete.read_text(encoding="utf-8"))
    assert incomplete_payload["nodes"][0]["data_compatible_now"] is False
    assert incomplete_payload["nodes"][0]["runnable_now"] is False
    marker_source=tmp_path / "data" / "progression_state_residual" / "_COMPLETE.json"
    (tmp_path / "data" / "healthy_prior" / "_COMPLETE.json").write_text(
        marker_source.read_text(encoding="utf-8"), encoding="utf-8"
    )
    (tmp_path / "data" / "healthy_prior" / "manifest.json").write_text(
        json.dumps({"sequence_length": 524288}), encoding="utf-8"
    )
    incompatible=tmp_path / "incompatible-plan.json"
    build_campaign_plan(
        tmp_path, "configs/campaign.json", "configs/profile.json", incompatible
    )
    incompatible_payload=json.loads(incompatible.read_text(encoding="utf-8"))
    assert incompatible_payload["nodes"][0]["data_compatible_now"] is False
    assert incompatible_payload["nodes"][0]["runnable_now"] is False
    reused=tmp_path / "reused-plan.json"
    build_campaign_plan(
        tmp_path, "configs/campaign.json", "configs/ablation.json", reused
    )
    reused_payload=json.loads(reused.read_text(encoding="utf-8"))
    assert reused_payload["node_count"] == 9
    assert reused_payload["nodes"][0]["ordinal"] == 2
    assert reused_payload["nodes"][0]["depends_on"] == [
        "profile.seed-11.01-healthy_prior"
    ]
    assert reused_payload["nodes"][0]["argv"][-1].replace("\\", "/").endswith(
        "profile/seed-11/01-healthy_prior/best.pt"
    )

def test_seed_ensemble_requires_every_registered_seed_and_exact_cohort(tmp_path):
    seeds=[11, 22, 33]
    paths=[]
    for index, seed in enumerate(seeds, start=1):
        path=tmp_path / f"seed-{seed}.npz"
        ids=np.asarray(["e0", "e1"])
        prediction=np.asarray([[float(index)], [float(index + 10)]])
        if seed == 22:
            ids=ids[::-1]
            prediction=prediction[::-1]
        save_raw_predictions(
            path,
            model="PDACircuitFormer",
            example_id=ids,
            prediction=prediction,
            metadata={
                "schema": "pdac-circuit.raw-predictions/1",
                "model": "PDACircuitFormer",
                "model_version": f"seed-{seed}",
                "weights_sha256": str(index) * 64,
                "track_mapping_sha256": "a" * 64,
                "component": "mean",
                "crop_bins": 896,
                "reverse_complement": True,
                "ablate_state_residual": False,
                "ablate_intervention_residual": False,
                "local_tiling": None,
                "label_free_cohort": {"sha256": "b" * 64, "examples": 2},
                "seed": seed,
                "command": f"predict --seed {seed}",
            },
        )
        paths.append(path)
    out=tmp_path / "ensemble.npz"
    report=ensemble_seed_raw_predictions(
        paths,
        out,
        registered_seeds=seeds,
        command="ensemble",
    )
    assert report["registered_seeds"] == seeds
    with np.load(out, allow_pickle=False) as archive:
        assert archive["example_id"].astype(str).tolist() == ["e0", "e1"]
        assert np.allclose(archive["prediction"], [[2.0], [12.0]])
        metadata=json.loads(str(archive["metadata"].item()))
    assert metadata["seed_ensemble"]["registered_seeds"] == seeds
    assert [row["seed"] for row in metadata["seed_ensemble"]["components"]] == seeds
    with pytest.raises(ValueError, match="require 3"):
        ensemble_seed_raw_predictions(
            paths[:2],
            tmp_path / "incomplete.npz",
            registered_seeds=seeds,
            command="ensemble",
        )

def test_exclusive_artifact_lock_is_fail_closed_and_cleans_up(tmp_path):
    lock=tmp_path / ".artifact.lock"
    with exclusive_artifact_lock(lock):
        owner=json.loads(lock.read_text(encoding="utf-8"))
        assert owner["pid"] > 0
        assert owner["created_at"]
        with pytest.raises(RuntimeError, match="already locked"):
            with exclusive_artifact_lock(lock):
                pass
        assert lock.exists()
    assert not lock.exists()

def test_ncbi_https_download_has_narrow_official_ftp_fallback():
    https="https://ftp.ncbi.nlm.nih.gov/geo/series/GSE99nnn/GSE99311/suppl/x.tar"
    assert _official_ncbi_ftp_fallback(https) == https.replace("https://", "ftp://")
    assert _official_ncbi_ftp_fallback("https://example.org/x.tar") is None

def test_pdacircuitformer_normal_is_explicit_baseline():
    import torch

    cfg=_tiny_config()
    model=PDACircuitFormer(cfg).eval()
    sequence=torch.zeros(2, 4, cfg.sequence_length)
    sequence[:, 0, :]=1
    assay=torch.zeros(2, cfg.assay_features)
    state=torch.zeros(2, cfg.state_features)
    perturb=torch.zeros(2, cfg.perturbation_features)
    disease=torch.tensor([0.0, 1.0])
    with torch.no_grad():
        output=model(sequence, assay, state, perturb, disease)
    assert output.mean.shape == (2, cfg.n_bins)
    assert output.circuit_factors.shape == (2, cfg.circuit_factors)
    assert output.intervention_factors.shape == (2, cfg.circuit_factors)
    assert output.intervention_axis_potentials.shape == (
        2,
        cfg.signed_perturbation_features,
        cfg.circuit_factors,
    )
    assert torch.equal(output.circuit_factors, torch.zeros_like(output.circuit_factors))
    assert torch.equal(output.state_residual, torch.zeros_like(output.state_residual))
    assert torch.equal(output.residual[0], torch.zeros_like(output.residual[0]))
    assert torch.equal(
        output.perturbation_residual,
        torch.zeros_like(output.perturbation_residual),
    )
    assert torch.equal(
        output.intervention_factors,
        torch.zeros_like(output.intervention_factors),
    )
    assert torch.allclose(output.mean[0], output.baseline[0])
    assert torch.isfinite(output.mean).all()
    with torch.no_grad():
        ensemble=model.predict_reverse_complement_ensemble(
            sequence, assay, state, perturb, disease
        )
    assert torch.isfinite(ensemble.log_variance).all()
    assert torch.all(ensemble.log_variance.exp() > 0)

def test_direct_conditional_cnn_is_a_profile_matched_noncausal_control():
    import torch

    cfg=replace(
        _tiny_config(),
        architecture="direct_conditional_cnn",
        sequence_length=2048,
        bin_size=128,
    )
    model=build_chromatin_model(cfg).eval()
    assert model.model_name == "DirectConditionalCNN"
    assert not hasattr(model, "mixers")
    sequence=torch.zeros(2, 4, cfg.sequence_length)
    sequence[:, 0, :]=1
    assay=torch.zeros(2, cfg.assay_features)
    state=torch.zeros(2, cfg.state_features)
    perturbation=torch.zeros(2, cfg.perturbation_features)
    state[1, 2]=1.0
    with torch.no_grad():
        output=model(sequence, assay, state, perturbation, torch.ones(2))
        ensemble=model.predict_reverse_complement_ensemble(
            sequence, assay, state, perturbation, torch.ones(2)
        )
    assert output.mean.shape == (2, 16)
    assert torch.equal(output.state_residual, torch.zeros_like(output.state_residual))
    assert torch.equal(
        output.perturbation_residual,
        torch.zeros_like(output.perturbation_residual),
    )
    assert torch.isfinite(ensemble.mean).all()
    stage=apply_training_stage(model, "progression_state_residual")
    assert stage["trainable_parameters"] == stage["total_parameters"]

def test_local_tiling_slices_profiles_and_preserves_parent_identity():
    class Source:
        epoch=0

        def __iter__(self):
            yield {
                "sequence": np.arange(4 * 32, dtype=np.float32).reshape(4, 32),
                "target": np.arange(8, dtype=np.float32),
                "signal_mask": np.ones(8, dtype=bool),
                "example_id": "parent",
                "start": 100,
                "end": 132,
            }

    source=Source()
    stream=LocalTiledChromatinStream(source, tile_bp=8, bin_size=4)
    tiles=list(stream)
    assert len(tiles) == 4
    assert [tile["target"].tolist() for tile in tiles] == [
        [0.0, 1.0],
        [2.0, 3.0],
        [4.0, 5.0],
        [6.0, 7.0],
    ]
    assert {tile["parent_example_id"] for tile in tiles} == {"parent"}
    assert [tile["tile_index"] for tile in tiles] == [0, 1, 2, 3]
    assert [(tile["start"], tile["end"]) for tile in tiles] == [
        (100, 108),
        (108, 116),
        (116, 124),
        (124, 132),
    ]
    stream.epoch=3
    assert source.epoch == 3

def test_intervention_operator_is_structurally_control_affine_and_odd():
    import torch

    cfg=_tiny_config()
    model=PDACircuitFormer(cfg).eval()
    sequence=torch.zeros(1, 4, cfg.sequence_length)
    sequence[:, 2, :]=1
    assay=torch.zeros(1, cfg.assay_features)
    state=torch.zeros(1, cfg.state_features)
    state[:, 2]=1.0
    positive=torch.zeros(1, cfg.perturbation_features)
    positive[:, 0]=1.0
    negative=positive.clone()
    negative[:, : cfg.signed_perturbation_features] *= -1
    control=positive.clone().zero_()
    with torch.no_grad():
        model.intervention_head[1].weight.fill_(0.05)
        model.intervention_head[1].bias.fill_(0.02)
        plus=model(sequence, assay, state, positive, torch.ones(1))
        minus=model(sequence, assay, state, negative, torch.ones(1))
        zero=model(sequence, assay, state, control, torch.ones(1))
    assert torch.equal(
        zero.intervention_factors, torch.zeros_like(zero.intervention_factors)
    )
    assert torch.equal(
        zero.perturbation_residual, torch.zeros_like(zero.perturbation_residual)
    )
    assert torch.allclose(
        plus.intervention_axis_potentials,
        minus.intervention_axis_potentials,
        atol=0,
        rtol=0,
    )
    assert torch.allclose(plus.intervention_factors, -minus.intervention_factors)
    assert torch.allclose(plus.perturbation_residual, -minus.perturbation_residual)
    with torch.no_grad():
        no_intervention=model(
            sequence,
            assay,
            state,
            positive,
            torch.ones(1),
            ablate_intervention_residual=True,
        )
    assert torch.equal(
        no_intervention.perturbation_residual,
        torch.zeros_like(no_intervention.perturbation_residual),
    )

def test_circuit_coefficients_are_a_causal_residual_bottleneck():
    import torch

    cfg=_tiny_config()
    model=PDACircuitFormer(cfg).eval()
    sequence=torch.zeros(1, 4, cfg.sequence_length)
    sequence[:, 0, :]=1
    assay=torch.zeros(1, cfg.assay_features)
    state=torch.zeros(1, cfg.state_features)
    state[:, 2]=1.0
    state[:, -2]=1.0
    perturbation=torch.zeros(1, cfg.perturbation_features)
    with torch.no_grad():
        model.circuit_head[1].weight.fill_(0.05)
        model.circuit_head[1].bias.fill_(0.05)
        original=model(sequence, assay, state, perturbation, torch.ones(1))
        model.circuit_head[1].weight.zero_()
        model.circuit_head[1].bias.zero_()
        ablated=model(sequence, assay, state, perturbation, torch.ones(1))
    assert original.counterfactual_factors.shape == (1, 4, cfg.circuit_factors)
    assert original.domain_counterfactual_factors.shape == (1, 2, cfg.circuit_factors)
    assert torch.count_nonzero(original.state_residual) > 0
    assert torch.count_nonzero(ablated.state_residual) == 0

def test_healthy_baseline_sees_species_but_not_disease_state():
    import torch

    cfg=_tiny_config()
    model=PDACircuitFormer(cfg).eval()
    sequence=torch.zeros(2, 4, cfg.sequence_length)
    sequence[:, 1, :]=1
    assay=torch.zeros(2, cfg.assay_features)
    state=torch.zeros(2, cfg.state_features)
    state[0, 0]=1.0
    state[1, 3]=1.0
    state[:, -2]=1.0
    with torch.no_grad():
        healthy=model(
            sequence[:1],
            assay[:1],
            state[:1],
            torch.zeros(1, cfg.perturbation_features),
            torch.tensor([0.0]),
        )
        disease=model(
            sequence[1:],
            assay[1:],
            state[1:],
            torch.zeros(1, cfg.perturbation_features),
            torch.tensor([1.0]),
        )
    assert torch.equal(healthy.baseline, disease.baseline)

def test_curriculum_stages_freeze_exact_causal_components(tmp_path):
    import torch

    cfg=_tiny_config()
    model=PDACircuitFormer(cfg)
    healthy=apply_training_stage(model, "healthy_prior")
    parameters=dict(model.named_parameters())
    assert parameters["baseline_head.2.weight"].requires_grad
    assert not parameters["circuit_head.1.weight"].requires_grad
    assert not parameters["intervention_head.1.weight"].requires_grad

    progression=apply_training_stage(model, "progression_state_residual")
    assert parameters["circuit_head.1.weight"].requires_grad
    assert not parameters["intervention_head.1.weight"].requires_grad
    assert progression["trainable_parameters"] > healthy["trainable_parameters"]

    apply_training_stage(model, "signed_intervention_residual")
    assert not parameters["baseline_head.2.weight"].requires_grad
    assert not parameters["circuit_head.1.weight"].requires_grad
    assert parameters["intervention_head.1.weight"].requires_grad

    full=apply_training_stage(model, "human_state_adaptation")
    assert all(parameter.requires_grad for parameter in model.parameters())
    assert full["trainable_parameters"] == full["total_parameters"]

    train=ChromatinTrainConfig(
        epochs=1,
        max_steps=1,
        warmup_steps=1,
        checkpoint_every=1,
        eval_every=1,
        gradient_accumulation=1,
        device="cpu",
        amp_dtype="float32",
    )
    healthy_model=PDACircuitFormer(cfg)
    apply_training_stage(healthy_model, "healthy_prior")
    healthy_trainer=MemoryBoundedTrainer(
        healthy_model, cfg, train, training_stage="healthy_prior"
    )
    checkpoint=tmp_path / "healthy.pt"
    healthy_trainer.save_checkpoint(checkpoint)
    progression_model=PDACircuitFormer(cfg)
    apply_training_stage(progression_model, "progression_state_residual")
    progression_trainer=MemoryBoundedTrainer(
        progression_model, cfg, train, training_stage="progression_state_residual"
    )
    with pytest.raises(ValueError, match="checkpoint/config hash mismatch"):
        progression_trainer.load_checkpoint(checkpoint)
    initialization=load_weights_for_initialization(
        progression_model, cfg, checkpoint, device="cpu"
    )
    assert initialization["source_training_stage"] == "healthy_prior"
    assert len(initialization["checkpoint_sha256"]) == 64
    variant_train=replace(train, loss_state_graph=0.0)
    variant_trainer=MemoryBoundedTrainer(
        progression_model,
        cfg,
        variant_train,
        training_stage="progression_state_residual",
        initialization_provenance=initialization,
    )
    variant_checkpoint=tmp_path / "variant.pt"
    variant_trainer.save_checkpoint(variant_checkpoint)
    saved_variant=torch.load(variant_checkpoint, map_location="cpu", weights_only=False)
    assert saved_variant["initialization_provenance"] == initialization
    same_stage_different_data=MemoryBoundedTrainer(
        PDACircuitFormer(cfg),
        cfg,
        train,
        training_stage="healthy_prior",
        data_fingerprint={"sha256": "different"},
    )
    with pytest.raises(ValueError, match="checkpoint/data fingerprint mismatch"):
        same_stage_different_data.load_checkpoint(checkpoint)

def test_chromatin_loss_is_finite_and_differentiable():
    import torch

    cfg=_tiny_config()
    model=PDACircuitFormer(cfg).train()
    sequence=torch.zeros(1, 4, cfg.sequence_length)
    sequence[:, 2, :]=1
    output=model(
        sequence,
        torch.zeros(1, cfg.assay_features),
        torch.zeros(1, cfg.state_features),
        torch.zeros(1, cfg.perturbation_features),
        torch.ones(1),
    )
    target=torch.linspace(0, 2, cfg.n_bins)[None]
    loss, parts=total_chromatin_loss(output, target)
    loss.backward()
    assert torch.isfinite(loss)
    assert set(parts) == {
        "profile",
        "correlation",
        "uncertainty",
        "residual_delta",
        "perturbation_delta",
        "healthy_zero",
        "state_graph",
        "domain_invariance",
    }
    assert any(parameter.grad is not None for parameter in model.parameters())

def test_masked_correlation_ignores_missing_bins():
    import torch

    target=torch.tensor([[1.0, 2.0, 999.0, 999.0]])
    prediction=torch.tensor([[1.0, 2.0, -999.0, -999.0]], requires_grad=True)
    mask=torch.tensor([[True, True, False, False]])
    loss=correlation_loss(prediction, target, mask)
    assert loss.item() == pytest.approx(0.0, abs=1e-6)
    loss.backward()
    assert prediction.grad is not None

def test_completed_checkpoint_resume_takes_no_extra_step(tmp_path):
    import torch

    model_cfg=_tiny_config()
    train_cfg=ChromatinTrainConfig(
        micro_batch_size=1,
        gradient_accumulation=1,
        epochs=2,
        warmup_steps=1,
        max_steps=1,
        checkpoint_every=1,
        eval_every=1,
        amp_dtype="float32",
        device="cpu",
    )
    batch={
        "sequence": torch.nn.functional.one_hot(
            torch.zeros((1, model_cfg.sequence_length), dtype=torch.long), num_classes=4
        )
        .permute(0, 2, 1)
        .float(),
        "target": torch.linspace(0, 1, model_cfg.n_bins)[None],
        "signal_mask": torch.ones((1, model_cfg.n_bins), dtype=torch.bool),
        "assay_features": torch.zeros((1, model_cfg.assay_features)),
        "state_features": torch.zeros((1, model_cfg.state_features)),
        "perturbation_features": torch.zeros((1, model_cfg.perturbation_features)),
        "disease_mask": torch.zeros(1),
        "sample_group": ["donor-a"],
    }
    first=MemoryBoundedTrainer(PDACircuitFormer(model_cfg), model_cfg, train_cfg)
    first_report=first.fit([batch], tmp_path, validation_loader=[batch])
    assert first_report["optimizer_step"] == 1
    assert first_report["last_validation"]["groups"] == 1
    assert (
        first_report["last_validation"]["selection_metric"]
        == "independent_group_mean_log_profile_loss"
    )
    assert first_report["last_validation"]["group_mean_loss"] == pytest.approx(
        first_report["last_validation"]["parts"]["profile"]
    )
    assert "objective_example_mean_loss" in first_report["last_validation"]
    assert (tmp_path / "best.pt").is_file()

    resumed=MemoryBoundedTrainer(PDACircuitFormer(model_cfg), model_cfg, train_cfg)
    resumed_report=resumed.fit([batch], tmp_path, resume=True, validation_loader=[batch])
    assert resumed_report["already_complete"] is True
    assert resumed_report["optimizer_step"] == 1
    assert resumed_report["global_step"] == 1

def test_nested_split_policy_assigns_joint_and_external_surfaces():
    policy=SplitPolicy(
        held_out_sample_groups=frozenset({"donor-h"}),
        external_studies=frozenset({"study-external"}),
    )
    records=[
        {"chrom": "chr1", "start": 0, "end": 100, "sample_group": "donor-a", "study": "train"},
        {"chrom": "chr1", "start": 100, "end": 200, "sample_group": "donor-h", "study": "train"},
        {"chrom": "chr8", "start": 0, "end": 100, "sample_group": "donor-h", "study": "train"},
        {"chrom": "chr8", "start": 100, "end": 200, "sample_group": "donor-a", "study": "train"},
        {"chrom": "chr1", "start": 200, "end": 300, "sample_group": "donor-x", "study": "study-external"},
    ]
    assert [assign_split(record, policy) for record in records] == [
        "train",
        "state_test",
        "joint_locus_state_test",
        "locus_test",
        "external_study_test",
    ]
    assert audit_split_records(records, policy)["ok"]

def test_compiler_split_roles_are_fail_closed(tmp_path):
    track_file=tmp_path / "track.bigWig"
    track_file.write_bytes(b"placeholder")
    base={
        "accession": "ENCFFTEST",
        "path": str(track_file),
        "assay_features": (1.0,),
        "state_features": (1.0,),
        "perturbation_features": (1.0,),
        "sample_group": "donor-a",
        "study": "study-a",
        "released": "2026-01-01",
        "disease": True,
    }
    train_track=TrackSpec(**base)
    validation_track=TrackSpec(**base, split_role="validation_study")
    held_out_track=TrackSpec(**base, split_role="held_out_state")
    external_track=TrackSpec(**base, split_role="external_study")
    assert split_for_window(train_track, Window("chr6", 0, 100)) == "validation"
    assert split_for_window(train_track, Window("chr8", 0, 100)) == "locus_test"
    assert split_for_window(train_track, Window("chr1", 0, 100)) == "train"
    assert split_for_window(validation_track, Window("chr1", 0, 100)) == "validation"
    assert split_for_window(validation_track, Window("chr8", 0, 100)) == "validation"
    assert (
        split_for_window(held_out_track, Window("chr8", 0, 100))
        == "joint_locus_state_test"
    )
    assert split_for_window(held_out_track, Window("chr1", 0, 100)) == "state_test"
    assert (
        split_for_window(external_track, Window("chr1", 0, 100))
        == "external_study_test"
    )

def test_indexed_fasta_fetch_and_one_hot(tmp_path):
    fasta=tmp_path / "tiny.fa"
    fasta.write_bytes(b">chr1\nACGTACGT\nTTGGCCAA\n")
    (tmp_path / "tiny.fa.fai").write_text("chr1\t16\t6\t8\t9\n", encoding="ascii")
    reader=IndexedFasta(fasta)
    assert reader.fetch("chr1", 2, 14) == "GTACGTTTGGCC"
    encoded=one_hot_sequence("ACGTN")
    assert encoded.shape == (4, 5)
    assert np.all(encoded[:, :4].sum(axis=0) == 1)
    assert encoded[:, 4].sum() == 0

def test_shard_stream_only_pads_legacy_interventions_when_entirely_zero(
    tmp_path, monkeypatch
):
    monkeypatch.setattr(IndexedFasta, "assert_genome", lambda self, genome: None)
    fasta=tmp_path / "tiny.fa"
    fasta.write_bytes(b">chr1\nACGTACGT\n")
    (tmp_path / "tiny.fa.fai").write_text("chr1\t8\t6\t8\t9\n", encoding="ascii")
    shard_path=tmp_path / "shard.npz"

    def write(perturbation):
        np.savez_compressed(
            shard_path,
            start=np.asarray([0]),
            end=np.asarray([8]),
            chrom=np.asarray(["chr1"]),
            genome=np.asarray(["hg38"]),
            split=np.asarray(["train"]),
            assay_features=np.zeros((1, 2), dtype=np.float16),
            state_features=np.zeros((1, 3), dtype=np.float16),
            perturbation_features=np.asarray([perturbation], dtype=np.float16),
            disease_mask=np.zeros(1, dtype=np.float16),
            target=np.zeros((1, 1), dtype=np.float16),
            valid=np.ones((1, 1), dtype=np.uint8),
        )

    dimensions={
        "assay_features": 2,
        "state_features": 3,
        "perturbation_features": 4,
    }
    write([0.0, 0.0])
    example=next(
        iter(
            ChromatinShardStream(
                [shard_path],
                fasta,
                shuffle=False,
                conditioning_dimensions=dimensions,
            )
        )
    )
    assert np.array_equal(example["perturbation_features"], np.zeros(4))

    write([1.0, 0.0])
    with pytest.raises(ValueError, match="perturbation_features has 2 features"):
        next(
            iter(
                ChromatinShardStream(
                    [shard_path],
                    fasta,
                    shuffle=False,
                    conditioning_dimensions=dimensions,
                )
            )
        )

def test_shard_stream_enforces_patient_study_validation_scope(tmp_path, monkeypatch):
    monkeypatch.setattr(IndexedFasta, "assert_genome", lambda self, genome: None)
    fasta=tmp_path / "tiny.fa"
    fasta.write_bytes(b">chr1\nACGTACGT\n")
    (tmp_path / "tiny.fa.fai").write_text("chr1\t8\t6\t8\t9\n", encoding="ascii")
    shard_path=tmp_path / "shard.npz"

    def write(splits):
        n=len(splits)
        np.savez_compressed(
            shard_path,
            start=np.zeros(n, dtype=np.int64),
            end=np.full(n, 8, dtype=np.int64),
            chrom=np.asarray(["chr1"] * n),
            genome=np.asarray(["hg38"] * n),
            split=np.asarray(splits),
            study=np.asarray(["GSE272463", "GSE149103"][:n]),
            assay_features=np.zeros((n, 2), dtype=np.float16),
            state_features=np.zeros((n, 3), dtype=np.float16),
            perturbation_features=np.zeros((n, 4), dtype=np.float16),
            disease_mask=np.ones(n, dtype=np.float16),
            target=np.zeros((n, 1), dtype=np.float16),
            valid=np.ones((n, 1), dtype=np.uint8),
        )

    dimensions={
        "assay_features": 2,
        "state_features": 3,
        "perturbation_features": 4,
    }
    write(["validation", "validation"])
    selected=list(
        ChromatinShardStream(
            [shard_path],
            fasta,
            shuffle=False,
            include_splits={"validation"},
            include_studies={"GSE272463"},
            validation_only_studies={"GSE272463"},
            conditioning_dimensions=dimensions,
        )
    )
    assert [example["study"] for example in selected] == ["GSE272463"]
    training=list(
        ChromatinShardStream(
            [shard_path],
            fasta,
            shuffle=False,
            include_splits={"train"},
            exclude_studies={"GSE272463"},
            validation_only_studies={"GSE272463"},
            conditioning_dimensions=dimensions,
        )
    )
    assert training == []

    write(["train", "validation"])
    with pytest.raises(ValueError, match="validation-only study GSE272463"):
        next(
            iter(
                ChromatinShardStream(
                    [shard_path],
                    fasta,
                    shuffle=False,
                    exclude_studies={"GSE272463"},
                    validation_only_studies={"GSE272463"},
                    conditioning_dimensions=dimensions,
                )
            )
        )

def test_checkpoint_selection_scope_fails_before_training_on_malformed_patient_shard(
    tmp_path,
):
    parent=tmp_path / "GSM-patient"
    parent.mkdir()
    shard_path=parent / "shard-00000.npz"
    manifest_path=parent / "manifest.json"
    manifest_path.write_text(
        json.dumps(
            {
                "track": {
                    "study": "GSE272463",
                    "split_role": "validation_study",
                }
            }
        ),
        encoding="utf-8",
    )

    def write(split):
        np.savez_compressed(
            shard_path,
            study=np.asarray(["GSE272463"]),
            split=np.asarray([split]),
        )

    write("validation")
    report=_validate_checkpoint_selection_scope([shard_path], ["GSE272463"])
    assert report == {
        "schema": "pdac-circuit.checkpoint-selection-scope/1",
        "studies": ["GSE272463"],
        "profiles": 1,
        "shards": 1,
        "examples": 1,
        "split": "validation",
        "excluded_from_gradients": True,
    }
    with pytest.raises(ValueError, match="absent from compiled shards"):
        _validate_checkpoint_selection_scope([shard_path], ["GSE000000"])
    write("train")
    with pytest.raises(ValueError, match="expected only GSE272463/validation"):
        _validate_checkpoint_selection_scope([shard_path], ["GSE272463"])

def test_genome_windows_stream_and_reservoir_are_bounded_and_deterministic():
    sizes={"chr1": 1000, "chr2": 1000}
    streamed=genome_windows(
        sizes, sequence_length=100, stride=100, chromosomes=("chr1", "chr2")
    )
    assert not isinstance(streamed, list)
    assert len(list(streamed)) == 20
    first=list(
        genome_windows(
            sizes,
            sequence_length=100,
            stride=10,
            chromosomes=("chr1", "chr2"),
            max_windows=5,
            seed=7,
        )
    )
    second=list(
        genome_windows(
            sizes,
            sequence_length=100,
            stride=10,
            chromosomes=("chr1", "chr2"),
            max_windows=5,
            seed=7,
        )
    )
    assert first == second
    assert len(first) == 5

def test_compiled_track_verifier_checks_hashes_and_native_schema(tmp_path):
    config_path=tmp_path / "config.json"
    config_path.write_text(
        json.dumps(
            {
                    "model": {
                        "sequence_length": 1024,
                        "bin_size": 32,
                        "landmark_tokens": 8,
                        "state_features": 6,
                    "domain_state_features": 2,
                    "perturbation_features": 2,
                    "signed_perturbation_features": 2,
                }
            }
        ),
        encoding="utf-8",
    )
    source=tmp_path / "track.bigWig"
    source.write_bytes(b"source")
    source_hash=sha256_file(source)
    spec={
        "accession": "TRACK1",
        "path": str(source),
        "assay_features": [0.0] * 12,
        "state_features": [0.0] * 6,
        "perturbation_features": [0.0] * 2,
        "sample_group": "group1",
        "study": "study1",
        "released": "2026-01-01",
        "disease": True,
        "source_sha256": source_hash,
        "split_role": "train_state",
        "genome": "mm9",
        "organism": "Mus musculus",
    }
    spec_path=tmp_path / "spec.json"
    spec_path.write_text(json.dumps(spec), encoding="utf-8")
    compiled=tmp_path / "compiled"
    compiled.mkdir()
    shard=compiled / "shard-000000.npz"
    np.savez_compressed(shard, example_id=np.asarray(["x"]))
    manifest={
        "schema": "pdac-circuit.chromatin-shards/3",
        "track": spec,
        "native_genome_validated": True,
        "sequence_length": 1024,
        "bin_size": 32,
        "windows_kept": 1,
        "shards": [
            {
                "path": shard.name,
                "examples": 1,
                "bytes": shard.stat().st_size,
                "sha256": sha256_file(shard),
            }
        ],
    }
    (compiled / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    assert verify_compiled_track(compiled, spec_path, config_path)["valid"]
    manifest["track"]["perturbation_features"]=[0.0]
    (compiled / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    migration=verify_compiled_track(compiled, spec_path, config_path)
    assert migration["valid"]
    assert migration["zero_perturbation_padding"]["from_features"] == 1
    manifest["track"]["perturbation_features"]=[1.0]
    (compiled / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    assert not verify_compiled_track(compiled, spec_path, config_path)["valid"]
    manifest["track"]["perturbation_features"]=[0.0, 0.0]
    (compiled / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    shard.write_bytes(b"corrupt")
    assert not verify_compiled_track(compiled, spec_path, config_path)["valid"]

def test_campaign_accepts_only_all_zero_legacy_intervention_padding(tmp_path):
    collection=tmp_path / "collection" / "track"
    collection.mkdir(parents=True)
    np.savez_compressed(collection / "shard.npz", value=np.zeros(1))
    manifest={
        "sequence_length": 256,
        "negative_keep_probability": 1.0,
        "track": {
            "assay_features": [0.0, 0.0],
            "state_features": [0.0, 0.0, 0.0],
            "perturbation_features": [0.0, 0.0],
        },
    }
    manifest_path=collection / "manifest.json"
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    dimensions={
        "assay_features": 2,
        "state_features": 3,
        "perturbation_features": 4,
    }
    summary=_glob_summary(
        tmp_path,
        "collection/**/*.npz",
        256,
        128,
        1.0,
        require_completion_marker=False,
        allow_local_tiling=False,
        expected_conditioning_dimensions=dimensions,
    )
    assert summary["compatible_now"]
    assert summary["zero_perturbation_padding_manifests"] == 1

    manifest["track"]["perturbation_features"]=[1.0, 0.0]
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    summary=_glob_summary(
        tmp_path,
        "collection/**/*.npz",
        256,
        128,
        1.0,
        require_completion_marker=False,
        allow_local_tiling=False,
        expected_conditioning_dimensions=dimensions,
    )
    assert not summary["compatible_now"]

def test_materialized_split_audit_allows_matched_state_loci_but_rejects_group_leakage(
    tmp_path,
):
    def write(path, ids, splits, groups, chroms, starts):
        n=len(ids)
        np.savez_compressed(
            path,
            example_id=np.asarray(ids),
            split=np.asarray(splits),
            sample_group=np.asarray(groups),
            study=np.repeat("study", n),
            genome=np.repeat("mm9", n),
            chrom=np.asarray(chroms),
            start=np.asarray(starts),
            end=np.asarray(starts) + 100,
        )

    development=tmp_path / "development.npz"
    held=tmp_path / "held.npz"
    write(
        development,
        ["train-a", "locus-a"],
        ["train", "locus_test"],
        ["group-a", "group-a"],
        ["chr1", "chr8"],
        [0, 100],
    )
    write(
        held,
        ["state-b", "joint-b"],
        ["state_test", "joint_locus_state_test"],
        ["group-b", "group-b"],
        ["chr1", "chr8"],
        [0, 100],
    )
    report=audit_compiled_splits([development, held])
    assert report["ok"]
    assert report["train_held_locus_interval_overlap"] == 0

    leaked=tmp_path / "leaked.npz"
    write(leaked, ["train-b"], ["train"], ["group-b"], ["chr2"], [0])
    report=audit_compiled_splits([development, held, leaked])
    assert not report["ok"]
    assert report["held_state_development_group_overlap"] == 1

def test_paired_shard_composer_materializes_signed_state_delta(tmp_path):
    reference=tmp_path / "reference.npz"
    treatment=tmp_path / "treatment.npz"
    common={
        "example_id": np.asarray(["x", "y"]),
        "accession": np.repeat("ACC", 2),
        "sample_group": np.repeat("donor", 2),
        "study": np.repeat("study", 2),
        "chrom": np.repeat("chr1", 2),
        "start": np.asarray([0, 100]),
        "end": np.asarray([100, 200]),
        "valid": np.ones((2, 4), dtype=np.uint8),
        "assay_features": np.zeros((2, 3), dtype=np.float16),
        "state_features": np.zeros((2, 4), dtype=np.float16),
        "perturbation_features": np.zeros((2, 2), dtype=np.float16),
        "disease_mask": np.ones(2, dtype=np.uint8),
        "pair_group": np.repeat("registered-pair-1", 2),
    }
    np.savez_compressed(
        reference,
        target=np.asarray([[1, 2, 3, 4], [2, 2, 2, 2]], dtype=np.float16),
        pair_relation=np.repeat("state_reference", 2),
        **common,
    )
    np.savez_compressed(
        treatment,
        target=np.asarray([[2, 1, 5, 4], [3, 1, 2, 4]], dtype=np.float16),
        pair_relation=np.repeat("state_treatment", 2),
        **common,
    )
    (tmp_path / "manifest.json").write_text(
        json.dumps(
            {
                "schema": "pdac-circuit.chromatin-shards/3",
                "sequence_length": 128,
                "bin_size": 32,
                "negative_keep_probability": 1.0,
            }
        ),
        encoding="utf-8",
    )
    output=tmp_path / "paired"
    report=compose_paired_shards(
        [reference],
        [treatment],
        output,
        mode="state",
        windows_per_shard=2,
        minimum_overlap_fraction=1.0,
    )
    assert report["matched_examples"] == 2
    with np.load(output / "shard-000000.npz", allow_pickle=False) as shard:
        assert np.array_equal(shard["paired_delta"][0], np.asarray([1, -1, 2, 0]))
        assert np.all(shard["pair_mask"])

def test_intervention_pair_plan_requires_one_exact_registered_control(tmp_path):
    specs=tmp_path / "specs"
    specs.mkdir()
    rows=[]
    for name, relation, perturbation in (
        ("control", "control", "Transduced with sgRosa lentivirus"),
        ("treatment", "intervention", "Transduced with sgp53 lentivirus"),
    ):
        path=specs / f"{name}.json"
        path.write_text(
            json.dumps(
                {
                    "accession": name,
                    "sample_accession": f"GSM-{name}",
                    "sample_group": "GSETEST:T6",
                    "genome": "mm10",
                    "pair_group": "GSE:test:H3K27ac:rep1:lentiviral_crispr",
                    "pair_relation": relation,
                    "pair_control_family": "lentiviral_crispr",
                    "perturbation_label": perturbation,
                }
            ),
            encoding="utf-8",
        )
        rows.append(
            {"spec": str(path.relative_to(tmp_path)), "assay": "H3K27ac"}
        )
    index=tmp_path / "index.json"
    index.write_text(
        json.dumps(
            {
                "schema": "pdac-circuit.geo-track-specs/1",
                "accession": "GSETEST",
                "written": rows,
            }
        ),
        encoding="utf-8",
    )
    plan=build_intervention_pair_plan(tmp_path, index)
    assert len(plan["pairs"]) == 1
    assert not plan["unresolved"]
    assert plan["pairs"][0]["assay"] == "H3K27ac"
    assert plan["pairs"][0]["independence_group"] == "GSETEST:T6"
    assert "No state pairs generated" in plan["state_pair_policy"]

def test_intervention_pair_plan_materializes_a_hash_bound_training_collection(
    tmp_path,
):
    specs=tmp_path / "data" / "track_specs" / "GSETEST"
    compiled=tmp_path / "data" / "compiled"
    specs.mkdir(parents=True)
    compiled.mkdir(parents=True)
    rows=[]
    pair_group="GSETEST:T6:h3k27ac:unspecified:lentiviral_crispr"
    for accession, relation, label, target in (
        ("control", "control", "Transduced with sgRosa lentivirus", [1.0, 2.0]),
        ("treatment", "intervention", "Transduced with sgp53 lentivirus", [3.0, 1.0]),
    ):
        spec=specs / f"{accession}.json"
        spec.write_text(
            json.dumps(
                {
                    "accession": accession,
                    "sample_accession": f"GSM-{accession}",
                    "sample_group": "GSETEST:T6",
                    "genome": "mm9",
                    "pair_group": pair_group,
                    "pair_relation": relation,
                    "pair_control_family": "lentiviral_crispr",
                    "perturbation_label": label,
                }
            ),
            encoding="utf-8",
        )
        rows.append(
            {"spec": str(spec.relative_to(tmp_path)), "assay": "H3K27ac"}
        )
        track=compiled / accession
        track.mkdir()
        np.savez_compressed(
            track / "shard-000000.npz",
            example_id=np.asarray([f"{accession}-a", f"{accession}-b"]),
            accession=np.repeat(accession, 2),
            sample_group=np.repeat("GSETEST:T6", 2),
            study=np.repeat("GSETEST", 2),
            genome=np.repeat("mm9", 2),
            organism=np.repeat("Mus musculus", 2),
            pair_group=np.repeat(pair_group, 2),
            pair_relation=np.repeat(relation, 2),
            split=np.repeat("train", 2),
            chrom=np.repeat("chr1", 2),
            start=np.asarray([0, 128]),
            end=np.asarray([128, 256]),
            target=np.asarray([target, target], dtype=np.float16),
            valid=np.ones((2, 2), dtype=np.uint8),
            assay_features=np.zeros((2, 3), dtype=np.float16),
            state_features=np.zeros((2, 4), dtype=np.float16),
            perturbation_features=np.zeros((2, 2), dtype=np.float16),
            disease_mask=np.ones(2, dtype=np.uint8),
        )
        (track / "manifest.json").write_text(
            json.dumps(
                {
                    "schema": "pdac-circuit.chromatin-shards/3",
                    "sequence_length": 256,
                    "bin_size": 128,
                    "negative_keep_probability": 1.0,
                }
            ),
            encoding="utf-8",
        )
    index=specs / "index.json"
    index.write_text(
        json.dumps(
            {
                "schema": "pdac-circuit.geo-track-specs/1",
                "accession": "GSETEST",
                "written": rows,
                "failures": [],
            }
        ),
        encoding="utf-8",
    )
    (compiled / "_COMPLETE.json").write_text(
        json.dumps(
            {
                "schema": "pdac-circuit.compiled-collection-completion/1",
                "successful": True,
                "track_index_sha256": sha256_file(index),
            }
        ),
        encoding="utf-8",
    )
    pair_plan_path=tmp_path / "data" / "pair-plan.json"
    plan=build_intervention_pair_plan(tmp_path, index, out=pair_plan_path)
    assert len(plan["pairs"]) == 1 and not plan["unresolved"]
    paired=tmp_path / "data" / "paired"
    report=materialize_intervention_pair_plan(
        tmp_path,
        pair_plan_path,
        compiled,
        paired,
        windows_per_shard=2,
        minimum_overlap_fraction=1.0,
    )
    assert report["registered_pairs"] == report["verified_pairs"] == 1
    assert report["examples"] == 2
    assert report["minimum_required_overlap_fraction"] == 1.0
    assert report["minimum_observed_overlap_fraction"] == 1.0
    assert report["pairs"][0]["assay"] == "H3K27ac"
    assert report["pairs"][0]["independence_group"] == "GSETEST:T6"
    assert report["pairs"][0]["composition_minimum_overlap_fraction"] == 1.0
    pair_output=paired / next(path.name for path in paired.iterdir() if path.is_dir())
    audit=verify_paired_output(pair_output, expected_pair_id=plan["pairs"][0]["pair_id"])
    assert audit["valid"]
    with np.load(pair_output / "shard-000000.npz", allow_pickle=False) as shard:
        assert np.array_equal(shard["perturbation_delta"][0], [2.0, -1.0])
    summary=_glob_summary(
        tmp_path,
        "data/paired/**/*.npz",
        256,
        128,
        1.0,
        require_completion_marker=True,
        allow_local_tiling=False,
        expected_conditioning_dimensions={
            "assay_features": 3,
            "state_features": 4,
            "perturbation_features": 2,
        },
    )
    assert summary["compatible_now"] is True
    assert summary["conditioning_lengths"] == {
        "assay_features": [3],
        "state_features": [4],
        "perturbation_features": [2],
    }
    supervision=_validate_stage_supervision(
        sorted(paired.glob("**/*.npz")),
        "signed_intervention_residual",
        project_root=tmp_path,
    )
    assert supervision["registered_pairs"] == 1
    with pytest.raises(ValueError, match="paired collection"):
        _validate_stage_supervision(
            sorted(compiled.glob("**/*.npz")),
            "signed_intervention_residual",
            project_root=tmp_path,
        )

def test_group_bootstrap_can_certify_clear_paired_win():
    rng=np.random.default_rng(7)
    groups=np.repeat([f"donor-{i}" for i in range(8)], 3)
    target=rng.normal(size=(len(groups), 16))
    candidate_prediction=target + rng.normal(scale=0.03, size=target.shape)
    baseline_prediction=target + rng.normal(scale=0.9, size=target.shape)
    common={
        "example_id": np.asarray([f"e{i}" for i in range(len(groups))]),
        "target": target,
        "group": groups,
        "split": np.repeat("joint_locus_state_test", len(groups)),
    }
    candidate=PredictionBundle(
        "candidate", prediction=candidate_prediction, lower=None, upper=None, **common
    )
    baseline=PredictionBundle(
        "Enformer", prediction=baseline_prediction, lower=None, upper=None, **common
    )
    rule=BenchmarkRule(
        axis="joint",
        metric="pearson",
        minimum_delta=0.1,
        minimum_groups=8,
    )
    result=compare_axis(
        candidate,
        baseline,
        rule,
        split="joint_locus_state_test",
        bootstrap=1000,
        seed=11,
    )
    assert result.passed
    assert result.delta > 0.1
    assert result.ci_low > 0

def test_benchmark_rejects_different_group_or_split_geometry():
    common={
        "example_id": np.asarray(["e0", "e1"]),
        "target": np.asarray([[1.0], [2.0]]),
        "prediction": np.asarray([[1.0], [2.0]]),
        "group": np.asarray(["donor-a", "donor-b"]),
        "split": np.asarray(["state_test", "state_test"]),
    }
    candidate=PredictionBundle("candidate", **common)
    different_group=PredictionBundle(
        "baseline", **{**common, "group": np.asarray(["donor-a", "donor-c"])}
    )
    with pytest.raises(ValueError, match="independent-group labels differ"):
        align_bundles(candidate, different_group)
    different_split=PredictionBundle(
        "baseline", **{**common, "split": np.asarray(["state_test", "locus_test"])}
    )
    with pytest.raises(ValueError, match="split labels differ"):
        align_bundles(candidate, different_split)
    extra_example=PredictionBundle(
        "baseline",
        example_id=np.asarray(["e0", "e1", "e2"]),
        target=np.asarray([[1.0], [2.0], [3.0]]),
        prediction=np.asarray([[1.0], [2.0], [3.0]]),
        group=np.asarray(["donor-a", "donor-b", "donor-c"]),
        split=np.asarray(["state_test", "state_test", "state_test"]),
    )
    with pytest.raises(ValueError, match="exactly the same example IDs"):
        align_bundles(candidate, extra_example)

def test_encode_assay_vector_is_frozen_and_strict():
    h3k27ac=assay_vector(
        {
            "assay": "Histone ChIP-seq",
            "target": "H3K27ac",
            "output_type": "signal p-value",
            "status": "released",
            "audit_errors": 0,
        }
    )
    assert len(h3k27ac) == 12
    assert h3k27ac[1] == 1.0
    assert h3k27ac[11] == 1.0
    with pytest.raises(ValueError, match="cannot map"):
        assay_vector({"assay": "unknown", "status": "released"})

def test_encode_track_resolution_is_parallel_resumable_and_atomic(tmp_path, monkeypatch):
    manifests=tmp_path / "data" / "manifests"
    raw=tmp_path / "data" / "raw" / "encode-bulk"
    configs=tmp_path / "configs"
    manifests.mkdir(parents=True)
    raw.mkdir(parents=True)
    configs.mkdir(parents=True)
    (configs / "encode-healthy-selection-policy.json").write_text(
        json.dumps(
            {
                "schema": "pdac-circuit.encode-healthy-selection-policy/1",
                "required_status": "released",
                "canonical_outputs": {
                    "DNase-seq": ["read-depth normalized signal"]
                },
                "supported_histone_targets": [],
            }
        ),
        encoding="utf-8",
    )
    artifacts=[]
    for accession in ("ENCFFAAAAAA", "ENCFFBBBBBB"):
        local=raw / f"{accession}.bigWig"
        local.write_bytes(b"bigwig")
        artifacts.append(
            {
                "name": local.name,
                "localPath": str(local.relative_to(tmp_path)),
                "sha256": ("a" if accession.endswith("A") else "b") * 64,
                "url": f"https://www.encodeproject.org/files/{accession}/@@download",
            }
        )
    (manifests / "encode-bulk.heavy.json").write_text(
        json.dumps({"artifacts": artifacts}), encoding="utf-8"
    )

    def resolved(accession):
        return {
            "accession": accession,
            "dataset": f"ENCSR{accession[-3:]}",
            "assay": "DNase-seq",
            "biosample": "pancreas",
            "target": "",
            "output_type": "read-depth normalized signal",
            "file_format": "bigWig",
            "biological_replicates": [1],
            "technical_replicates": ["1_1"],
            "released": "2026-01-01",
            "status": "released",
            "audit_errors": 0,
            "audit_warnings": 0,
        }

    monkeypatch.setattr("pdac_circuit.chromatin.encode.resolve_encode_accession", resolved)
    report=build_encode_track_specs(tmp_path, metadata_workers=2)
    assert len(report["written"]) == 2
    assert report["failures"] == []
    cache=json.loads(
        (tmp_path / "data" / "metadata" / "encode_tracks.json").read_text(
            encoding="utf-8"
        )
    )
    assert set(cache) == {"ENCFFAAAAAA", "ENCFFBBBBBB"}
    assert not list(tmp_path.rglob("*.partial-*"))

def test_compile_index_rejects_unresolved_track_specs(tmp_path):
    index=tmp_path / "index.json"
    index.write_text(
        json.dumps(
            {
                "schema": "pdac-circuit.encode-track-specs/1",
                "written": [],
                "failures": [{"accession": "ENCFFFAILED", "error": "unresolved"}],
            }
        ),
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="unresolved failures"):
        run_compile_index(
            config_path=str(tmp_path / "unused-config.json"),
            track_index_path=str(index),
            output_dir=str(tmp_path / "output"),
        )

def test_replicate_quality_filter_is_manifest_bound(tmp_path):
    shards=[]
    for name, quality in (("gold", 1.0), ("audited", 0.5)):
        directory=tmp_path / name
        directory.mkdir()
        shard=directory / "shard.npz"
        shard.write_bytes(b"npz")
        (directory / "manifest.json").write_text(
            json.dumps({"track": {"assay_features": [0.0] * 11 + [quality]}}),
            encoding="utf-8",
        )
        shards.append(shard)
    assert _filter_shards_by_replicate_quality(shards, 1.0) == [shards[0]]
    assert _filter_shards_by_replicate_quality(shards, None) == shards
    with pytest.raises(ValueError, match=r"\[0, 1\]"):
        _filter_shards_by_replicate_quality(shards, 1.1)

def test_geo_plan_helpers_and_protected_fetch_guard(tmp_path):
    assert geo_series_bucket("GSE99275") == "GSE99nnn"
    assert geo_series_bucket("GSE195623") == "GSE195nnn"
    assert supplementary_url("GSE195623").endswith("/GSE195nnn/GSE195623/suppl/")
    with pytest.raises(ValueError, match="not a GEO"):
        geo_series_bucket("SRP123")
    html=(
        '<a href="../">Parent</a><a href="sample%20one.tsv.gz">one</a>'
        '<a href="https://www.hhs.gov/policy/index.html">external policy</a>'
    )
    assert _supplementary_links(html, supplementary_url("GSE195623")) == [
        {
            "name": "sample one.tsv.gz",
            "url": supplementary_url("GSE195623") + "sample%20one.tsv.gz",
            "bytes": None,
        }
    ]
    plan_path=tmp_path / "protected.json"
    plan_path.write_text(
        json.dumps(
            {
                "schema": "pdac-circuit.geo-download-plan/1",
                "accession": "GSE124229",
                "protected_from_training": True,
                "known_total_bytes": 0,
                "files": [],
            }
        ),
        encoding="utf-8",
    )
    with pytest.raises(PermissionError, match="registered test/holdout"):
        fetch_geo_plan(plan_path, tmp_path / "data")
    with pytest.raises(PermissionError, match="pre-access seal"):
        fetch_geo_plan(
            plan_path,
            tmp_path / "data",
            allow_protected_study=True,
        )

def test_geo_archive_inventory_parses_assay_but_not_state(tmp_path):
    archive_path=tmp_path / "study.tar"
    payload=b"bigwig-placeholder"
    with tarfile.open(archive_path, "w") as archive:
        info=tarfile.TarInfo("GSM123_N5_ATAC_rep2.bigWig")
        info.size=len(payload)
        archive.addfile(info, io.BytesIO(payload))
    report=inspect_geo_archive(archive_path)
    assert report["member_count"] == 1
    candidate=report["track_candidates"][0]
    assert candidate["assay"].lower() == "atac"
    assert candidate["sample_label"] == "N5"
    assert candidate["replicate"] == 2
    assert candidate["state"] == "UNRESOLVED_REQUIRES_REGISTERED_METADATA"

    unsafe_path=tmp_path / "unsafe.tar"
    with tarfile.open(unsafe_path, "w") as archive:
        info=tarfile.TarInfo("../escape.bigWig")
        info.size=len(payload)
        archive.addfile(info, io.BytesIO(payload))
    with pytest.raises(ValueError, match="unsafe tar member"):
        inspect_geo_archive(unsafe_path)

def test_geo_soft_state_and_signed_intervention_are_authoritative():
    soft=(
        "^SAMPLE = GSM123\n"
        "!Sample_title = misleading_normal_filename\n"
        "!Sample_source_name_ch1 = T3/FOXA1 organoid\n"
        "!Sample_organism_ch1 = Mus musculus\n"
        "!Sample_status = Public on Jul 31 2017\n"
        "!Sample_characteristics_ch1 = cell type: Mouse PDA primary tumor organoid (T3)\n"
        "!Sample_characteristics_ch1 = peturbation: Transduced with MSCV/V5-FOXA1 retrovirus\n"
    )
    sample=parse_geo_soft(soft)["GSM123"]
    assert canonical_state(sample) == "primary_PDAC"
    sample["data_processing"]=[
        "Reads mapped to reference mouse (mm9) genome assembly; Genome_build: mm9"
    ]
    assert canonical_genome(sample, "mouse") == "mm9"
    foxa1=signed_perturbation_vector(
        sample["characteristics"]["peturbation"][0]
    )
    assert len(foxa1) == 22
    assert foxa1[12] == 1.0
    assert foxa1[21] == 1.0
    assert not any(signed_perturbation_vector("Transduced with sgRosa lentivirus"))
    assert signed_perturbation_vector(
        "Transduced with miR-E/shFoxa1.2959 retrovirus"
    )[12] == -1.0
    assert perturbation_control_family("none") == "unperturbed"
    assert (
        perturbation_control_family("Transduced with MSCV/emp retrovirus")
        == "mscv_overexpression"
    )
    assert (
        perturbation_control_family("Transduced with miR-E/shFoxa1.2959 retrovirus")
        == "mire_shrna"
    )
    assert (
        perturbation_control_family("Transduced with sgp53 lentivirus")
        == "lentiviral_crispr"
    )

    human_soft=(
        "^SAMPLE = GSM2640402\n"
        "!Sample_title = hT1 H3K27ac ChIP-Seq\n"
        "!Sample_source_name_ch1 = hT1 organoid\n"
        "!Sample_organism_ch1 = Homo sapiens\n"
        "!Sample_characteristics_ch1 = cell type: Patient-derived PDA primary tumor organoid (hT1)\n"
        "!Sample_characteristics_ch1 = strain: PDA patient\n"
        "!Sample_data_processing = Genome_build: mm9 and hg19\n"
    )
    human=parse_geo_soft(human_soft)["GSM2640402"]
    excluded=resolve_sample_metadata(
        human,
        {
            "metadata_decoder": "bojq_progression_v1",
            "sample_profile_exclusions": {
                "GSM2640402": "archived profile has an hg18 chromosome signature"
            },
        },
    )
    assert excluded["canonical_genome"] == "hg19"
    assert excluded["profile_eligible"] is False
    assert "hg18" in excluded["profile_exclusion_reason"]

def test_geo_state_never_falls_back_to_title_tokens():
    sample={
        "gsm": "GSM999",
        "title": "N5_ATAC",
        "characteristics": {"cell type": ["unregistered epithelial culture"]},
    }
    with pytest.raises(ValueError, match="explicit review required"):
        canonical_state(sample)

def test_human_driver_and_patient_validation_metadata_are_explicit():
    engineered_soft=(
        "^SAMPLE = GSM8403070\n"
        "!Sample_title = ATAC_KCP_LY_1\n"
        "!Sample_source_name_ch1 = Pancreatic (hPSC derived)\n"
        "!Sample_organism_ch1 = Homo sapiens\n"
        "!Sample_characteristics_ch1 = cell type: Pancreatic progenitor organoid\n"
        "!Sample_characteristics_ch1 = genotype: KCP\n"
        "!Sample_characteristics_ch1 = treatment: ERKi\n"
        "!Sample_data_processing = Assembly: hg19\n"
        "!Sample_supplementary_file_1 = ftp://example/GSM8403070_signal.bw\n"
    )
    engineered=parse_geo_soft(engineered_soft)["GSM8403070"]
    resolved=resolve_sample_metadata(
        engineered,
        {"metadata_decoder": "human_progenitor_organoid_progression_v1"},
    )
    assert resolved["canonical_state"] == "primary_PDAC"
    assert resolved["canonical_genome"] == "hg19"
    assert resolved["canonical_assay"] == "ATAC"
    assert resolved["state_features"][13] == 1.0
    assert resolved["perturbation_features"][0] == -1.0
    assert resolved["perturbation_features"][14] == -1.0
    assert resolved["perturbation_features"][15] == -1.0
    assert resolved["perturbation_features"][18] == 1.0
    assert resolved["perturbation_features"][21] == 1.0
    assert engineered["supplementary_files"] == [
        "ftp://example/GSM8403070_signal.bw"
    ]

    patient_soft=(
        "^SAMPLE = GSM8403129\n"
        "!Sample_title = MSKB1\n"
        "!Sample_source_name_ch1 = patient tumor\n"
        "!Sample_organism_ch1 = Homo sapiens\n"
        "!Sample_characteristics_ch1 = tissue type: Pancreatic ductal adenocarcinoma\n"
        "!Sample_characteristics_ch1 = kras allele: G12D\n"
        "!Sample_characteristics_ch1 = status: DOD\n"
        "!Sample_characteristics_ch1 = os: 573\n"
        "!Sample_characteristics_ch1 = dfs: 41\n"
        "!Sample_data_processing = Assembly: hg19\n"
    )
    patient=parse_geo_soft(patient_soft)["GSM8403129"]
    patient_resolved=resolve_sample_metadata(
        patient,
        {"metadata_decoder": "patient_pdac_validation_v1"},
    )
    assert patient_resolved["sample_group"] == "MSKB1"
    assert patient_resolved["canonical_state"] == "primary_PDAC"
    assert patient_resolved["state_features"][13] == 1.0
    assert not any(patient_resolved["perturbation_features"])

    kcp=driver_perturbation_vector("KCP")
    assert kcp[14] == kcp[15] == -1.0

def test_enformer_target_mapping_is_label_blind_and_fail_closed(tmp_path):
    metadata=tmp_path / "targets.tsv"
    metadata.write_text(
        "index\tgenome\tidentifier\tdescription\n"
        "0\t0\tENCFF_A\tDNASE:pancreas donor A\n"
        "1\t0\tENCFF_B\tDNASE:pancreas donor B\n"
        "2\t0\tENCFF_C\tDNASE:liver donor C\n",
        encoding="utf-8",
    )
    policy=tmp_path / "policy.json"
    policy.write_text(
        json.dumps(
            {
                "schema": "pdac-circuit.enformer-target-policy/1",
                "rules": [
                    {
                        "name": "pancreas_accessibility",
                        "description_regex": "^DNASE:.*pancreas",
                        "minimum_matches": 2,
                        "maximum_matches": 2,
                        "aggregation": "arithmetic_mean",
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    out=tmp_path / "map.json"
    result=resolve_enformer_target_map(metadata, policy, out)
    assert result["label_blind"] is True
    assert result["head"] == "human"
    assert result["allowed_genomes"] == ["hg38", "hg19"]
    assert result["head_index_offset"] == 0
    assert result["rules"][0]["target_indices"] == [0, 1]
    assert result["rules"][0]["source_indices"] == [0, 1]
    assert result["rules"][0]["targets"][0]["source_index"] == 0
    assert len(result["policy_sha256"]) == 64

    broken=json.loads(policy.read_text(encoding="utf-8"))
    broken["rules"][0]["minimum_matches"]=3
    policy.write_text(json.dumps(broken), encoding="utf-8")
    with pytest.raises(ValueError, match="matched 2 targets"):
        resolve_enformer_target_map(metadata, policy, out)

    mouse_metadata=tmp_path / "targets_mouse.tsv"
    mouse_metadata.write_text(
        "index\tgenome\tidentifier\tdescription\n"
        "5313\t1\tMOUSE_A\tCHIP:H3K27ac:mouse tissue A\n"
        "5314\t1\tMOUSE_B\tCHIP:H3K27ac:mouse tissue B\n",
        encoding="utf-8",
    )
    mouse_policy=tmp_path / "mouse-policy.json"
    mouse_policy.write_text(
        json.dumps(
            {
                "schema": "pdac-circuit.enformer-target-policy/1",
                "head": "mouse",
                "head_index_offset": 5313,
                "output_target_count": 1643,
                "allowed_genomes": ["mm10", "mm9"],
                "rules": [
                    {
                        "name": "mouse_H3K27ac_all_tissues",
                        "description_regex": "^CHIP:H3K27ac:",
                        "minimum_matches": 2,
                        "maximum_matches": 2,
                        "aggregation": "arithmetic_mean",
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    mouse_out=tmp_path / "mouse-map.json"
    mouse=resolve_enformer_target_map(mouse_metadata, mouse_policy, mouse_out)
    assert mouse["head"] == "mouse"
    assert mouse["head_index_offset"] == 5313
    assert mouse["rules"][0]["source_indices"] == [5313, 5314]
    assert mouse["rules"][0]["target_indices"] == [0, 1]
    assert mouse["rules"][0]["targets"][1]["source_index"] == 5314

def test_borzoi_target_mapping_freezes_transforms_and_strand_pairs(tmp_path):
    metadata=tmp_path / "targets.tsv"
    metadata.write_text(
        "\tidentifier\tfile\tclip\tclip_soft\tscale\tsum_stat\tstrand_pair\tdescription\n"
        "0\tENCFF_PLUS\ta\t768\t384\t0.3\tsum_sqrt\t1\tRNA:pancreas adult\n"
        "1\tENCFF_MINUS\tb\t768\t384\t0.3\tsum_sqrt\t0\tRNA:pancreas adult\n",
        encoding="utf-8",
    )
    policy=tmp_path / "policy.json"
    policy.write_text(
        json.dumps(
            {
                "schema": "pdac-circuit.borzoi-target-policy/1",
                "source_url": "official",
                "source_repository": "official-repository",
                "source_commit": "a" * 40,
                "head": "human",
                "allowed_genomes": ["hg38"],
                "input_bp": 524288,
                "native_bin_bp": 32,
                "native_output_bins": 16384,
                "comparison_bin_bp": 128,
                "comparison_bins": 896,
                "rules": [
                    {
                        "name": "pancreas_RNA",
                        "description_regex": "^RNA:.*pancreas",
                        "minimum_matches": 2,
                        "maximum_matches": 2,
                        "aggregation": "arithmetic_mean",
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    out=tmp_path / "map.json"
    result=resolve_borzoi_target_map(metadata, policy, out)
    assert result["label_blind"] is True
    assert result["target_count"] == 2
    assert result["strand_pair_index"] == [1, 0]
    assert result["rules"][0]["target_indices"] == [0, 1]
    assert result["rules"][0]["targets"][0]["sum_stat"] == "sum_sqrt"
    assert result["rules"][0]["targets"][0]["scale"] == pytest.approx(0.3)

def test_enformer_state_adapter_is_identity_initialized_and_small():
    import torch

    config=EnformerAdapterConfig(
        bins=32,
        assay_features=3,
        state_features=6,
        perturbation_features=2,
        channels=16,
        layers=2,
        dropout=0.0,
    )
    model=EnformerStateAdapter(config).eval()
    baseline=torch.linspace(0, 5, config.bins)[None]
    with torch.no_grad():
        prediction=model(
            baseline,
            torch.zeros(1, config.assay_features),
            torch.zeros(1, config.state_features),
            torch.zeros(1, config.perturbation_features),
        )
    assert torch.allclose(prediction, baseline, atol=1e-6, rtol=1e-6)
    assert adapter_parameter_report(config)["parameters"] < 100_000

def test_evaluation_windows_and_conditions_are_label_free(tmp_path):
    shard=tmp_path / "shard.npz"
    np.savez_compressed(
        shard,
        example_id=np.asarray(["train-1", "test-1"]),
        genome=np.repeat("hg38", 2),
        chrom=np.repeat("chr1", 2),
        start=np.asarray([0, 1024]),
        end=np.asarray([1024, 2048]),
        split=np.asarray(["train", "joint_locus_state_test"]),
        sample_group=np.asarray(["donor-train", "donor-test"]),
        assay_features=np.zeros((2, 12), dtype=np.float16),
        state_features=np.zeros((2, 18), dtype=np.float16),
        perturbation_features=np.zeros((2, 22), dtype=np.float16),
        target=np.ones((2, 8), dtype=np.float16),
    )
    mouse_shard=tmp_path / "mouse-shard.npz"
    np.savez_compressed(
        mouse_shard,
        example_id=np.asarray(["mouse-test"]),
        genome=np.asarray(["mm9"]),
        chrom=np.asarray(["chr1"]),
        start=np.asarray([2048]),
        end=np.asarray([3072]),
        split=np.asarray(["joint_locus_state_test"]),
        sample_group=np.asarray(["mouse-donor"]),
        assay_features=np.zeros((1, 12), dtype=np.float16),
        state_features=np.zeros((1, 18), dtype=np.float16),
        perturbation_features=np.zeros((1, 22), dtype=np.float16),
        target=np.ones((1, 8), dtype=np.float16),
    )
    windows=tmp_path / "windows.json"
    conditions=tmp_path / "conditions.npz"
    report=freeze_evaluation_windows(
        [shard, mouse_shard],
        windows,
        conditions,
        split="joint_locus_state_test",
        output_bins=8,
        genomes={"hg38"},
    )
    payload=json.loads(windows.read_text(encoding="utf-8"))
    assert report["examples"] == 1
    assert payload["examples"] == [
        {
            "chrom": "chr1",
            "end": 2048,
            "example_id": "test-1",
            "genome": "hg38",
            "source_end": 2048,
            "source_start": 1024,
            "start": 1024,
        }
    ]
    assert "target" not in windows.read_text(encoding="utf-8")
    with np.load(conditions, allow_pickle=False) as archive:
        assert "target" not in archive.files
        assert archive["example_id"].tolist() == ["test-1"]
        metadata=json.loads(str(archive["metadata"].item()))
    assert metadata["contains_targets"] is False
    assert _load_label_free_example_ids(windows) == frozenset({"test-1"})
    truth_like=tmp_path / "truth-like.npz"
    np.savez_compressed(
        truth_like,
        example_id=np.asarray(["test-1"]),
        metadata=np.asarray(
            json.dumps({"schema": "pdac-circuit.profile-truth/1"}, sort_keys=True)
        ),
    )
    with pytest.raises(ValueError, match="label-free conditions"):
        _load_label_free_example_ids(truth_like)

def test_evaluation_window_sampling_is_target_blind_and_condition_stratified(tmp_path):
    shard=tmp_path / "sampling-shard.npz"
    perturbation=np.zeros((6, 2), dtype=np.float16)
    perturbation[3:, 0]=1.0
    np.savez_compressed(
        shard,
        example_id=np.asarray([f"condition-example-{index}" for index in range(6)]),
        genome=np.repeat("mm9", 6),
        chrom=np.repeat("chr1", 6),
        start=np.arange(6) * 1024,
        end=np.arange(1, 7) * 1024,
        split=np.repeat("train", 6),
        sample_group=np.repeat("GSE99311:T3", 6),
        assay_features=np.zeros((6, 2), dtype=np.float16),
        state_features=np.zeros((6, 3), dtype=np.float16),
        perturbation_features=perturbation,
        target=np.arange(48, dtype=np.float16).reshape(6, 8),
    )
    selections=[]
    for suffix in ("a", "b"):
        windows=tmp_path / f"sampled-{suffix}.json"
        conditions=tmp_path / f"sampled-{suffix}.npz"
        report=freeze_evaluation_windows(
            [shard],
            windows,
            conditions,
            split="train",
            output_bins=8,
            max_examples_per_condition_group=1,
            sampling_seed=17,
        )
        assert report["examples"] == 2
        assert report["label_free_sampling"]["strata"] == 2
        assert report["label_free_sampling"]["signal_access"] is False
        payload=json.loads(windows.read_text(encoding="utf-8"))
        selections.append([row["example_id"] for row in payload["examples"]])
        assert "target" not in windows.read_text(encoding="utf-8")
    assert selections[0] == selections[1]

def test_model_native_contexts_share_exact_label_free_cohort(tmp_path):
    shard=tmp_path / "shard.npz"
    np.savez_compressed(
        shard,
        example_id=np.asarray(["edge", "middle"]),
        genome=np.repeat("hg38", 2),
        chrom=np.repeat("chr1", 2),
        start=np.asarray([0, 4096]),
        end=np.asarray([1024, 5120]),
        split=np.repeat("state_test", 2),
        sample_group=np.asarray(["g0", "g1"]),
        assay_features=np.zeros((2, 3), dtype=np.float16),
        state_features=np.zeros((2, 6), dtype=np.float16),
        perturbation_features=np.zeros((2, 2), dtype=np.float16),
        target=np.ones((2, 16), dtype=np.float16),
        valid=np.ones((2, 16), dtype=np.uint8),
    )
    borzoi_windows=tmp_path / "borzoi.json"
    borzoi_conditions=tmp_path / "borzoi-conditions.npz"
    report=freeze_evaluation_windows(
        [shard],
        borzoi_windows,
        borzoi_conditions,
        split="state_test",
        output_bins=8,
        context_length=4096,
        chrom_sizes={"hg38": {"chr1": 10_000}},
    )
    assert report["dropped_edge_windows"] == 1
    assert report["examples"] == 1
    expanded=json.loads(borzoi_windows.read_text(encoding="utf-8"))
    assert expanded["sequence_length"] == 4096
    assert expanded["examples"][0]["example_id"] == "middle"
    assert (expanded["examples"][0]["start"], expanded["examples"][0]["end"]) == (
        2560,
        6656,
    )

    enformer_windows=tmp_path / "enformer.json"
    enformer_conditions=tmp_path / "enformer-conditions.npz"
    native=freeze_evaluation_windows(
        [shard],
        enformer_windows,
        enformer_conditions,
        split="state_test",
        output_bins=8,
        context_length=1024,
        chrom_sizes={"hg38": {"chr1": 10_000}},
        example_ids={"middle"},
    )
    assert native["examples"] == 1
    unexpanded=json.loads(enformer_windows.read_text(encoding="utf-8"))
    assert (unexpanded["examples"][0]["start"], unexpanded["examples"][0]["end"]) == (
        4096,
        5120,
    )
    truth=tmp_path / "truth.npz"
    truth_report=freeze_profile_truth(
        [shard], truth, split="state_test", crop_bins=8, example_ids={"middle"}
    )
    assert truth_report["examples"] == 1
    with np.load(truth, allow_pickle=False) as archive:
        assert archive["example_id"].tolist() == ["middle"]

def test_raw_baseline_merge_requires_one_frozen_identity(tmp_path):
    inputs=[]
    for name, example_id, target_rule in (
        ("a", "example-a", "pancreas_accessibility"),
        ("b", "example-b", "pancreas_H3K27ac"),
    ):
        path=tmp_path / f"{name}.npz"
        save_raw_predictions(
            path,
            model="Enformer",
            example_id=np.asarray([example_id]),
            prediction=np.ones((1, 8), dtype=np.float32),
            metadata={
                "model_version": "official",
                "weights_sha256": "a" * 64,
                "track_mapping_sha256": "b" * 64,
                "target_rule": target_rule,
                "example_id_sha256": hashlib.sha256(
                    f"{example_id}\n".encode("utf-8")
                ).hexdigest(),
            },
        )
        inputs.append(path)
    destination=tmp_path / "merged.npz"
    report=merge_raw_predictions(inputs, destination, command="merge")
    assert report["examples"] == 2
    assert report["target_rules"] == [
        "pancreas_H3K27ac",
        "pancreas_accessibility",
    ]
    with np.load(destination, allow_pickle=False) as archive:
        assert archive["example_id"].tolist() == ["example-a", "example-b"]

def test_enformer_adapter_training_is_group_disjoint_and_inference_is_label_free(tmp_path):
    config_path=tmp_path / "adapter.json"
    config_path.write_text(
        json.dumps(
            {
                "schema": "pdac-circuit.enformer-state-adapter/1",
                "model": {
                    "bins": 8,
                    "assay_features": 2,
                    "state_features": 3,
                    "perturbation_features": 2,
                    "channels": 8,
                    "layers": 1,
                    "kernel_size": 3,
                    "dilation_cycle": [1],
                    "dropout": 0.0,
                },
            }
        ),
        encoding="utf-8",
    )

    def surface(prefix, split, groups):
        n=len(groups)
        ids=np.asarray([f"{prefix}-{index}" for index in range(n)])
        cohort_hash=hashlib.sha256(
            "".join(f"{value}\n" for value in sorted(ids.astype(str))).encode(
                "utf-8"
            )
        ).hexdigest()
        base=np.full((n, 8), 0.5, dtype=np.float32)
        target=base + np.arange(n, dtype=np.float32)[:, None] * 0.05
        raw=tmp_path / f"{prefix}-raw.npz"
        np.savez_compressed(
            raw,
            model=np.asarray("Enformer"),
            example_id=ids,
            prediction=base,
            metadata=np.asarray(
                json.dumps(
                    {
                        "weights_sha256": "a" * 64,
                        "track_mapping_sha256": "b" * 64,
                        "example_id_sha256": cohort_hash,
                    }
                )
            ),
        )
        conditions=tmp_path / f"{prefix}-conditions.npz"
        np.savez_compressed(
            conditions,
            example_id=ids,
            group=np.asarray(groups),
            split=np.repeat(split, n),
            assay_features=np.zeros((n, 2), dtype=np.float32),
            state_features=np.zeros((n, 3), dtype=np.float32),
            perturbation_features=np.zeros((n, 2), dtype=np.float32),
            metadata=np.asarray(
                json.dumps(
                    {
                        "schema": "pdac-circuit.baseline-conditions/1",
                        "contains_targets": False,
                        "candidate_feature_access": False,
                        "example_id_sha256": cohort_hash,
                    }
                )
            ),
        )
        truth=tmp_path / f"{prefix}-truth.npz"
        np.savez_compressed(
            truth,
            example_id=ids,
            target=target,
            mask=np.ones_like(target, dtype=np.uint8),
            group=np.asarray(groups),
            split=np.repeat(split, n),
        )
        return raw, conditions, truth

    train_raw, train_conditions, train_truth=surface(
        "train", "train", ["train-a", "train-a", "train-b", "train-b"]
    )
    val_raw, val_conditions, val_truth=surface(
        "validation", "validation", ["val-a", "val-b"]
    )
    checkpoint=tmp_path / "adapter.pt"
    report=train_enformer_adapter(
        config_path,
        train_raw=train_raw,
        train_truth=train_truth,
        train_conditions=train_conditions,
        validation_raw=val_raw,
        validation_truth=val_truth,
        validation_conditions=val_conditions,
        out=checkpoint,
        epochs=2,
        batch_size=2,
        device="cpu",
    )
    assert report["train_groups"] == 2
    assert report["validation_groups"] == 2
    assert report["validation_scope"] == "group_disjoint"
    assert report["overlapping_validation_groups"] == 0

    locus_config_path=tmp_path / "adapter-locus-disjoint.json"
    locus_config=json.loads(config_path.read_text(encoding="utf-8"))
    locus_config["training_policy"]={
        "validation_scope": "locus_disjoint_same_groups_allowed"
    }
    locus_config_path.write_text(json.dumps(locus_config), encoding="utf-8")
    overlap_raw, overlap_conditions, overlap_truth=surface(
        "validation-overlap", "validation", ["train-a", "train-b"]
    )
    locus_report=train_enformer_adapter(
        locus_config_path,
        train_raw=train_raw,
        train_truth=train_truth,
        train_conditions=train_conditions,
        validation_raw=overlap_raw,
        validation_truth=overlap_truth,
        validation_conditions=overlap_conditions,
        out=tmp_path / "adapter-locus-disjoint.pt",
        epochs=1,
        batch_size=2,
        device="cpu",
    )
    assert locus_report["validation_scope"] == "locus_disjoint_same_groups_allowed"
    assert locus_report["overlapping_validation_groups"] == 2
    adapted=tmp_path / "adapted.npz"
    prediction_report=predict_enformer_adapter(
        config_path,
        checkpoint_path=checkpoint,
        raw_path=val_raw,
        conditions_path=val_conditions,
        out=adapted,
        batch_size=1,
        device="cpu",
    )
    assert prediction_report["test_label_access"] is False
    assert prediction_report["ablate_intervention_residual"] is False
    with np.load(adapted, allow_pickle=False) as archive:
        assert "target" not in archive.files
        assert archive["example_id"].tolist() == ["validation-0", "validation-1"]

    ablated=tmp_path / "adapted-zero-perturbation.npz"
    ablated_report=predict_enformer_adapter(
        config_path,
        checkpoint_path=checkpoint,
        raw_path=val_raw,
        conditions_path=val_conditions,
        out=ablated,
        batch_size=1,
        device="cpu",
        ablate_intervention_residual=True,
    )
    assert ablated_report["ablate_intervention_residual"] is True
    contrast=tmp_path / "adapter-perturbation-delta.npz"
    contrast_raw_predictions(
        ablated,
        adapted,
        contrast,
        mode="perturbation",
        command="test-adapter-contrast",
    )
    with np.load(contrast, allow_pickle=False) as archive:
        metadata=json.loads(str(archive["metadata"].item()))
        assert metadata["component"] == "perturbation_delta"

def test_state_invariant_raw_and_exact_id_assembly(tmp_path):
    truth_path=tmp_path / "truth.npz"
    np.savez_compressed(
        truth_path,
        example_id=np.asarray(["b", "a"]),
        target=np.asarray([[1.0, -1.0], [0.5, -0.5]]),
        mask=np.asarray([[1, 0], [1, 1]], dtype=np.uint8),
        group=np.asarray(["donor-b", "donor-a"]),
        split=np.repeat("state_test", 2),
    )
    raw_path=tmp_path / "raw.npz"
    make_state_invariant_raw(
        truth_path,
        raw_path,
        model="Enformer",
        model_version="official-test",
        weights_sha256="a" * 64,
        track_mapping_sha256="b" * 64,
        command="enformer zero contrast",
    )
    bundle_path=tmp_path / "bundle.npz"
    provenance_path=tmp_path / "bundle.provenance.json"
    report=assemble_prediction_bundle(
        raw_path,
        truth_path,
        bundle_path,
        provenance_path,
        training_use="predictions_only",
    )
    bundle=load_prediction_bundle(bundle_path)
    assert report["examples"] == 2
    assert np.count_nonzero(bundle.prediction) == 0
    assert bundle.mask is not None
    assert int(bundle.mask.sum()) == 3
    assert provenance_path.exists()

    with np.load(raw_path, allow_pickle=False) as archive:
        payload={key: archive[key].copy() for key in archive.files}
    payload["example_id"]=np.asarray(["b", "missing"])
    np.savez_compressed(raw_path, **payload)
    with pytest.raises(ValueError, match="selective omission is forbidden"):
        assemble_prediction_bundle(
            raw_path,
            truth_path,
            bundle_path,
            provenance_path,
            training_use="predictions_only",
        )

def test_group_block_conformal_uses_independent_group_scores(tmp_path):
    metadata={
        "model": "PDACircuitFormer",
        "model_version": "checkpoint-test",
        "weights_sha256": "a" * 64,
        "track_mapping_sha256": "b" * 64,
        "component": "mean",
        "command": "predict calibration",
    }
    calibration_truth=tmp_path / "calibration_truth.npz"
    ids=np.asarray([f"cal-{index}" for index in range(6)])
    groups=np.repeat(["donor-a", "donor-b", "donor-c"], 2)
    target=np.ones((6, 2))
    errors=np.repeat(np.asarray([0.1, 0.2, 0.3]), 2)[:, None]
    np.savez_compressed(
        calibration_truth,
        example_id=ids,
        target=target,
        group=groups,
        split=np.repeat("calibration", 6),
    )
    calibration_raw=tmp_path / "calibration_raw.npz"
    save_raw_predictions(
        calibration_raw,
        model="PDACircuitFormer",
        example_id=ids,
        prediction=target + errors,
        metadata=metadata,
    )
    target_raw=tmp_path / "target_raw.npz"
    save_raw_predictions(
        target_raw,
        model="PDACircuitFormer",
        example_id=np.asarray(["test-a", "test-b"]),
        prediction=np.ones((2, 2)),
        metadata=metadata,
    )
    out=tmp_path / "conformal.npz"
    report=conformalize_raw_predictions(
        calibration_raw,
        calibration_truth,
        target_raw,
        out,
        nominal=0.9,
        command="conformalize",
    )
    assert report["independent_groups"] == 3
    assert report["radius"] == pytest.approx(0.3)
    with np.load(out, allow_pickle=False) as archive:
        assert np.allclose(archive["lower"], 0.7)
        assert np.allclose(archive["upper"], 1.3)

def test_multi_axis_benchmark_accepts_distinct_target_spaces(tmp_path):
    candidate_root=tmp_path / "candidate"
    baseline_root=tmp_path / "baseline"
    candidate_root.mkdir()
    baseline_root.mkdir()
    groups=np.repeat(["donor-a", "donor-b", "donor-c"], 4)
    example_id=np.asarray([f"example-{i}" for i in range(len(groups))])

    profile_target=np.tile(np.asarray([0.0, 1.0, 3.0, 2.0]), (len(groups), 1))
    binary_target=np.tile(np.asarray([0.0, 1.0, 0.0, 1.0]), 3)
    bundles={
        "profile_axis": (
            profile_target,
            profile_target + 0.001,
            profile_target[:, ::-1],
            "joint_locus_state_test",
        ),
        "binary_axis": (
            binary_target,
            np.tile(np.asarray([0.01, 0.99, 0.02, 0.98]), 3),
            np.tile(np.asarray([0.99, 0.01, 0.98, 0.02]), 3),
            "external_study_test",
        ),
    }
    for axis, (target, candidate_prediction, baseline_prediction, split_name) in bundles.items():
        common={
            "example_id": example_id,
            "target": target,
            "group": groups,
            "split": np.repeat(split_name, len(groups)),
        }
        interval=np.full_like(target, 0.01, dtype=float)
        candidate=PredictionBundle(
            "PDACircuitFormer",
            prediction=candidate_prediction,
            lower=target - interval,
            upper=target + interval,
            **common,
        )
        baseline=PredictionBundle(
            "Enformer",
            prediction=baseline_prediction,
            lower=None,
            upper=None,
            **common,
        )
        candidate_path=candidate_root / f"{axis}.npz"
        baseline_path=baseline_root / f"{axis}.npz"
        save_prediction_bundle(candidate, candidate_path)
        save_prediction_bundle(baseline, baseline_path)
        for root, path, model, training_use in (
            (candidate_root, candidate_path, "PDACircuitFormer", "candidate_model"),
            (baseline_root, baseline_path, "Enformer", "predictions_only"),
        ):
            write_prediction_manifest(
                root / f"{axis}.provenance.json",
                model=model,
                model_version="frozen-test",
                prediction_bundle_path=path,
                weights_sha256="a" * 64,
                track_mapping_sha256="b" * 64,
                data_snapshot_sha256="c" * 64,
                command=f"predict {axis}",
                training_use=training_use,
            )

    registry={
        "rules": [
            {
                "axis": "profile_axis",
                "split": "joint_locus_state_test",
                "metric": "pearson",
                "minimum_delta": 0.1,
                "minimum_groups": 3,
            },
            {
                "axis": "binary_axis",
                "split": "external_study_test",
                "metric": "average_precision",
                "minimum_delta": 0.1,
                "minimum_groups": 3,
            },
        ],
        "split_policy": {"primary_surface": "joint_locus_state_test"},
        "calibration": {
            "axis": "profile_axis",
            "nominal": 1.0,
            "minimum_groups": 3,
            "max_width_iqr_multiplier": 4.0,
        },
    }
    registry_path=tmp_path / "registry.json"
    registry_path.write_text(json.dumps(registry), encoding="utf-8")
    report_path=tmp_path / "report.json"
    assert (
        run_benchmark(
            candidate_root=str(candidate_root),
            baseline_root=str(baseline_root),
            registry_path=str(registry_path),
            out=str(report_path),
            bootstrap=500,
        )
        == 0
    )
    report=json.loads(report_path.read_text(encoding="utf-8"))
    assert report["verdict"] == "BEATS_BASELINE"
    assert {axis["axis"] for axis in report["axes"]} == {"profile_axis", "binary_axis"}

    registry["comparison_model_policy"]={
        "headline_enformer": {
            "candidate_model": "PDACircuitFormer",
            "baseline_model": "Enformer",
            "supports_requested_headline_claim": True,
        }
    }
    registry_path.write_text(json.dumps(registry), encoding="utf-8")
    assert (
        run_benchmark(
            candidate_root=str(candidate_root),
            baseline_root=str(baseline_root),
            registry_path=str(registry_path),
            out=str(tmp_path / "role-bound-report.json"),
            bootstrap=500,
        )
        == 0
    )
    registry["comparison_model_policy"]["headline_enformer"][
        "baseline_model"
    ]="Weak substitute"
    registry_path.write_text(json.dumps(registry), encoding="utf-8")
    with pytest.raises(ValueError, match="requires.*Weak substitute"):
        run_benchmark(
            candidate_root=str(candidate_root),
            baseline_root=str(baseline_root),
            registry_path=str(registry_path),
            out=str(tmp_path / "wrong-role-report.json"),
            bootstrap=100,
        )
    registry["comparison_model_policy"]["headline_enformer"][
        "baseline_model"
    ]="Enformer"

    registry["candidate_seed_policy"]={
        "registered_seeds": [11, 22, 33],
        "aggregation": "arithmetic_mean",
        "individual_axis_policy": "every registered seed must have strictly positive delta",
    }
    registry_path.write_text(json.dumps(registry), encoding="utf-8")
    gated_report_path=tmp_path / "gated-report.json"
    assert (
        run_benchmark(
            candidate_root=str(candidate_root),
            baseline_root=str(baseline_root),
            registry_path=str(registry_path),
            out=str(gated_report_path),
            bootstrap=500,
        )
        == 2
    )
    gated=json.loads(gated_report_path.read_text(encoding="utf-8"))
    assert gated["verdict"] == "ABSTAIN"
    assert gated["seed_robustness"]["ok"] is False
    assert "require 3" in gated["seed_robustness"]["reason"]

def test_multi_seed_benchmark_rejects_one_regressive_seed(tmp_path):
    candidate_root=tmp_path / "ensemble"
    baseline_root=tmp_path / "baseline"
    candidate_root.mkdir()
    baseline_root.mkdir()
    seeds=[11, 22, 33]
    seed_roots=[]
    groups=np.repeat(["donor-a", "donor-b", "donor-c"], 4)
    example_id=np.asarray([f"example-{i}" for i in range(len(groups))])
    target=np.tile(np.asarray([0.0, 1.0, 3.0, 2.0]), (len(groups), 1))
    candidate_prediction=target + 0.001
    baseline_prediction=target[:, ::-1]
    common={
        "example_id": example_id,
        "target": target,
        "group": groups,
        "split": np.repeat("joint_locus_state_test", len(groups)),
    }
    ensemble_bundle=PredictionBundle(
        "PDACircuitFormer",
        prediction=candidate_prediction,
        lower=target - 0.01,
        upper=target + 0.01,
        **common,
    )
    baseline_bundle=PredictionBundle(
        "Enformer", prediction=baseline_prediction, **common
    )
    ensemble_path=candidate_root / "profile_axis.npz"
    baseline_path=baseline_root / "profile_axis.npz"
    save_prediction_bundle(ensemble_bundle, ensemble_path)
    save_prediction_bundle(baseline_bundle, baseline_path)
    components=[
        {
            "seed": seed,
            "raw_sha256": str(index) * 64,
            "weights_sha256": chr(ord("a") + index - 1) * 64,
        }
        for index, seed in enumerate(seeds, start=1)
    ]
    seed_ensemble={
        "schema": "pdac-circuit.seed-ensemble/1",
        "registered_seeds": seeds,
        "aggregation": "arithmetic_mean",
        "exact_example_ids": True,
        "components": components,
    }
    write_prediction_manifest(
        candidate_root / "profile_axis.provenance.json",
        model="PDACircuitFormer",
        model_version="ensemble-test",
        prediction_bundle_path=ensemble_path,
        weights_sha256="d" * 64,
        track_mapping_sha256="e" * 64,
        data_snapshot_sha256="f" * 64,
        command="ensemble",
        training_use="candidate_model",
        seed_ensemble=seed_ensemble,
        raw_prediction_sha256="0" * 64,
    )
    write_prediction_manifest(
        baseline_root / "profile_axis.provenance.json",
        model="Enformer",
        model_version="baseline-test",
        prediction_bundle_path=baseline_path,
        weights_sha256="a" * 64,
        track_mapping_sha256="e" * 64,
        data_snapshot_sha256="f" * 64,
        command="baseline",
        training_use="predictions_only",
    )
    for component in components:
        root=tmp_path / f"seed-{component['seed']}"
        root.mkdir()
        path=root / "profile_axis.npz"
        save_prediction_bundle(
            PredictionBundle(
                "PDACircuitFormer", prediction=candidate_prediction, **common
            ),
            path,
        )
        write_prediction_manifest(
            root / "profile_axis.provenance.json",
            model="PDACircuitFormer",
            model_version=f"seed-{component['seed']}",
            prediction_bundle_path=path,
            weights_sha256=component["weights_sha256"],
            track_mapping_sha256="e" * 64,
            data_snapshot_sha256="f" * 64,
            command=f"predict --seed {component['seed']}",
            training_use="candidate_model",
            seed=component["seed"],
            raw_prediction_sha256=component["raw_sha256"],
        )
        seed_roots.append(str(root))
    registry={
        "rules": [
            {
                "axis": "profile_axis",
                "split": "joint_locus_state_test",
                "metric": "pearson",
                "minimum_delta": 0.1,
                "minimum_groups": 3,
            }
        ],
        "split_policy": {"primary_surface": "joint_locus_state_test"},
        "calibration": {
            "axis": "profile_axis",
            "nominal": 1.0,
            "minimum_groups": 3,
            "max_width_iqr_multiplier": 4.0,
        },
        "candidate_seed_policy": {
            "registered_seeds": seeds,
            "aggregation": "arithmetic_mean",
            "individual_axis_policy": "every seed strictly positive",
        },
    }
    registry_path=tmp_path / "registry.json"
    registry_path.write_text(json.dumps(registry), encoding="utf-8")
    report_path=tmp_path / "report.json"
    assert (
        run_benchmark(
            candidate_root=str(candidate_root),
            baseline_root=str(baseline_root),
            registry_path=str(registry_path),
            out=str(report_path),
            bootstrap=500,
            candidate_seed_roots=seed_roots,
        )
        == 0
    )

    regressive_root=Path(seed_roots[-1])
    regressive_path=regressive_root / "profile_axis.npz"
    save_prediction_bundle(
        PredictionBundle(
            "PDACircuitFormer", prediction=baseline_prediction, **common
        ),
        regressive_path,
    )
    component=components[-1]
    write_prediction_manifest(
        regressive_root / "profile_axis.provenance.json",
        model="PDACircuitFormer",
        model_version=f"seed-{component['seed']}",
        prediction_bundle_path=regressive_path,
        weights_sha256=component["weights_sha256"],
        track_mapping_sha256="e" * 64,
        data_snapshot_sha256="f" * 64,
        command=f"predict --seed {component['seed']}",
        training_use="candidate_model",
        seed=component["seed"],
        raw_prediction_sha256=component["raw_sha256"],
    )
    assert (
        run_benchmark(
            candidate_root=str(candidate_root),
            baseline_root=str(baseline_root),
            registry_path=str(registry_path),
            out=str(tmp_path / "regressive-report.json"),
            bootstrap=500,
            candidate_seed_roots=seed_roots,
        )
        == 2
    )

def test_chromatin_registries_are_valid_json():
    from pdac_circuit.core.paths import ROOT

    chromatin=json.loads((ROOT / "chromatin_registry.json").read_text(encoding="utf-8"))
    assets=json.loads((ROOT / "pdac_chromatin_assets.json").read_text(encoding="utf-8"))
    enformer_assets=json.loads(
        (ROOT / "baseline_assets" / "enformer-model.json").read_text(encoding="utf-8")
    )
    mouse_policy=json.loads(
        (ROOT / "configs" / "enformer_mouse_target_policy.json").read_text(encoding="utf-8")
    )
    mouse_map=json.loads(
        (ROOT / "data" / "metadata" / "enformer_target_map_mouse.json").read_text(
            encoding="utf-8"
        )
    )
    assert chromatin["schema"] == "pdac-circuit.chromatin-registry/2"
    assert len(chromatin["conditioning"]["state_features"]) == 18
    assert len(chromatin["conditioning"]["perturbation_features"]) == 22
    assert len(chromatin["rules"]) >= 6
    assert any(asset["id"] == "GSE124229" for asset in assets["assets"])
    cohort=validate_human_cohort_contract(ROOT)
    assert cohort["validation_studies"] == ["GSE272463"]
    assert cohort["protected_test_studies"] == ["GSE124229", "GSE124230"]
    assert "GSE272463" not in cohort["training_studies"]
    assert enformer_assets["schema"] == "pdac-circuit.enformer-assets/1"
    assert enformer_assets["model_url"] == "https://tfhub.dev/deepmind/enformer/1"
    assert enformer_assets["training_use"] == "predictions_only"
    assert mouse_policy["head"] == mouse_map["head"] == "mouse"
    assert mouse_policy["head_index_offset"] == mouse_map["head_index_offset"] == 5313
    assert mouse_map["output_target_count"] == 1643
    mouse_h3k27ac=next(
        row
        for row in mouse_map["rules"]
        if row["name"] == "mouse_H3K27ac_all_tissues"
    )
    assert len(mouse_h3k27ac["target_indices"]) == 107
    assert min(mouse_h3k27ac["source_indices"]) >= 5313
    assert max(mouse_h3k27ac["target_indices"]) < 1643
    enformer_runner=(
        ROOT / "baseline_runners" / "enformer_export.py"
    ).read_text(encoding="utf-8")
    assert "--model-url" not in enformer_runner
    assert "verify_materialized_enformer_assets" in enformer_runner
    assert 'head not in {"human", "mouse"}' in enformer_runner
    assert '"mm9": {"chr1": 197_195_432' in enformer_runner

def test_human_patient_track_index_fails_closed_if_any_profile_can_train(tmp_path):
    from pdac_circuit.core.paths import ROOT

    (tmp_path / "configs").mkdir()
    for relative in (
        "chromatin_registry.json",
        "pdac_chromatin_assets.json",
        "configs/chromatin-campaign.json",
        "configs/chromatin-human-cohort.json",
    ):
        source=ROOT / relative
        destination=tmp_path / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source, destination)

    spec_path=tmp_path / "data" / "track_specs" / "GSE272463" / "bad.json"
    spec_path.parent.mkdir(parents=True)
    spec_path.write_text(
        json.dumps({"study": "GSE272463", "split_role": "train_state"}),
        encoding="utf-8",
    )
    (spec_path.parent / "index.json").write_text(
        json.dumps(
            {
                "schema": "pdac-circuit.geo-track-specs/1",
                "accession": "GSE272463",
                "failures": [],
                "written": [
                    {
                        "spec": str(spec_path.relative_to(tmp_path)).replace("\\", "/"),
                        "split_role": "train_state",
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="can contribute gradients"):
        validate_human_cohort_contract(tmp_path)

def test_protected_human_metadata_requires_final_three_seed_release(tmp_path):
    from pdac_circuit.core.paths import ROOT

    seal=ROOT / "results" / "frozen" / "protected-studies.seal.json"
    report=validate_protected_study_seal(ROOT, seal, accession="GSE124229")
    assert report["ok"] is True
    assert report["seal"]["protected_metadata_present_at_seal"] is False
    missing_release=tmp_path / "missing-release.json"
    with pytest.raises(PermissionError, match="missing protected metadata release"):
        fetch_geo_soft_metadata(
            ROOT,
            "GSE124229",
            allow_protected_metadata=True,
            protected_release_path=missing_release,
        )
    with pytest.raises(ValueError, match="require 3"):
        authorize_protected_metadata_release(
            ROOT,
            seal_path=seal,
            checkpoint_paths=[],
            out=tmp_path / "release.json",
        )

def test_protected_external_target_download_requires_same_post_training_release(tmp_path):
    from pdac_circuit.core.paths import ROOT

    seal=ROOT / "results" / "frozen" / "protected-studies.seal.json"
    plan=ROOT / "data" / "manifests" / "studies" / "GSE301272.plan.json"
    with pytest.raises(PermissionError, match="post-training release manifest"):
        fetch_geo_plan(
            plan,
            tmp_path / "data",
            allow_protected_study=True,
            protected_seal_path=seal,
            project_root=ROOT,
        )

def test_circuit_interpretation_gate_is_rotation_invariant_and_rejects_collapse(
    tmp_path,
):
    gate={
        "circuit_interpretation_gate": {
            "minimum_seeds": 3,
            "minimum_pairwise_linear_cka": 0.4,
            "minimum_median_pairwise_linear_cka": 0.6,
            "minimum_effective_rank": 4.0,
            "maximum_single_factor_variance_fraction": 0.75,
        }
    }
    rng=np.random.default_rng(20260715)
    example_id=np.asarray([f"example-{index:03d}" for index in range(96)])
    latent=rng.normal(size=(len(example_id), 8)).astype(np.float32)
    stable_paths=[]
    for seed in (11, 22, 33):
        rotation, _=np.linalg.qr(rng.normal(size=(8, 8)))
        factors=(latent @ rotation).astype(np.float32)
        ids=example_id
        if seed == 22:
            ids=ids[::-1]
            factors=factors[::-1]
        path=tmp_path / f"stable-{seed}.npz"
        save_raw_predictions(
            path,
            model="PDACircuitFormer",
            example_id=ids,
            prediction=factors,
            metadata={
                "schema": "pdac-circuit.raw-predictions/1",
                "component": "circuit_factors",
                "seed": seed,
            },
        )
        stable_paths.append(path)
    report=audit_circuit_stability(stable_paths, gate)
    assert report["interpretation_status"] == "PASS"
    assert report["coordinate_identifiability_claimed"] is False
    assert report["summary"]["minimum_pairwise_linear_cka"] == pytest.approx(1.0)
    out=tmp_path / "audit.json"
    write_circuit_stability_audit(out, report)
    assert json.loads(out.read_text(encoding="utf-8"))["interpretation_status"] == "PASS"

    collapsed_paths=[]
    one_factor=rng.normal(size=(len(example_id), 1)).astype(np.float32)
    for seed in (44, 55, 66):
        collapsed=np.repeat(one_factor, 8, axis=1)
        path=tmp_path / f"collapsed-{seed}.npz"
        save_raw_predictions(
            path,
            model="PDACircuitFormer",
            example_id=example_id,
            prediction=collapsed,
            metadata={
                "schema": "pdac-circuit.raw-predictions/1",
                "component": "circuit_factors",
                "seed": seed,
            },
        )
        collapsed_paths.append(path)
    collapsed_report=audit_circuit_stability(collapsed_paths, gate)
    assert collapsed_report["interpretation_status"] == "ABSTAIN"
    assert collapsed_report["checks"]["minimum_effective_rank"] is False

def test_claim_surface_contract_separates_within_study_and_external_perturbation():
    from pdac_circuit.core.paths import ROOT

    report=validate_claim_surface_contract(ROOT)
    assert report["ok"] is True
    assert report["axes"] == 7
    assert report["external_perturbation_groups"] == [
        "ekstrom_johnsen::L36pl_clone2",
        "cunniff_vakoc::AsPC1",
        "cunniff_vakoc::T3M4",
    ]
    registry=json.loads((ROOT / "chromatin_registry.json").read_text(encoding="utf-8"))
    axes=[row["axis"] for row in registry["rules"]]
    assert "within_study_perturbation_direction" in axes
    assert "external_KLF5_perturbation_direction" in axes
    assets=json.loads((ROOT / "pdac_chromatin_assets.json").read_text(encoding="utf-8"))
    by_id={row["id"]: row for row in assets["assets"]}
    assert by_id["GSE146486"]["split"] == "excluded_from_PDAC_claim"
    assert by_id["GSE301272"]["split"] == "external_study_test"

def test_registered_axis_groups_reject_pseudoreplicate_inflation():
    allowed=["program-a::line-1", "program-b::line-2"]
    bundle=PredictionBundle(
        model="candidate",
        example_id=np.asarray(["a", "b", "c", "d"]),
        target=np.asarray([1.0, -1.0, 1.0, -1.0]),
        prediction=np.asarray([1.0, -1.0, 1.0, -1.0]),
        group=np.asarray([allowed[0], allowed[0], allowed[1], allowed[1]]),
        split=np.repeat("external_study_test", 4),
    )
    report=validate_registered_axis_groups(
        bundle,
        split="external_study_test",
        allowed_groups=allowed,
        exact_groups_required=True,
    )
    assert report["observed_groups"] == allowed
    inflated=replace(
        bundle,
        group=np.asarray(
            [allowed[0], f"{allowed[0]}::rep2", allowed[1], allowed[1]]
        ),
    )
    with pytest.raises(ValueError, match="unexpected"):
        validate_registered_axis_groups(
            inflated,
            split="external_study_test",
            allowed_groups=allowed,
            exact_groups_required=True,
        )

def test_external_klf5_decoders_encode_zero_control_and_signed_four_hour_loss():
    def sample(title: str, gsm: str = "GSM1") -> dict:
        return {
            "gsm": gsm,
            "title": title,
            "source_name": "human pancreatic cancer cell line",
            "organism": "Homo sapiens",
            "characteristics": {},
            "data_processing": ["reads aligned to hg38"],
        }

    timecourse_asset={"metadata_decoder": "external_klf5_dtag_timecourse_v1"}
    control=resolve_sample_metadata(
        sample("ATACseq_L36plClone2_dTAG_0h_Rep1"), timecourse_asset
    )
    treatment=resolve_sample_metadata(
        sample("ATACseq_L36plClone2_dTAG_4h_Rep1"), timecourse_asset
    )
    assert control["sample_group"] == "ekstrom_johnsen::L36pl_clone2"
    assert control["pair_relation"] == "control"
    assert not any(control["perturbation_features"])
    assert treatment["pair_relation"] == "intervention"
    assert treatment["perturbation_features"][16] == -1.0
    assert treatment["perturbation_features"][19] == pytest.approx(4 / 24)
    lineage=resolve_sample_metadata(
        sample("ChIPseq_AsPC1_dTAG_4h_H3K27ac_Rep2", gsm="GSM2"),
        {"metadata_decoder": "external_klf5_lineage_dtag_v1"},
    )
    assert lineage["sample_group"] == "cunniff_vakoc::AsPC1"
    assert lineage["canonical_assay"] == "H3K27ac"
    assert lineage["pair_relation"] == "intervention"

def test_raw_contrast_is_exact_id_aligned_and_truth_blind(tmp_path):
    ids=np.asarray(["a", "b"])
    common={
        "schema": "pdac-circuit.raw-predictions/1",
        "model": "PDACircuitFormer",
        "model_version": "frozen",
        "weights_sha256": "a" * 64,
        "track_mapping_sha256": "b" * 64,
        "seed": 11,
        "crop_bins": 2,
        "reverse_complement": True,
        "local_tiling": False,
        "label_free_cohort": "external",
        "ablate_state_residual": False,
    }
    reference=tmp_path / "reference.npz"
    treatment=tmp_path / "treatment.npz"
    save_raw_predictions(
        reference,
        model="PDACircuitFormer",
        example_id=ids,
        prediction=np.asarray([[1.0, 2.0], [3.0, 4.0]]),
        metadata={**common, "ablate_intervention_residual": True},
    )
    save_raw_predictions(
        treatment,
        model="PDACircuitFormer",
        example_id=ids[::-1],
        prediction=np.asarray([[5.0, 7.0], [2.0, 5.0]]),
        metadata={**common, "ablate_intervention_residual": False},
    )
    out=tmp_path / "delta.npz"
    contrast_raw_predictions(
        reference, treatment, out, mode="perturbation", command="test contrast"
    )
    with np.load(out, allow_pickle=False) as archive:
        assert archive["example_id"].tolist() == ["a", "b"]
        assert np.allclose(archive["prediction"], [[1.0, 3.0], [2.0, 3.0]])
        metadata=json.loads(str(archive["metadata"].item()))
    assert metadata["component"] == "perturbation_delta"
    assert metadata["contrast"]["exact_example_ids"] is True

def test_external_pair_plan_nests_assays_and_replicates_inside_three_contexts(tmp_path):
    from pdac_circuit.core.paths import ROOT

    for relative in (
        "chromatin_registry.json",
        "pdac_chromatin_assets.json",
        "configs/chromatin-campaign.json",
        "configs/chromatin-claim-surfaces.json",
    ):
        destination=tmp_path / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(ROOT / relative, destination)
    definitions={
        "GSE301272": {"ekstrom_johnsen::L36pl_clone2": ["ATAC"]},
        "GSE301284": {"ekstrom_johnsen::L36pl_clone2": ["H3K27ac"]},
        "GSE295354": {
            "cunniff_vakoc::AsPC1": ["H3K27ac"],
            "cunniff_vakoc::T3M4": ["H3K27ac"],
        },
    }
    index_paths=[]
    for study, group_assays in definitions.items():
        index_path=tmp_path / "data" / "evaluation_track_specs" / study / "index.json"
        index_path.parent.mkdir(parents=True, exist_ok=True)
        written=[]
        for group, assays in group_assays.items():
            for assay in assays:
                for replicate in ("rep1", "rep2"):
                    pair_group=f"{study}:{group}:{assay}:{replicate}:KLF5_dTAG"
                    for relation in ("control", "intervention"):
                        track=f"{study}-{group.split('::')[-1]}-{assay}-{replicate}-{relation}"
                        vector=[0.0] * 22
                        if relation == "intervention":
                            vector[16]=-1.0
                            vector[18]=1.0
                            vector[19]=4 / 24
                            vector[20]=1.0
                            vector[21]=1.0
                        spec_path=index_path.parent / f"{track}.json"
                        spec_path.write_text(
                            json.dumps(
                                {
                                    "accession": track,
                                    "sample_group": group,
                                    "study": study,
                                    "sample_accession": f"GSM-{track}",
                                    "genome": "hg38",
                                    "perturbation_features": vector,
                                    "perturbation_label": (
                                        "none" if relation == "control" else "KLF5_dTAG_4h"
                                    ),
                                    "pair_group": pair_group,
                                    "pair_relation": relation,
                                    "pair_control_family": "KLF5_dTAG",
                                    "biological_replicate": replicate,
                                }
                            ),
                            encoding="utf-8",
                        )
                        written.append(
                            {
                                "track": track,
                                "assay": assay,
                                "spec": str(spec_path.relative_to(tmp_path)).replace("\\", "/"),
                            }
                        )
        index_path.write_text(
            json.dumps(
                {
                    "schema": "pdac-circuit.geo-track-specs/1",
                    "accession": study,
                    "evaluation_only": True,
                    "written": written,
                    "failures": [],
                }
            ),
            encoding="utf-8",
        )
        index_paths.append(index_path)
    report=build_external_perturbation_pair_plan(tmp_path, index_paths)
    assert report["unresolved"] == []
    assert len(report["pairs"]) == 8
    assert {row["independence_group"] for row in report["pairs"]} == {
        "ekstrom_johnsen::L36pl_clone2",
        "cunniff_vakoc::AsPC1",
        "cunniff_vakoc::T3M4",
    }
    merged=json.loads(Path(report["merged_index_path"]).read_text(encoding="utf-8"))
    assert merged["evaluation_only"] is True
    assert len(merged["written"]) == 16

def test_claim_suite_requires_condition_aware_pass_and_identical_candidate():
    from pdac_circuit.core.paths import ROOT

    registry=json.loads((ROOT / "chromatin_registry.json").read_text(encoding="utf-8"))
    axes=[row["axis"] for row in registry["rules"]]
    contract_hash=validate_claim_surface_contract(ROOT)["sha256"]

    def report(role: str, verdict: str = "BEATS_BASELINE") -> dict:
        identity=registry["comparison_model_policy"][role]
        return {
            "schema": "pdac-circuit.enformer-benchmark/1",
            "comparison_role": role,
            "candidate": identity["candidate_model"],
            "baseline": identity["baseline_model"],
            "verdict": verdict,
            "axes": [{"axis": axis, "passed": True} for axis in axes],
            "rules": [{"axis": axis} for axis in axes],
            "failed_required_axes": [],
            "missing_required_axes": [],
            "provenance": {
                "axes": {
                    axis: {
                        "candidate": {
                            "manifest": {
                                "prediction_bundle_sha256": "a" * 64,
                                "weights_sha256": "b" * 64,
                                "claim_surface_contract_sha256": contract_hash,
                                "seed_ensemble": {
                                    "registered_seeds": registry["candidate_seed_policy"][
                                        "registered_seeds"
                                    ]
                                },
                            }
                        }
                    }
                    for axis in axes
                }
            },
        }

    reports={
        "headline_enformer": report("headline_enformer"),
        "diagnostic_enformer_adapter": report("diagnostic_enformer_adapter"),
        "secondary_borzoi": report("secondary_borzoi", verdict="ABSTAIN"),
    }
    suite=claim_suite_report(registry, reports)
    assert suite["verdict"] == "BEATS_ENFORMER_WITH_CONDITION_AWARE_ROBUSTNESS"
    reports["diagnostic_enformer_adapter"]["provenance"]["axes"][axes[0]][
        "candidate"
    ]["manifest"]["prediction_bundle_sha256"]="c" * 64
    drifted=claim_suite_report(registry, reports)
    assert drifted["verdict"] == "ABSTAIN"
    assert any("candidate bundles differ" in failure for failure in drifted["failures"])
