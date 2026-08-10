from __future__ import annotations

import json

from ..core.paths import MANIFESTS, MODELS, REGISTRY_JSON, ROOT

def _check_imports():
    import importlib

    for m in ["core.contract", "stats", "validation.calibration", "harness.trainer",
              "targeting", "parts", "circuit", "seqopt", "grna", "scoring", "signal", "pipeline.orchestrator"]:
        importlib.import_module(f"pdac_circuit.{m}")
    return ("imports", "ok", "all modules import")

def _check_calibration():
    from ..validation.calibration import run_all

    rep=run_all(fast=True, write=False)
    if not rep["ok"]:
        return ("calibration", "fail", f"FDR={rep['fdr_control_0.05']:.3f} cov={rep['conformal_coverage_0.90']:.3f}")
    return ("calibration", "ok", f"FDR={rep['fdr_control_0.05']:.3f} cov={rep['conformal_coverage_0.90']:.3f} permI={rep['permutation_typeI_0.05']:.3f}")

def _check_circuit_golden():
    from ..circuit import assess, robust_circuit

    a=assess(robust_circuit())
    if not (a["steady_state_ok"] and a["stability_ok"]):
        return ("circuit-golden", "fail", "robust_circuit not stable")
    return ("circuit-golden", "ok", f"robust cert={a['cert']}")

def _check_model_fixtures():
    from ..harness.fixtures import load_fixture, verify_fixture

    specs=[
        ("promoter", "..parts.promoter_model", "PromoterModel"),
        ("enhancer", "..parts.enhancer_model", "EnhancerModel"),
        ("grna_ontarget", "..grna.efficiency_model", "GRNAModel"),
    ]
    import importlib

    results=[]
    for key, mod, cls in specs:
        fx=load_fixture(key)
        pt=MODELS / f"{key}.pt"
        if fx is None or not pt.exists():
            results.append((key, "skip"))
            continue
        m=getattr(importlib.import_module(mod, __package__), cls).load(pt)
        v=verify_fixture(key, m.cnn)
        if not v["ok"]:
            return ("model-fixtures", "fail", f"{key}: max_diff={v.get('max_abs_diff')}")
        results.append((key, "ok"))
    if not any(r[1] == "ok" for r in results):
        return ("model-fixtures", "warn", "no trained models present (run `pdac train --all`)")
    return ("model-fixtures", "ok", ",".join(f"{k}:{s}" for k, s in results))

def _check_gan_fixture():
    import json as _json

    from ..core.seeds import sha256_text

    fx=MODELS / "promoter_gan.fixture.json"
    pt=MODELS / "promoter_gan.pt"
    if not fx.exists() or not pt.exists():
        return ("gan-fixture", "skip", "GAN not trained")
    from ..generate.promoter_gan import PromoterGAN

    rec=_json.loads(fx.read_text(encoding="utf-8"))
    gan=PromoterGAN.load(pt)
    seqs=gan.generate(rec["n"], seed=0)
    if sha256_text("".join(seqs)) != rec["seqs_sha256"]:
        return ("gan-fixture", "fail", "GAN generation not reproducible")
    return ("gan-fixture", "ok", "GAN generation reproduces")

def _check_signal():
    from ..signal.chromatin import chromatin_features, classify_state

    if classify_state({"H3K4me3": 5.0, "H3K27ac": 3.0}) != "active_promoter":
        return ("signal-chromatin", "fail", "active-promoter classification wrong")
    if classify_state({"H3K27me3": 5.0}) != "polycomb_repressed":
        return ("signal-chromatin", "fail", "repressed classification wrong")
    if chromatin_features({"H3K27ac": 5.0, "H3K4me1": 4.0})["activity_score"] <= 0:
        return ("signal-chromatin", "fail", "activity sign wrong")
    from ..signal.precompute import load_states

    rep=load_states()
    detail="logic OK; " + (f"{rep['n_loci']} loci x {rep['n_bams_used']} ChIP BAMs precomputed" if rep else "states not precomputed")
    return ("signal-chromatin", "ok", detail)

def _check_data_honesty():
    from ..core.provenance import verify_provenance

    manifests=[m for m in MANIFESTS.glob("*.json") if not m.name.endswith(".heavy.json")]
    if not manifests:
        return ("data-honesty", "warn", "no manifests (run fetch-data)")
    for m in manifests:
        res=verify_provenance(json.loads(m.read_text(encoding="utf-8")), ROOT)
        if not res["ok"]:
            return ("data-honesty", "fail", f"{m.stem}: {res['failures'][0]['reason']}")
    return ("data-honesty", "ok", f"{len(manifests)} manifests honest")

def _check_gitignore():
    gi=(ROOT / ".gitignore").read_text(encoding="utf-8")
    for must in ["/data/raw/", "/models/*.pt", "__pycache__", ".venv"]:
        if must not in gi:
            return ("gitignore", "fail", f"missing {must}")
    return ("gitignore", "ok", "raw/weights/venv ignored")

def _check_prereg():
    reg=json.loads(REGISTRY_JSON.read_text(encoding="utf-8"))
    pre=reg.get("prereg", {})
    if "committed_at" not in pre:
        return ("prereg", "fail", "no committed_at")
    for mod in ["module_I", "module_II", "module_III", "module_IV", "module_V", "module_VI"]:
        if mod not in pre:
            return ("prereg", "fail", f"missing {mod}")
    return ("prereg", "ok", "frozen pre-registration for I-VI")

def _check_chromatin_program():

    from ..chromatin.circuit_audit import CircuitInterpretationGate
    from ..chromatin.config import load_chromatin_config
    from ..chromatin.streaming import TrackSpec, sha256_file

    registry=json.loads((ROOT / "chromatin_registry.json").read_text(encoding="utf-8"))
    if registry.get("schema") != "pdac-circuit.chromatin-registry/2":
        return ("chromatin-program", "fail", "unsupported chromatin registry schema")
    try:
        interpretation_gate=CircuitInterpretationGate.from_registry(registry)
    except (TypeError, ValueError) as exc:
        return ("chromatin-program", "fail", f"invalid circuit interpretation gate: {exc}")
    interpretation_policy=registry.get("circuit_interpretation_gate", {})
    if (
        interpretation_gate.minimum_seeds != 3
        or interpretation_gate.minimum_pairwise_linear_cka != 0.4
        or interpretation_gate.minimum_median_pairwise_linear_cka != 0.6
        or interpretation_gate.minimum_effective_rank != 4.0
        or interpretation_gate.maximum_single_factor_variance_fraction != 0.75
        or interpretation_policy.get("failure_status") != "ABSTAIN"
        or interpretation_policy.get("coordinate_identifiability_claimed") is not False
    ):
        return ("chromatin-program", "fail", "circuit interpretation policy drifted")
    conditioning=registry.get("conditioning", {})
    expected={
        "assay_features": len(conditioning.get("assay_features", [])),
        "state_features": len(conditioning.get("state_features", [])),
        "perturbation_features": len(conditioning.get("perturbation_features", [])),
    }
    campaign_path=ROOT / "configs" / "chromatin-campaign.json"
    campaign=json.loads(campaign_path.read_text(encoding="utf-8"))
    if campaign.get("schema") != "pdac-circuit.chromatin-campaign/1":
        return ("chromatin-program", "fail", "unsupported chromatin campaign schema")
    from ..chromatin.human_cohort import validate_human_cohort_contract

    try:
        human_cohort=validate_human_cohort_contract(ROOT)
    except (FileNotFoundError, KeyError, TypeError, ValueError) as exc:
        return ("chromatin-program", "fail", f"invalid human adaptation cohort: {exc}")
    from ..chromatin.claim_surfaces import validate_claim_surface_contract

    try:
        claim_surfaces=validate_claim_surface_contract(ROOT)
    except (FileNotFoundError, KeyError, TypeError, ValueError) as exc:
        return ("chromatin-program", "fail", f"invalid claim surfaces: {exc}")
    if claim_surfaces.get("axes") != len(registry.get("rules", [])):
        return ("chromatin-program", "fail", "claim-surface rule count drifted")
    seeds=campaign.get("seeds", [])
    if len(seeds) < 3 or len(set(seeds)) != len(seeds):
        return ("chromatin-program", "fail", "campaign requires at least three unique seeds")
    seed_policy=registry.get("candidate_seed_policy", {})
    if (
        seed_policy.get("registered_seeds") != seeds
        or seed_policy.get("minimum_seeds") != len(seeds)
        or seed_policy.get("aggregation") != "arithmetic_mean"
        or seed_policy.get("failure_status") != "ABSTAIN"
        or "strictly positive delta" not in seed_policy.get(
            "individual_axis_policy", ""
        )
    ):
        return ("chromatin-program", "fail", "candidate multi-seed claim policy drifted")
    pipeline_cli=(ROOT / "src" / "pdac_circuit" / "pipeline" / "cli.py").read_text(
        encoding="utf-8"
    )
    if (
        "chromatin-ensemble-seeds" not in pipeline_cli
        or "--candidate-seed-root" not in pipeline_cli
        or "--comparison-role" not in pipeline_cli
    ):
        return ("chromatin-program", "fail", "multi-seed claim executors are missing")
    comparison_policy=registry.get("comparison_model_policy", {})
    expected_comparisons={
        "headline_enformer": ("PDACircuitFormer", "Enformer", True),
        "diagnostic_enformer_adapter": (
            "PDACircuitFormer",
            "Enformer + grouped PDAC state adapter",
            False,
        ),
        "secondary_borzoi": ("PDACircuitFormer", "Borzoi", False),
    }
    observed_comparisons={
        role: (
            row.get("candidate_model"),
            row.get("baseline_model"),
            row.get("supports_requested_headline_claim"),
        )
        for role, row in comparison_policy.items()
    }
    if observed_comparisons != expected_comparisons:
        return ("chromatin-program", "fail", "comparison model identities drifted")
    claim_suite_policy=registry.get("claim_suite_policy", {})
    if (
        claim_suite_policy.get("schema")
        != "pdac-circuit.chromatin-claim-suite-policy/1"
        or claim_suite_policy.get("official_verdict")
        != "BEATS_ENFORMER_WITH_CONDITION_AWARE_ROBUSTNESS"
        or claim_suite_policy.get("required_roles")
        != ["headline_enformer", "diagnostic_enformer_adapter"]
        or claim_suite_policy.get("reported_secondary_roles") != ["secondary_borzoi"]
        or claim_suite_policy.get("failure_status") != "ABSTAIN"
        or "chromatin-benchmark-suite" not in pipeline_cli
    ):
        return ("chromatin-program", "fail", "condition-aware claim suite drifted")
    from ..chromatin.protected import (
        validate_protected_metadata_release,
        validate_protected_study_seal,
    )

    protected_policy=registry.get("protected_study_policy", {})
    seal_path=ROOT / protected_policy.get("pre_access_seal", "")
    release_path=ROOT / protected_policy.get("post_training_release", "")
    protected_accessions=protected_policy.get("accessions", [])
    if (
        protected_accessions
        != ["GSE124229", "GSE124230", "GSE301272", "GSE301284", "GSE295354"]
        or protected_policy.get("maximum_test_tuning_uses") != 0
        or protected_policy.get("target_download_requires_release") is not True
        or protected_policy.get("failure_status") != "SEALED"
        or not seal_path.is_file()
    ):
        return ("chromatin-program", "fail", "protected-study policy/seal is incomplete")
    seal_report=validate_protected_study_seal(ROOT, seal_path)
    if not seal_report["ok"] or [
        row.get("accession") for row in seal_report.get("seal", {}).get("studies", [])
    ] != protected_accessions:
        return ("chromatin-program", "fail", "protected-study pre-access seal drifted")
    if release_path.is_file():
        for accession in protected_accessions:
            if not validate_protected_metadata_release(
                ROOT, release_path, accession=accession
            )["ok"]:
                return (
                    "chromatin-program",
                    "fail",
                    f"protected metadata release drifted for {accession}",
                )
    elif any(
        (ROOT / "data" / "metadata" / "geo" / accession).exists()
        for accession in protected_accessions
    ):
        return (
            "chromatin-program",
            "fail",
            "protected metadata exists before a final-checkpoint release",
        )
    if (
        "chromatin-seal-protected-studies" not in pipeline_cli
        or "chromatin-authorize-protected-metadata" not in pipeline_cli
        or "--protected-release" not in pipeline_cli
    ):
        return ("chromatin-program", "fail", "protected-study executors are missing")
    stages=[row.get("stage") for row in campaign.get("curriculum", [])]
    if stages != list(range(1, len(stages) + 1)) or len(stages) < 5:
        return ("chromatin-program", "fail", "campaign curriculum stages are incomplete")
    if campaign.get("selection", {}).get("maximum_tuning_uses_of_test_surfaces") != 0:
        return ("chromatin-program", "fail", "campaign permits test-surface tuning")
    ablation_path=ROOT / "configs" / "chromatin-ablation-registry.json"
    ablation_registry=json.loads(ablation_path.read_text(encoding="utf-8"))
    ablation_rows=ablation_registry.get("ablations", [])
    if (
        ablation_registry.get("schema")
        != "pdac-circuit.chromatin-ablation-registry/1"
        or ablation_registry.get("test_tuning_uses") != 0
        or ablation_registry.get("seeds") != seeds
        or {row.get("label") for row in ablation_rows}
        != set(campaign.get("required_ablations", []))
        or len(ablation_rows) != len(campaign.get("required_ablations", []))
    ):
        return ("chromatin-program", "fail", "frozen ablation registry drifted")
    for row in ablation_rows:
        for key in ("config", "runner", "mapping", "legacy_checkpoint"):
            if row.get(key) and not (ROOT / row[key]).is_file():
                return (
                    "chromatin-program",
                    "fail",
                    f"ablation {row.get('label')} is missing {row[key]}",
                )
    profile_paths=[ROOT / row.get("config", "") for row in campaign.get("profiles", [])]
    if len(profile_paths) < 4 or any(not path.is_file() for path in profile_paths):
        return ("chromatin-program", "fail", "campaign hardware profiles are missing")
    ablation_profile_rows=campaign.get("ablation_profiles", [])
    ablation_profile_paths=[
        ROOT / row.get("config", "") for row in ablation_profile_rows
    ]
    if (
        len(ablation_profile_paths) != 5
        or any(not path.is_file() for path in ablation_profile_paths)
        or len({str(path) for path in profile_paths + ablation_profile_paths})
        != len(profile_paths) + len(ablation_profile_paths)
    ):
        return ("chromatin-program", "fail", "campaign ablation profiles are incomplete")
    train_stage_names=[row.get("name") for row in campaign.get("curriculum", [])[:4]]
    if campaign.get("execution", {}).get("require_complete_markers") is not True:
        return ("chromatin-program", "fail", "campaign permits partial compiled corpora")
    if campaign.get("execution", {}).get("baseline_asset_manifests") != {
        "enformer": "baseline_assets/enformer-model.json",
        "borzoi": "baseline_assets/borzoi-models.json",
    }:
        return ("chromatin-program", "fail", "campaign baseline asset manifests drifted")
    if campaign.get("execution", {}).get("baseline_target_maps") != {
        "enformer_human": "data/metadata/enformer_target_map.json",
        "enformer_mouse": "data/metadata/enformer_target_map_mouse.json",
        "borzoi_human": "data/metadata/borzoi_target_map.json",
    }:
        return ("chromatin-program", "fail", "campaign baseline target maps drifted")
    if campaign.get("execution", {}).get("baseline_adapter_configs") != {
        "enformer_human": "configs/enformer-state-adapter.json",
        "enformer_mouse": "configs/enformer-mouse-state-adapter.json",
    }:
        return ("chromatin-program", "fail", "campaign baseline adapter configs drifted")
    mouse_adapter_path=ROOT / "configs" / "enformer-mouse-state-adapter.json"
    mouse_adapter=json.loads(mouse_adapter_path.read_text(encoding="utf-8"))
    mouse_adapter_policy=mouse_adapter.get("training_policy", {})
    mouse_sampling=mouse_adapter_policy.get("target_blind_sampling", {})
    if (
        mouse_adapter.get("schema") != "pdac-circuit.enformer-state-adapter/1"
        or mouse_adapter_policy.get("head") != "mouse"
        or mouse_adapter_policy.get("allowed_genomes") != ["mm10", "mm9"]
        or mouse_adapter_policy.get("target_map")
        != "data/metadata/enformer_target_map_mouse.json"
        or mouse_adapter_policy.get("primary_rule")
        != "mouse_H3K27ac_all_tissues"
        or mouse_adapter_policy.get("validation_scope")
        != "locus_disjoint_same_groups_allowed"
        or mouse_sampling.get("method")
        != "sha256_rank_within_group_and_exact_condition_vector"
        or mouse_sampling.get("seed") != 20260715
        or mouse_sampling.get("maximum_train_examples_per_condition_group") != 1024
        or mouse_sampling.get("maximum_validation_examples_per_condition_group") != 256
        or mouse_sampling.get("test_surface_capped") is not False
    ):
        return ("chromatin-program", "fail", "mouse Enformer adapter policy drifted")
    compile_contracts=campaign.get("execution", {}).get("stage_compile_contracts", {})
    if set(compile_contracts) != set(train_stage_names):
        return ("chromatin-program", "fail", "campaign compile contracts are incomplete")
    expected_negative_sampling={
        "healthy_prior": 0.05,
        "progression_state_residual": 1.0,
        "signed_intervention_residual": 1.0,
        "human_state_adaptation": 1.0,
    }
    observed_negative_sampling={
        stage: contract.get("negative_keep_probability")
        for stage, contract in compile_contracts.items()
    }
    if observed_negative_sampling != expected_negative_sampling:
        return ("chromatin-program", "fail", "campaign negative-window policy drifted")
    stage_data=campaign.get("execution", {}).get("stage_data", {})
    if set(stage_data) != set(train_stage_names):
        return ("chromatin-program", "fail", "campaign stage-to-data bindings are incomplete")
    if (
        stage_data.get("signed_intervention_residual")
        != "data/processed/chromatin_gse99311_paired_v1/**/*.npz"
        or stage_data.get("human_state_adaptation") != human_cohort["stage_globs"]
        or "pair-plan materialization" not in campaign.get("execution", {}).get(
            "data_binding", ""
        )
    ):
        return (
            "chromatin-program",
            "fail",
            "signed intervention stage is not bound to certified paired deltas",
        )
    registered_profiles={
        row.get("config")
        for row in campaign.get("profiles", []) + ablation_profile_rows
    }
    for profile, bindings in campaign.get("execution", {}).get(
        "profile_stage_data", {}
    ).items():
        if profile not in registered_profiles or set(bindings) != set(train_stage_names):
            return (
                "chromatin-program",
                "fail",
                f"invalid profile-specific data binding for {profile}",
            )
        if (
            profile == "configs/chromatin-scale-80gb.json"
            and bindings.get("signed_intervention_residual")
            != "data/processed/chromatin_gse99311_paired_524kb_v1/**/*.npz"
        ):
            return (
                "chromatin-program",
                "fail",
                "524 kb intervention stage is not bound to paired deltas",
            )
    plan_paths=sorted((ROOT / "results" / "frozen").glob("chromatin-campaign-*-plan.json"))
    if len(plan_paths) != len(profile_paths):
        return ("chromatin-program", "fail", "not every hardware profile has a frozen run DAG")
    for plan_path in plan_paths:
        plan=json.loads(plan_path.read_text(encoding="utf-8"))
        if (
            plan.get("schema") != "pdac-circuit.chromatin-campaign-plan/1"
            or plan.get("campaign_sha256") != sha256_file(campaign_path)
            or plan.get("node_count") != len(seeds) * len(train_stage_names)
            or {node.get("seed") for node in plan.get("nodes", [])} != set(seeds)
            or any(
                not str(node.get("selected_checkpoint", "")).endswith("best.pt")
                or not str(node.get("resume_checkpoint", "")).endswith("latest.pt")
                or (node.get("depends_on") and "--initialize-from" not in node.get("argv", []))
                for node in plan.get("nodes", [])
            )
        ):
            return ("chromatin-program", "fail", f"invalid frozen run DAG {plan_path.name}")
    ablation_plan_paths=sorted(
        (ROOT / "results" / "frozen").glob("chromatin-ablation-*-plan.json")
    )
    expected_ablation_profiles={
        row["config"]: 3 if row.get("reuse_healthy_from") else 4
        for row in ablation_profile_rows
    }
    if len(ablation_plan_paths) != len(expected_ablation_profiles):
        return ("chromatin-program", "fail", "ablation run DAGs are incomplete")
    observed_ablation_profiles=set()
    for plan_path in ablation_plan_paths:
        plan=json.loads(plan_path.read_text(encoding="utf-8"))
        profile=plan.get("profile", {}).get("config")
        expected_stages=expected_ablation_profiles.get(profile)
        if (
            plan.get("schema") != "pdac-circuit.chromatin-campaign-plan/1"
            or plan.get("campaign_sha256") != sha256_file(campaign_path)
            or expected_stages is None
            or plan.get("node_count") != len(seeds) * expected_stages
            or {node.get("seed") for node in plan.get("nodes", [])} != set(seeds)
            or any(
                not str(node.get("selected_checkpoint", "")).endswith("best.pt")
                or not str(node.get("resume_checkpoint", "")).endswith("latest.pt")
                or (node.get("depends_on") and "--initialize-from" not in node.get("argv", []))
                for node in plan.get("nodes", [])
            )
        ):
            return (
                "chromatin-program",
                "fail",
                f"invalid frozen ablation DAG {plan_path.name}",
            )
        observed_ablation_profiles.add(profile)
    if observed_ablation_profiles != set(expected_ablation_profiles):
        return ("chromatin-program", "fail", "ablation DAG profiles drifted")
    baseline_maps={
        "enformer": (
            ROOT / "data" / "metadata" / "enformer_target_map.json",
            "pdac-circuit.enformer-target-map/1",
            "human",
        ),
        "enformer-mouse": (
            ROOT / "data" / "metadata" / "enformer_target_map_mouse.json",
            "pdac-circuit.enformer-target-map/1",
            "mouse",
        ),
        "borzoi": (
            ROOT / "data" / "metadata" / "borzoi_target_map.json",
            "pdac-circuit.borzoi-target-map/1",
            "human",
        ),
    }
    loaded_maps={}
    for name, (path, schema, expected_head) in baseline_maps.items():
        if not path.is_file():
            return ("chromatin-program", "fail", f"missing frozen {name} target map")
        mapping=json.loads(path.read_text(encoding="utf-8"))
        if (
            mapping.get("schema") != schema
            or mapping.get("label_blind") is not True
            or mapping.get("head") != expected_head
            or not mapping.get("rules")
        ):
            return ("chromatin-program", "fail", f"invalid frozen {name} target map")
        loaded_maps[name]=mapping
    borzoi_map=loaded_maps["borzoi"]
    enformer_commit="24403ec79bc71c803c258efce9e98ddc2ca9a48d"
    if any(
        loaded_maps[name].get("source_commit") != enformer_commit
        for name in ("enformer", "enformer-mouse")
    ):
        return ("chromatin-program", "fail", "Enformer target metadata is not commit-pinned")
    mouse_map=loaded_maps["enformer-mouse"]
    mouse_policy=ROOT / "enformer_mouse_target_policy.json"
    mouse_metadata=ROOT / "data" / "metadata" / "enformer_targets_mouse.txt"
    mouse_rules={row.get("name"): row for row in mouse_map.get("rules", [])}
    if (
        mouse_map.get("head_index_offset") != 5313
        or mouse_map.get("output_target_count") != 1643
        or mouse_map.get("allowed_genomes") != ["mm10", "mm9"]
        or mouse_map.get("policy_sha256") != sha256_file(mouse_policy)
        or mouse_map.get("target_metadata_sha256") != sha256_file(mouse_metadata)
        or len(
            mouse_rules.get("mouse_H3K27ac_all_tissues", {}).get(
                "target_indices", []
            )
        )
        != 107
    ):
        return ("chromatin-program", "fail", "Enformer mouse-head contract drifted")
    enformer_assets_path=ROOT / "baseline_assets" / "enformer-model.json"
    enformer_assets=json.loads(enformer_assets_path.read_text(encoding="utf-8"))
    enformer_environment=(
        ROOT / "environments" / "enformer-baseline.yml"
    ).read_text(encoding="utf-8")
    enformer_runner=(
        ROOT / "baseline_runners" / "enformer_export.py"
    ).read_text(encoding="utf-8")
    if (
        enformer_assets.get("schema") != "pdac-circuit.enformer-assets/1"
        or enformer_assets.get("model_url")
        != "https://tfhub.dev/deepmind/enformer/1"
        or enformer_assets.get("cache_destination")
        != "baseline_assets/enformer/tfhub-cache"
        or enformer_assets.get("training_use") != "predictions_only"
        or "tensorflow==2.15.*" not in enformer_environment
        or "tensorflow-hub==0.16.*" not in enformer_environment
        or "--model-url" in enformer_runner
        or "verify_materialized_enformer_assets" not in enformer_runner
    ):
        return ("chromatin-program", "fail", "Enformer asset contract drifted")
    for required in (
        ROOT / "baseline_runners" / "enformer_assets.py",
        ROOT / "baseline_runners" / "fetch_enformer_assets.py",
        ROOT / "baseline_runners" / "enformer_export.py",
        ROOT / "environments" / "enformer-baseline.yml",
    ):
        if not required.is_file():
            return ("chromatin-program", "fail", f"missing Enformer runtime: {required.name}")
    if (
        borzoi_map.get("source_commit")
        != "5c9358222b5026abb733ed5fb84f3f6c77239b37"
        or borzoi_map.get("target_count") != 7611
        or len(borzoi_map.get("strand_pair_index", [])) != 7611
    ):
        return ("chromatin-program", "fail", "Borzoi target/strand transform drifted")
    assets_path=ROOT / "baseline_assets" / "borzoi-models.json"
    assets=json.loads(assets_path.read_text(encoding="utf-8"))
    models=assets.get("models", [])
    if (
        assets.get("schema") != "pdac-circuit.borzoi-assets/1"
        or len(models) != 4
        or len({row.get("url") for row in models}) != 4
        or {row.get("expected_bytes") for row in models} != {744112468}
        or len({row.get("expected_md5") for row in models}) != 4
    ):
        return ("chromatin-program", "fail", "Borzoi four-replicate asset plan drifted")
    params=assets.get("parameters", {})
    params_path=ROOT / params.get("destination", "")
    if (
        not params_path.is_file()
        or not params.get("sha256")
        or sha256_file(params_path) != params["sha256"]
    ):
        return ("chromatin-program", "fail", "Borzoi parameter artifact is not frozen")
    for required in (
        ROOT / "baseline_runners" / "borzoi_export.py",
        ROOT / "baseline_runners" / "fetch_borzoi_assets.py",
        ROOT / "environments" / "borzoi-baseline.yml",
    ):
        if not required.is_file():
            return ("chromatin-program", "fail", f"missing Borzoi runtime: {required.name}")
    for path in sorted((ROOT / "configs").glob("chromatin-*.json")):
        payload=json.loads(path.read_text(encoding="utf-8"))
        if payload.get("schema") in {
            "pdac-circuit.chromatin-campaign/1",
            "pdac-circuit.chromatin-ablation-registry/1",
        }:
            continue
        model, _, _=load_chromatin_config(path)
        observed={key: getattr(model, key) for key in expected}
        if observed != expected:
            return (
                "chromatin-program",
                "fail",
                f"{path.name} conditioning {observed} != registry {expected}",
            )
        if model.domain_state_features != len(conditioning.get("domain_state_suffix", [])):
            return (
                "chromatin-program",
                "fail",
                f"{path.name} domain-state suffix does not match registry",
            )

    from ..chromatin.model import build_chromatin_model
    from ..chromatin.trainer import parameter_report

    capacity_configs={
        "candidate": ROOT / "configs" / "chromatin-local-12gb.json",
        "mean_only": ROOT
        / "configs"
        / "chromatin-ablation-mean-only-landmarks.json",
        "direct_long": ROOT / "configs" / "chromatin-ablation-direct-long-cnn.json",
        "direct_local": ROOT / "configs" / "chromatin-ablation-direct-2kb-cnn.json",
    }
    capacity={}
    capacity_models={}
    for name, path in capacity_configs.items():
        model_config, _, _=load_chromatin_config(path)
        capacity_models[name]=model_config
        capacity[name]=parameter_report(build_chromatin_model(model_config))["parameters"]
    if (
        capacity["mean_only"] != capacity["candidate"]
        or capacity_models["candidate"].landmark_routing != "dual_statistic"
        or capacity_models["mean_only"].landmark_routing != "mean_only"
        or capacity["direct_long"] != capacity["direct_local"]
        or abs(capacity["direct_long"] - capacity["candidate"])
        / capacity["candidate"]
        > 0.01
        or capacity_models["direct_long"].architecture
        != "direct_conditional_cnn"
        or capacity_models["direct_long"].sequence_length != 196_608
        or capacity_models["direct_local"].sequence_length != 2_048
    ):
        return ("chromatin-program", "fail", "matched-capacity architecture controls drifted")

    metadata_count=0
    for path in sorted((ROOT / "data" / "metadata" / "geo").glob("GSE*/metadata.json")):
        payload=json.loads(path.read_text(encoding="utf-8"))
        if payload.get("errors"):
            return (
                "chromatin-program",
                "fail",
                f"{payload.get('accession')} has unresolved authoritative metadata",
            )
        if payload.get("accession") == "GSE99311":
            allowed_families={
                "unperturbed",
                "mscv_overexpression",
                "mire_shrna",
                "lentiviral_crispr",
            }
            families={
                row.get("perturbation_control_family")
                for row in payload.get("samples", {}).values()
            }
            if families - allowed_families or not families:
                return (
                    "chromatin-program",
                    "fail",
                    "GSE99311 perturbation control technologies are unresolved",
                )
        metadata_count += 1

    spec_count=0
    for index_path in sorted((ROOT / "data" / "track_specs").glob("*/index.json")):
        index=json.loads(index_path.read_text(encoding="utf-8"))
        if index.get("failures"):
            first=index["failures"][0]
            identity=first.get("accession") or first.get("path") or "unknown track"
            return (
                "chromatin-program",
                "fail",
                f"{index_path.parent.name} has unresolved TrackSpec failure for {identity}",
            )
        if index.get("schema") == "pdac-circuit.encode-track-specs/1":
            policy_path=ROOT / index.get("selection_policy", "")
            if (
                not policy_path.is_file()
                or sha256_file(policy_path) != index.get("selection_policy_sha256")
                or index.get("candidate_tracks")
                != len(index.get("written", [])) + len(index.get("excluded", []))
            ):
                return (
                    "chromatin-program",
                    "fail",
                    "ENCODE healthy selection policy or accounting drifted",
                )
        for row in index.get("written", []):
            spec_path=ROOT / row["spec"]
            payload=json.loads(spec_path.read_text(encoding="utf-8"))
            for key in ("assay_features", "state_features", "perturbation_features"):
                payload[key]=tuple(payload[key])
            TrackSpec(**payload).validate()
            conditioning_drift=[]
            for key, expected_length in expected.items():
                values=payload[key]
                exact_zero_padding=(
                    key == "perturbation_features"
                    and 0 < len(values) < expected_length
                    and not any(values)
                )
                if len(values) != expected_length and not exact_zero_padding:
                    conditioning_drift.append(key)
            if conditioning_drift:
                return (
                    "chromatin-program",
                    "fail",
                    f"conditioning drift in {spec_path.name}: {conditioning_drift}",
                )
            spec_count += 1
    gse99311_index=ROOT / "data" / "track_specs" / "GSE99311" / "index.json"
    if gse99311_index.is_file():
        pair_plan_path=ROOT / "data" / "pair_specs" / "GSE99311.intervention.json"
        if not pair_plan_path.is_file():
            return ("chromatin-program", "fail", "GSE99311 pair plan is missing")
        pair_plan=json.loads(pair_plan_path.read_text(encoding="utf-8"))
        pairs=pair_plan.get("pairs", [])
        within_rule=next(
            row
            for row in registry.get("rules", [])
            if row.get("axis") == "within_study_perturbation_direction"
        )
        primary_pairs=[
            row for row in pairs if row.get("assay") in within_rule.get("primary_assays", [])
        ]
        observed_primary_groups=sorted(
            {str(row.get("independence_group", "")) for row in primary_pairs}
        )
        realized=registry.get("mouse_Enformer_and_realized_GSE99311_amendment", {})
        if (
            pair_plan.get("schema")
            != "pdac-circuit.intervention-pair-plan/1"
            or pair_plan.get("track_index_sha256") != sha256_file(gse99311_index)
            or pair_plan.get("unresolved")
            or len(pairs) != realized.get("realized_profile_pairs")
            or len(primary_pairs) != realized.get("primary_H3K27ac_pairs")
            or set(observed_primary_groups)
            != set(within_rule.get("allowed_groups", []))
            or any(
                not row.get("assay") or not row.get("independence_group")
                for row in pairs
            )
            or any(
                not row.get("reference", {}).get("pair_control_family")
                or row.get("reference", {}).get("pair_control_family")
                != row.get("treatment", {}).get("pair_control_family")
                for row in pair_plan.get("pairs", [])
            )
        ):
            return (
                "chromatin-program",
                "fail",
                "GSE99311 exact vector/control pair plan drifted",
            )
        paired_root=ROOT / "data" / "processed" / "chromatin_gse99311_paired_v1"
        if paired_root.exists():
            completion_path=paired_root / "_COMPLETE.json"
            if not completion_path.is_file():
                return (
                    "chromatin-program",
                    "fail",
                    "GSE99311 paired corpus exists without an atomic completion marker",
                )
            completion=json.loads(completion_path.read_text(encoding="utf-8"))
            if (
                completion.get("schema")
                != "pdac-circuit.paired-collection-completion/1"
                or completion.get("successful") is not True
                or completion.get("pair_plan_sha256") != sha256_file(pair_plan_path)
                or completion.get("registered_pairs") != len(pairs)
                or completion.get("verified_pairs") != len(pairs)
                or completion.get("minimum_required_overlap_fraction") != 0.995
                or float(completion.get("minimum_observed_overlap_fraction", 0.0))
                < 0.995
                or any(
                    not row.get("assay")
                    or not row.get("independence_group")
                    or float(row.get("overlap_fraction", 0.0)) < 0.995
                    for row in completion.get("pairs", [])
                )
            ):
                return (
                    "chromatin-program",
                    "fail",
                    "GSE99311 paired-corpus overlap or provenance contract drifted",
                )
    return (
        "chromatin-program",
        "ok",
        f"{metadata_count} GEO registries resolved; {spec_count} runnable TrackSpecs valid; "
        f"{len(seeds)} seeds/{len(profile_paths)} hardware profiles/DAGs; "
        f"human/mouse Enformer+Borzoi label-blind maps frozen",
    )

def _check_gpu():
    try:
        import torch

        if torch.cuda.is_available():
            return ("gpu", "ok", torch.cuda.get_device_name(0))
    except Exception:
        pass
    return ("gpu", "warn", "CUDA unavailable (CPU fallback)")

CHECKS=[_check_imports, _check_calibration, _check_circuit_golden, _check_model_fixtures,
          _check_gan_fixture, _check_signal, _check_data_honesty, _check_gitignore, _check_prereg,
          _check_chromatin_program, _check_gpu]

def run_predeploy() -> int:
    print("[predeploy] fail-closed gates")
    failed=0
    for chk in CHECKS:
        try:
            name, status, detail=chk()
        except Exception as e:
            name, status, detail=(chk.__name__, "fail", f"{type(e).__name__}: {e}")
        mark={"ok": "OK  ", "warn": "WARN", "skip": "SKIP", "fail": "FAIL"}[status]
        print(f"  {mark} {name:16s} {detail}")
        if status == "fail":
            failed += 1
    print(f"[predeploy] {'GREEN' if failed == 0 else f'{failed} FAILURE(S)'}")
    return 1 if failed else 0
