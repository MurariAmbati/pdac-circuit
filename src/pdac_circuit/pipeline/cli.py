from __future__ import annotations

import argparse
import sys

def _cmd_fetch_data(args: argparse.Namespace) -> int:
    from ..data.fetch import run_fetch

    return run_fetch(
        only=args.corpus,all_open=args.all_open,all_corpora=args.all,
        heavy=args.heavy,dry_run=args.dry_run,list_only=args.list,
    )

def _cmd_verify_data(args: argparse.Namespace) -> int:
    from ..data.fetch import run_verify

    return run_verify()

def _cmd_bulk_fetch(args: argparse.Namespace) -> int:
    from ..data.bulk import bulk_fetch

    return bulk_fetch(target_gb=args.target_gb,workers=args.workers,include_gnomad=not args.no_gnomad)

def _cmd_train(args: argparse.Namespace) -> int:
    from ..harness.run_training import run_training

    only = args.only.split(",") if args.only else None
    return run_training(only=only,quick=args.quick)

def _cmd_run_pipeline(args: argparse.Namespace) -> int:
    from .orchestrator import run_pipeline

    return run_pipeline(subtype=args.subtype,top_k=args.top_k,seed=args.seed,out=args.out)

def _cmd_run_deep(args: argparse.Namespace) -> int:
    from .deep import run_deep_design

    return run_deep_design(
        subtype=args.subtype,max_targets=args.max_targets,multi_top=args.multi_top,
        sweep_n=args.sweep_n,seed=args.seed,out=args.out,
    )

def _cmd_attractor_design(args: argparse.Namespace) -> int:

    from ..attractor.run import run_attractor_control

    res = run_attractor_control(
        max_nodes=args.max_nodes,
        coexpr_threshold=args.coexpr_threshold,
        motif_edges=not args.no_motif,
        epochs=args.epochs,
        max_control_targets=args.max_targets,
        ensemble=args.ensemble,
        seed=args.seed,
    )
    v = res["validation"]
    g = res["graph"]
    print("RAC regulatory-attractor-control complete (data_class REAL)")
    print(f"  graph: {g['n_nodes']} nodes, {g['n_edges']} edges, "
          f"{g['n_motif_supported_edges']} motif-supported, {g['n_pdac_lines']} PDAC lines; "
          f"CNA {g.get('cna_covered',0)}/{g['n_nodes']} genes ({g.get('n_amplified_nodes',0)} amplified)")
    print(f"  fit: fixed-point error {res['fit']['fixed_point_error']} on {res['fit']['device']}")
    if v.get("primary"):
        p = v["primary"]
        print(f"  validation (DepMap held out): AUC collapse {p['auc_collapse']} "
              f"CI{p['auc_collapse_ci95']} vs degree {p['auc_degree']} / eigen {p['auc_eigencentrality']} "
              f"(perm p {v['permutation_p_primary']})")
    if v.get("ensemble"):
        e = v["ensemble"]
        print(f"  ensemble ({e['members']} fits): AUC thr0.4 {e['auc_ensemble_thr0.4_mean_ci'][0]} "
              f"CI{e['auc_ensemble_thr0.4_mean_ci'][1]} worst-member {e['auc_ensemble_thr0.4_worst_member']}")
    print(f"  control targets: {res['control']['targets']}  net healthy shift {res['control']['net_healthy_shift']}")
    print("  top convergent targets:")
    for t in res["convergent_targets"][:8]:
        print(f"    {t['gene']:9} conv={t['convergence_score']} up={t['disease_log2fc']:+.2f} "
              f"{t['healthy_action']} ess={t['abs_essential']} driver={t['intogen_driver']} {t['subtype']}")
    print("  wrote results/attractor_{map,validation,control,targets}.json")
    return 0

def _cmd_predeploy(args: argparse.Namespace) -> int:
    from .predeploy import run_predeploy

    return run_predeploy()

def _cmd_figures(args: argparse.Namespace) -> int:
    from .figures import make_figures

    made = make_figures()
    for m in made:
        print("wrote",m)
    return 0

def _cmd_chromatin_inventory(args: argparse.Namespace) -> int:
    from ..chromatin.cli import run_inventory

    return run_inventory(out=args.out)

def _cmd_chromatin_study_plan(args: argparse.Namespace) -> int:
    from ..chromatin.cli import run_study_plan

    return run_study_plan(
        accession=args.accession,
        out=args.out,
        skip_size_probe=args.skip_size_probe,
    )

def _cmd_chromatin_fetch_study(args: argparse.Namespace) -> int:
    from ..chromatin.cli import run_fetch_study

    return run_fetch_study(
        plan_path=args.plan,
        output_root=args.output_root,
        allow_protected_study=args.allow_protected_study,
        protected_seal=args.protected_seal,
        protected_release=args.protected_release,
        max_total_gb=args.max_total_gb,
    )

def _cmd_chromatin_inspect_geo_archive(args: argparse.Namespace) -> int:
    from ..chromatin.cli import run_inspect_geo_archive

    return run_inspect_geo_archive(archive=args.archive,out=args.out)

def _cmd_chromatin_extract_geo_archive(args: argparse.Namespace) -> int:
    from ..chromatin.cli import run_extract_geo_archive

    return run_extract_geo_archive(
        archive=args.archive,
        output=args.output,
        max_unpacked_gb=args.max_unpacked_gb,
    )

def _cmd_chromatin_geo_metadata(args: argparse.Namespace) -> int:
    from ..chromatin.cli import run_geo_metadata

    return run_geo_metadata(
        accession=args.accession,
        refresh=args.refresh,
        allow_protected_metadata=args.allow_protected_metadata,
        protected_release=args.protected_release,
    )

def _cmd_chromatin_seal_protected(args: argparse.Namespace) -> int:
    from ..chromatin.cli import run_seal_protected_studies

    return run_seal_protected_studies(
        campaign=args.campaign,
        registry=args.registry,
        assets=args.assets,
        out=args.out,
    )

def _cmd_chromatin_authorize_protected(args: argparse.Namespace) -> int:
    from ..chromatin.cli import run_authorize_protected_metadata

    return run_authorize_protected_metadata(
        seal=args.seal,
        checkpoints=args.checkpoint,
        out=args.out,
    )

def _cmd_chromatin_geo_track_specs(args: argparse.Namespace) -> int:
    from ..chromatin.cli import run_geo_track_specs

    return run_geo_track_specs(
        accession=args.accession,
        extracted_dir=args.extracted,
        metadata_path=args.metadata,
        evaluation_only=args.evaluation_only,
        protected_release=args.protected_release,
    )

def _cmd_chromatin_fetch_reference(args: argparse.Namespace) -> int:
    from ..chromatin.cli import run_fetch_reference

    return run_fetch_reference(
        genome=args.genome,
        discard_compressed=args.discard_compressed,
    )

def _cmd_chromatin_intervention_pair_plan(args: argparse.Namespace) -> int:
    from ..chromatin.cli import run_intervention_pair_plan

    return run_intervention_pair_plan(track_index=args.track_index,out=args.out)

def _cmd_chromatin_external_perturbation_pair_plan(args: argparse.Namespace) -> int:
    from ..chromatin.cli import run_external_perturbation_pair_plan

    return run_external_perturbation_pair_plan(
        track_indexes=args.track_index,
        contract=args.contract,
        out=args.out,
        merged_index_out=args.merged_index_out,
    )

def _cmd_chromatin_model_info(args: argparse.Namespace) -> int:
    from ..chromatin.cli import run_model_info

    return run_model_info(
        config_path=args.config,
        forward_check=args.forward_check,
        device=args.device,
        minimum_free_gb=args.min_free_vram_gb,
        allow_low_vram=args.allow_low_vram,
    )

def _cmd_chromatin_encode_specs(args: argparse.Namespace) -> int:
    from ..chromatin.cli import run_encode_specs

    return run_encode_specs(
        refresh=args.refresh,
        limit=args.limit,
        metadata_workers=args.metadata_workers,
    )

def _cmd_chromatin_enformer_target_map(args: argparse.Namespace) -> int:
    from ..chromatin.cli import run_enformer_target_map

    return run_enformer_target_map(
        refresh=args.refresh,
        out=args.out,
        policy=args.policy,
        metadata_cache=args.metadata_cache,
    )

def _cmd_chromatin_borzoi_target_map(args: argparse.Namespace) -> int:
    from ..chromatin.cli import run_borzoi_target_map

    return run_borzoi_target_map(refresh=args.refresh,out=args.out)

def _cmd_chromatin_plan_campaign(args: argparse.Namespace) -> int:
    from ..chromatin.cli import run_plan_campaign

    return run_plan_campaign(
        campaign_path=args.campaign,
        profile_config=args.profile,
        out=args.out,
    )

def _cmd_chromatin_zero_baseline(args: argparse.Namespace) -> int:
    from ..chromatin.cli import run_zero_baseline

    return run_zero_baseline(
        truth_path=args.truth,
        out=args.out,
        model=args.model,
        model_version=args.model_version,
        weights_sha256=args.weights_sha256,
        track_mapping_sha256=args.track_mapping_sha256,
    )

def _cmd_chromatin_assemble_bundle(args: argparse.Namespace) -> int:
    from ..chromatin.cli import run_assemble_bundle

    return run_assemble_bundle(
        raw_path=args.raw,
        truth_path=args.truth,
        out=args.out,
        provenance_out=args.provenance_out,
        training_use=args.training_use,
        claim_surface_contract=args.claim_surface_contract,
    )

def _cmd_chromatin_contrast_raw(args: argparse.Namespace) -> int:
    from ..chromatin.cli import run_contrast_raw

    return run_contrast_raw(
        reference_raw=args.reference_raw,
        treatment_raw=args.treatment_raw,
        out=args.out,
        mode=args.mode,
    )

def _cmd_chromatin_conformalize(args: argparse.Namespace) -> int:
    from ..chromatin.cli import run_conformalize

    return run_conformalize(
        calibration_raw_path=args.calibration_raw,
        calibration_truth_path=args.calibration_truth,
        target_raw_path=args.target_raw,
        out=args.out,
        nominal=args.nominal,
    )

def _cmd_chromatin_freeze_profile_truth(args: argparse.Namespace) -> int:
    from ..chromatin.cli import run_freeze_profile_truth

    return run_freeze_profile_truth(
        shards_glob=args.shards,
        split=args.split,
        out=args.out,
        crop_bins=args.crop_bins,
        genomes=args.genome,
        example_ids_from=args.example_ids_from,
        target_field=args.target_field,
        mask_field=args.mask_field,
    )

def _cmd_chromatin_freeze_evaluation_windows(args: argparse.Namespace) -> int:
    from ..chromatin.cli import run_freeze_evaluation_windows

    return run_freeze_evaluation_windows(
        shards_glob=args.shards,
        split=args.split,
        windows_out=args.windows_out,
        conditions_out=args.conditions_out,
        output_bins=args.output_bins,
        genomes=args.genome,
        context_length=args.context_length,
        example_ids_from=args.example_ids_from,
        max_examples_per_condition_group=args.max_examples_per_condition_group,
        sampling_seed=args.sampling_seed,
    )

def _cmd_chromatin_adapter_train(args: argparse.Namespace) -> int:
    from ..chromatin.cli import run_adapter_train

    return run_adapter_train(
        config_path=args.config,
        train_raw=args.train_raw,
        train_truth=args.train_truth,
        train_conditions=args.train_conditions,
        validation_raw=args.validation_raw,
        validation_truth=args.validation_truth,
        validation_conditions=args.validation_conditions,
        out=args.out,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        seed=args.seed,
        device=args.device,
    )

def _cmd_chromatin_adapter_predict(args: argparse.Namespace) -> int:
    from ..chromatin.cli import run_adapter_predict

    return run_adapter_predict(
        config_path=args.config,
        checkpoint_path=args.checkpoint,
        raw_path=args.raw,
        conditions_path=args.conditions,
        out=args.out,
        batch_size=args.batch_size,
        device=args.device,
        ablate_intervention_residual=args.ablate_intervention_residual,
    )

def _cmd_chromatin_merge_raw(args: argparse.Namespace) -> int:
    from ..chromatin.cli import run_merge_raw_predictions

    return run_merge_raw_predictions(inputs_glob=args.inputs,out=args.out)

def _cmd_chromatin_ensemble_seeds(args: argparse.Namespace) -> int:
    from ..chromatin.cli import run_ensemble_seed_predictions

    return run_ensemble_seed_predictions(
        inputs_glob=args.inputs,
        campaign_path=args.campaign,
        out=args.out,
    )

def _cmd_chromatin_audit_compiled_splits(args: argparse.Namespace) -> int:
    from ..chromatin.cli import run_audit_compiled_splits

    return run_audit_compiled_splits(shards_glob=args.shards,out=args.out)

def _cmd_chromatin_compile(args: argparse.Namespace) -> int:
    from ..chromatin.cli import run_compile

    return run_compile(
        config_path=args.config,
        track_spec_path=args.track_spec,
        output_dir=args.output,
        fasta_path=args.fasta,
        stride=args.stride,
        max_windows=args.max_windows,
        windows_per_shard=args.windows_per_shard,
        negative_keep_probability=args.negative_keep_probability,
    )

def _cmd_chromatin_compile_index(args: argparse.Namespace) -> int:
    from ..chromatin.cli import run_compile_index

    return run_compile_index(
        config_path=args.config,
        track_index_path=args.track_index,
        output_dir=args.output,
        max_tracks=args.max_tracks,
        stride=args.stride,
        max_windows=args.max_windows,
        windows_per_shard=args.windows_per_shard,
        negative_keep_probability=args.negative_keep_probability,
    )

def _cmd_chromatin_pair(args: argparse.Namespace) -> int:
    from ..chromatin.cli import run_pair_shards

    return run_pair_shards(
        reference_glob=args.reference,
        treatment_glob=args.treatment,
        output_dir=args.output,
        mode=args.mode,
        windows_per_shard=args.windows_per_shard,
        minimum_overlap_fraction=args.minimum_overlap_fraction,
    )

def _cmd_chromatin_materialize_intervention_pairs(args: argparse.Namespace) -> int:
    from ..chromatin.cli import run_materialize_intervention_pairs

    return run_materialize_intervention_pairs(
        pair_plan=args.pair_plan,
        compiled_root=args.compiled_root,
        output_root=args.output,
        windows_per_shard=args.windows_per_shard,
        minimum_overlap_fraction=args.minimum_overlap_fraction,
    )

def _cmd_chromatin_train(args: argparse.Namespace) -> int:
    from ..chromatin.cli import run_train

    return run_train(
        config_path=args.config,
        shards_glob=args.shards,
        checkpoint_dir=args.checkpoint_dir,
        fasta_path=args.fasta,
        device=args.device,
        resume=not args.no_resume,
        minimum_free_gb=args.min_free_vram_gb,
        allow_low_vram=args.allow_low_vram,
        stage=args.stage,
        initialize_from=args.initialize_from,
        seed=args.seed,
        minimum_replicate_quality=args.minimum_replicate_quality,
        validation_study=args.validation_study,
    )

def _cmd_chromatin_predict(args: argparse.Namespace) -> int:
    from ..chromatin.cli import run_predict

    return run_predict(
        config_path=args.config,
        checkpoint_path=args.checkpoint,
        shards_glob=args.shards,
        out=args.out,
        track_mapping_path=args.track_mapping,
        fasta_path=args.fasta,
        device=args.device,
        component=args.component,
        crop_bins=args.crop_bins,
        reverse_complement=not args.no_reverse_complement,
        minimum_free_gb=args.min_free_vram_gb,
        allow_low_vram=args.allow_low_vram,
        example_ids_from=args.example_ids_from,
        ablate_state_residual=args.ablate_state_residual,
        ablate_intervention_residual=args.ablate_intervention_residual,
        seed=args.seed,
    )

def _cmd_chromatin_circuit_audit(args: argparse.Namespace) -> int:
    from ..chromatin.cli import run_circuit_audit

    return run_circuit_audit(
        inputs_glob=args.inputs,
        registry_path=args.registry,
        out=args.out,
    )

def _cmd_chromatin_benchmark(args: argparse.Namespace) -> int:
    from ..chromatin.cli import run_benchmark

    return run_benchmark(
        candidate_root=args.candidate_root,
        baseline_root=args.baseline_root,
        registry_path=args.registry,
        out=args.out,
        bootstrap=args.bootstrap,
        candidate_seed_roots=args.candidate_seed_root,
        comparison_role=args.comparison_role,
    )

def _cmd_chromatin_benchmark_suite(args: argparse.Namespace) -> int:
    from ..chromatin.cli import run_benchmark_suite

    return run_benchmark_suite(
        headline=args.headline,
        adapter=args.adapter,
        borzoi=args.borzoi,
        registry_path=args.registry,
        out=args.out,
    )

def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="pdac",description="PDAC synthetic gene-circuit design pipeline (RUO).")
    sub = p.add_subparsers(dest="command",required=True)

    f = sub.add_parser("fetch-data",help="download real open corpora -> sha256 -> manifests")
    f.add_argument("corpus",nargs="?",default=None,help="single corpus id")
    f.add_argument("--all-open",action="store_true",help="fetch every OPEN corpus with pinned files")
    f.add_argument("--all",action="store_true",help="write a manifest for every corpus per posture")
    f.add_argument("--heavy",default=None,help="resolve/pull a heavy file for the given corpus id")
    f.add_argument("--dry-run",action="store_true")
    f.add_argument("--list",action="store_true",help="print access posture for every corpus")
    f.set_defaults(func=_cmd_fetch_data)

    v = sub.add_parser("verify-data",help="re-hash REAL artifacts; enforce honesty contract")
    v.set_defaults(func=_cmd_verify_data)

    b = sub.add_parser("bulk-fetch",help="download large real open raw data until data/raw crosses target GB")
    b.add_argument("--target-gb",type=float,default=110.0)
    b.add_argument("--workers",type=int,default=6)
    b.add_argument("--no-gnomad",action="store_true")
    b.set_defaults(func=_cmd_bulk_fetch)

    t = sub.add_parser("train",help="train the 3 ML models from scratch")
    t.add_argument("--only",default=None,help="comma list: promoter,enhancer,grna")
    t.add_argument("--all",action="store_true")
    t.add_argument("--quick",action="store_true",help="short run for smoke testing")
    t.set_defaults(func=_cmd_train)

    r = sub.add_parser("run-pipeline",help="run Module I->VI for a subtype")
    r.add_argument("--subtype",default="basal",choices=["basal","classical"])
    r.add_argument("--top-k",type=int,default=10)
    r.add_argument("--seed",type=int,default=20260620)
    r.add_argument("--out",default="results/run.json")
    r.set_defaults(func=_cmd_run_pipeline)

    rd = sub.add_parser("run-deep",help="thousands of individually-simulated distinct circuits (depth, not combinatorial)")
    rd.add_argument("--subtype",default="classical",choices=["basal","classical"])
    rd.add_argument("--max-targets",type=int,default=None,help="cap candidate TFs (default: all)")
    rd.add_argument("--multi-top",type=int,default=30,help="top TFs combined into multi-TF AND circuits")
    rd.add_argument("--sweep-n",type=int,default=24,help="robustness sweep samples per circuit")
    rd.add_argument("--seed",type=int,default=20260620)
    rd.add_argument("--out",default="results/deep.json")
    rd.set_defaults(func=_cmd_run_deep)

    ad = sub.add_parser("attractor-design",help="RAC: regulatory-attractor-control TF-circuit design (DepMap/JASPAR/chromatin)")
    ad.add_argument("--max-nodes",type=int,default=360)
    ad.add_argument("--coexpr-threshold",type=float,default=0.35)
    ad.add_argument("--no-motif",action="store_true",help="skip JASPAR motif-directed edge refinement")
    ad.add_argument("--epochs",type=int,default=2400)
    ad.add_argument("--ensemble",type=int,default=8,help="bootstrap ensemble members for collapse CIs")
    ad.add_argument("--max-targets",type=int,default=6)
    ad.add_argument("--seed",type=int,default=20260620)
    ad.set_defaults(func=_cmd_attractor_design)

    d = sub.add_parser("predeploy",help="fail-closed predeploy gates (exit 1 on any fail)")
    d.set_defaults(func=_cmd_predeploy)

    fg = sub.add_parser("figures",help="regenerate publication figures from results/")
    fg.set_defaults(func=_cmd_figures)

    ci = sub.add_parser("chromatin-inventory",help="audit the 100+ GB chromatin corpus and its PDAC gap")
    ci.add_argument("--out",default=None,help="optional JSON report path")
    ci.set_defaults(func=_cmd_chromatin_inventory)

    cm = sub.add_parser("chromatin-model-info",help="validate a long-range model profile and report its size")
    cm.add_argument("--config",default="configs/chromatin-local-12gb.json")
    cm.add_argument("--forward-check",action="store_true",help="run one real-hg38 forward pass")
    cm.add_argument("--device",default="cpu")
    cm.add_argument("--min-free-vram-gb",type=float,default=8.0)
    cm.add_argument("--allow-low-vram",action="store_true")
    cm.set_defaults(func=_cmd_chromatin_model_info)

    ce = sub.add_parser("chromatin-encode-specs",help="resolve ENCODE bigWigs into grouped assay/state TrackSpecs")
    ce.add_argument("--refresh",action="store_true",help="refresh cached ENCODE metadata")
    ce.add_argument("--limit",type=int,default=None,help="resolve only the first N tracks")
    ce.add_argument(
        "--metadata-workers",
        type=int,
        default=8,
        help="bounded parallel official-API metadata requests (1-16)",
    )
    ce.set_defaults(func=_cmd_chromatin_encode_specs)

    cet = sub.add_parser(
        "chromatin-enformer-target-map",
        help="freeze label-blind pancreas output mappings for official Enformer targets",
    )
    cet.add_argument("--refresh",action="store_true")
    cet.add_argument("--out",default=None)
    cet.add_argument(
        "--policy",
        default=None,
        help="optional human or mouse label-blind target policy",
    )
    cet.add_argument(
        "--metadata-cache",
        default=None,
        help="optional path for the corresponding official target table",
    )
    cet.set_defaults(func=_cmd_chromatin_enformer_target_map)

    cbt = sub.add_parser(
        "chromatin-borzoi-target-map",
        help="freeze label-blind pancreas output mappings from official Borzoi metadata",
    )
    cbt.add_argument("--refresh",action="store_true")
    cbt.add_argument("--out",default=None)
    cbt.set_defaults(func=_cmd_chromatin_borzoi_target_map)

    cpc = sub.add_parser(
        "chromatin-plan-campaign",
        help="materialize a frozen multi-seed, staged training DAG without starting it",
    )
    cpc.add_argument("--campaign",default="configs/chromatin-campaign.json")
    cpc.add_argument("--profile",default="configs/chromatin-local-12gb.json")
    cpc.add_argument("--out",default="results/frozen/chromatin-campaign-plan.json")
    cpc.set_defaults(func=_cmd_chromatin_plan_campaign)

    czb = sub.add_parser(
        "chromatin-zero-baseline",
        help="construct the honest state-invariant Enformer contrast for a frozen truth set",
    )
    czb.add_argument("--truth",required=True)
    czb.add_argument("--out",required=True)
    czb.add_argument("--model",default="Enformer")
    czb.add_argument("--model-version",required=True)
    czb.add_argument("--weights-sha256",required=True)
    czb.add_argument("--track-mapping-sha256",required=True)
    czb.set_defaults(func=_cmd_chromatin_zero_baseline)

    cab = sub.add_parser(
        "chromatin-assemble-bundle",
        help="join exact-ID raw predictions to frozen truth and emit provenance",
    )
    cab.add_argument("--raw",required=True)
    cab.add_argument("--truth",required=True)
    cab.add_argument("--out",required=True)
    cab.add_argument("--provenance-out",default=None)
    cab.add_argument(
        "--claim-surface-contract",
        default="configs/chromatin-claim-surfaces.json",
        help="hash-bind the bundle to the frozen biological source/grouping contract",
    )
    cab.add_argument(
        "--training-use",
        required=True,
        choices=["predictions_only","candidate_model"],
    )
    cab.set_defaults(func=_cmd_chromatin_assemble_bundle)

    ccr = sub.add_parser(
        "chromatin-contrast-raw",
        help="subtract exact-ID reference profiles from treatment profiles before truth joins",
    )
    ccr.add_argument("--reference-raw",required=True)
    ccr.add_argument("--treatment-raw",required=True)
    ccr.add_argument("--out",required=True)
    ccr.add_argument("--mode",choices=["state","perturbation"],required=True)
    ccr.set_defaults(func=_cmd_chromatin_contrast_raw)

    ccf = sub.add_parser(
        "chromatin-conformalize",
        help="add donor/study-block split-conformal intervals without reading target truth",
    )
    ccf.add_argument("--calibration-raw",required=True)
    ccf.add_argument("--calibration-truth",required=True)
    ccf.add_argument("--target-raw",required=True)
    ccf.add_argument("--out",required=True)
    ccf.add_argument("--nominal",type=float,default=0.90)
    ccf.set_defaults(func=_cmd_chromatin_conformalize)

    cft = sub.add_parser(
        "chromatin-freeze-profile-truth",
        help="freeze one masked profile truth surface from already split shards",
    )
    cft.add_argument("--shards",action="append",required=True)
    cft.add_argument("--split",required=True)
    cft.add_argument("--out",required=True)
    cft.add_argument("--crop-bins",type=int,default=896)
    cft.add_argument(
        "--target-field",
        choices=["target","paired_delta","perturbation_delta"],
        default="target",
    )
    cft.add_argument(
        "--mask-field",
        choices=["valid","pair_mask","perturbation_mask"],
        default="valid",
    )
    cft.add_argument("--genome",action="append",choices=["hg38","hg19","mm10","mm9"])
    cft.add_argument(
        "--example-ids-from",
        default=None,
        help="exact label-free cohort from evaluation-window JSON or conditions NPZ",
    )
    cft.set_defaults(func=_cmd_chromatin_freeze_profile_truth)

    cew = sub.add_parser(
        "chromatin-freeze-evaluation-windows",
        help="freeze label-free coordinates and condition vectors for isolated baselines",
    )
    cew.add_argument("--shards",action="append",required=True)
    cew.add_argument("--split",required=True)
    cew.add_argument("--windows-out",required=True)
    cew.add_argument("--conditions-out",required=True)
    cew.add_argument("--output-bins",type=int,default=896)
    cew.add_argument("--genome",action="append",choices=["hg38","hg19","mm10","mm9"])
    cew.add_argument(
        "--context-length",
        type=int,
        default=None,
        help="center each native shard interval in this model-specific input span",
    )
    cew.add_argument(
        "--example-ids-from",
        default=None,
        help="exact label-free cohort from a prior native-context manifest",
    )
    cew.add_argument(
        "--max-examples-per-condition-group",
        type=int,
        default=None,
        help="target-blind SHA-256 cap within biological-group and condition strata",
    )
    cew.add_argument("--sampling-seed",type=int,default=20260715)
    cew.set_defaults(func=_cmd_chromatin_freeze_evaluation_windows)

    cat = sub.add_parser(
        "chromatin-adapter-train",
        help="fit the strong grouped condition-aware adapter on frozen Enformer profiles",
    )
    cat.add_argument("--config",default="configs/enformer-state-adapter.json")
    cat.add_argument("--train-raw",required=True)
    cat.add_argument("--train-truth",required=True)
    cat.add_argument("--train-conditions",required=True)
    cat.add_argument("--validation-raw",required=True)
    cat.add_argument("--validation-truth",required=True)
    cat.add_argument("--validation-conditions",required=True)
    cat.add_argument("--out",required=True)
    cat.add_argument("--epochs",type=int,default=30)
    cat.add_argument("--batch-size",type=int,default=16)
    cat.add_argument("--learning-rate",type=float,default=2e-4)
    cat.add_argument("--weight-decay",type=float,default=1e-4)
    cat.add_argument("--seed",type=int,default=20260620)
    cat.add_argument("--device",choices=["cpu","cuda","auto"],default="cpu")
    cat.set_defaults(func=_cmd_chromatin_adapter_train)

    cap = sub.add_parser(
        "chromatin-adapter-predict",
        help="apply the frozen Enformer adapter using conditions but no truth artifact",
    )
    cap.add_argument("--config",default="configs/enformer-state-adapter.json")
    cap.add_argument("--checkpoint",required=True)
    cap.add_argument("--raw",required=True)
    cap.add_argument("--conditions",required=True)
    cap.add_argument("--out",required=True)
    cap.add_argument("--batch-size",type=int,default=32)
    cap.add_argument("--device",choices=["cpu","cuda","auto"],default="cpu")
    cap.add_argument(
        "--ablate-intervention-residual",
        action="store_true",
        help="emit the exact zero-perturbation reference for a signed adapter contrast",
    )
    cap.set_defaults(func=_cmd_chromatin_adapter_predict)

    cmr = sub.add_parser(
        "chromatin-merge-raw",
        help="merge disjoint assay-specific outputs from one frozen baseline identity",
    )
    cmr.add_argument("--inputs",required=True)
    cmr.add_argument("--out",required=True)
    cmr.set_defaults(func=_cmd_chromatin_merge_raw)

    ces = sub.add_parser(
        "chromatin-ensemble-seeds",
        help="mean exact-cohort raw predictions across every registered candidate seed",
    )
    ces.add_argument("--inputs",required=True)
    ces.add_argument("--campaign",default="configs/chromatin-campaign.json")
    ces.add_argument("--out",required=True)
    ces.set_defaults(func=_cmd_chromatin_ensemble_seeds)

    cas = sub.add_parser(
        "chromatin-audit-compiled-splits",
        help="audit group, locus, study, and exact-ID leakage in materialized shards",
    )
    cas.add_argument("--shards",required=True)
    cas.add_argument("--out",default=None)
    cas.set_defaults(func=_cmd_chromatin_audit_compiled_splits)

    csp = sub.add_parser(
        "chromatin-study-plan",
        help="resolve a registered GEO study without downloading its data",
    )
    csp.add_argument("accession",help="registered GSE accession")
    csp.add_argument("--out",default=None)
    csp.add_argument("--skip-size-probe",action="store_true")
    csp.set_defaults(func=_cmd_chromatin_study_plan)

    cfs = sub.add_parser(
        "chromatin-fetch-study",
        help="sequentially fetch a resolved GEO plan with resume and leakage guards",
    )
    cfs.add_argument("--plan",required=True)
    cfs.add_argument("--output-root",default="data/studies")
    cfs.add_argument("--allow-protected-study",action="store_true")
    cfs.add_argument(
        "--protected-seal",
        default="results/frozen/protected-studies.seal.json",
    )
    cfs.add_argument(
        "--protected-release",
        default=None,
        help="required final-checkpoint release before any protected target bytes download",
    )
    cfs.add_argument("--max-total-gb",type=float,default=25.0)
    cfs.set_defaults(func=_cmd_chromatin_fetch_study)

    cia = sub.add_parser(
        "chromatin-inspect-geo-archive",
        help="inventory a downloaded GEO tar and parse assay labels without inferring state",
    )
    cia.add_argument("--archive",required=True)
    cia.add_argument("--out",default=None)
    cia.set_defaults(func=_cmd_chromatin_inspect_geo_archive)

    cea = sub.add_parser(
        "chromatin-extract-geo-archive",
        help="safely extract a hashed GEO tar under an unpacked-size cap",
    )
    cea.add_argument("--archive",required=True)
    cea.add_argument("--output",required=True)
    cea.add_argument("--max-unpacked-gb",type=float,default=25.0)
    cea.set_defaults(func=_cmd_chromatin_extract_geo_archive)

    cgm = sub.add_parser(
        "chromatin-geo-metadata",
        help="cache and strictly resolve depositor GEO SOFT sample metadata",
    )
    cgm.add_argument("accession",help="registered GSE accession")
    cgm.add_argument("--refresh",action="store_true")
    cgm.add_argument("--allow-protected-metadata",action="store_true")
    cgm.add_argument(
        "--protected-release",
        default="results/frozen/protected-studies.release.json",
    )
    cgm.set_defaults(func=_cmd_chromatin_geo_metadata)

    cpss = sub.add_parser(
        "chromatin-seal-protected-studies",
        help="freeze protected study identity before any download or metadata access",
    )
    cpss.add_argument("--campaign",default="configs/chromatin-campaign.json")
    cpss.add_argument("--registry",default="chromatin_registry.json")
    cpss.add_argument("--assets",default="pdac_chromatin_assets.json")
    cpss.add_argument("--out",default="results/frozen/protected-studies.seal.json")
    cpss.set_defaults(func=_cmd_chromatin_seal_protected)

    cpam = sub.add_parser(
        "chromatin-authorize-protected-metadata",
        help="release protected labels only after all final campaign seeds are frozen",
    )
    cpam.add_argument(
        "--seal",default="results/frozen/protected-studies.seal.json"
    )
    cpam.add_argument("--checkpoint",action="append",required=True)
    cpam.add_argument(
        "--out",default="results/frozen/protected-studies.release.json"
    )
    cpam.set_defaults(func=_cmd_chromatin_authorize_protected)

    cgts = sub.add_parser(
        "chromatin-geo-track-specs",
        help="join extracted GEO bigWigs to authoritative metadata and native-genome specs",
    )
    cgts.add_argument("accession",help="registered GSE accession")
    cgts.add_argument("--extracted",required=True)
    cgts.add_argument("--metadata",default=None)
    cgts.add_argument(
        "--evaluation-only",
        action="store_true",
        help="emit isolated external-study TrackSpecs after the protected release",
    )
    cgts.add_argument(
        "--protected-release",
        default=None,
        help="required final-checkpoint release for protected evaluation TrackSpecs",
    )
    cgts.set_defaults(func=_cmd_chromatin_geo_track_specs)

    cfr = sub.add_parser(
        "chromatin-fetch-reference",
        help="materialize a pinned native reference FASTA with checksum and index",
    )
    cfr.add_argument("genome",choices=["mm9","mm10","hg19"])
    cfr.add_argument("--discard-compressed",action="store_true")
    cfr.set_defaults(func=_cmd_chromatin_fetch_reference)

    cipp = sub.add_parser(
        "chromatin-intervention-pair-plan",
        help="register exact control/intervention TrackSpec pairs without inventing state pairs",
    )
    cipp.add_argument("--track-index",required=True)
    cipp.add_argument("--out",default=None)
    cipp.set_defaults(func=_cmd_chromatin_intervention_pair_plan)

    cepp = sub.add_parser(
        "chromatin-external-perturbation-plan",
        help="freeze exact external KLF5 0h-to-4h pairs with replicate nesting",
    )
    cepp.add_argument(
        "--track-index",
        action="append",
        required=True,
        help="repeat in frozen GSE301272, GSE301284, GSE295354 order",
    )
    cepp.add_argument(
        "--contract",default="configs/chromatin-claim-surfaces.json"
    )
    cepp.add_argument(
        "--out",default="data/pair_specs/external_KLF5.intervention.json"
    )
    cepp.add_argument(
        "--merged-index-out",
        default="data/evaluation_track_specs/external_KLF5/index.json",
    )
    cepp.set_defaults(func=_cmd_chromatin_external_perturbation_pair_plan)

    cc = sub.add_parser("chromatin-compile",help="stream one bigWig into bounded-memory coordinate shards")
    cc.add_argument("--config",default="configs/chromatin-local-12gb.json")
    cc.add_argument("--track-spec",required=True,help="JSON TrackSpec with provenance and condition vectors")
    cc.add_argument("--output",default="data/processed/chromatin_shards")
    cc.add_argument("--fasta",default=None)
    cc.add_argument("--stride",type=int,default=None)
    cc.add_argument("--max-windows",type=int,default=None)
    cc.add_argument("--windows-per-shard",type=int,default=64)
    cc.add_argument("--negative-keep-probability",type=float,default=0.05)
    cc.set_defaults(func=_cmd_chromatin_compile)

    cci = sub.add_parser(
        "chromatin-compile-index",
        help="sequentially compile and hash-verify every TrackSpec in an index",
    )
    cci.add_argument("--config",default="configs/chromatin-local-12gb.json")
    cci.add_argument("--track-index",required=True)
    cci.add_argument("--output",required=True)
    cci.add_argument("--max-tracks",type=int,default=None)
    cci.add_argument("--stride",type=int,default=None)
    cci.add_argument("--max-windows",type=int,default=None)
    cci.add_argument("--windows-per-shard",type=int,default=64)
    cci.add_argument("--negative-keep-probability",type=float,default=0.05)
    cci.set_defaults(func=_cmd_chromatin_compile_index)

    cpair = sub.add_parser(
        "chromatin-pair",
        help="merge registered normal/disease or control/intervention shards by coordinate",
    )
    cpair.add_argument("--reference",required=True,help="glob for normal/control shards")
    cpair.add_argument("--treatment",required=True,help="glob for PDAC/intervention shards")
    cpair.add_argument("--output",required=True)
    cpair.add_argument("--mode",required=True,choices=["state","perturbation"])
    cpair.add_argument("--windows-per-shard",type=int,default=64)
    cpair.add_argument("--minimum-overlap-fraction",type=float,default=0.80)
    cpair.set_defaults(func=_cmd_chromatin_pair)

    cmip = sub.add_parser(
        "chromatin-materialize-intervention-pairs",
        help="execute and certify every exact registered control/intervention pair",
    )
    cmip.add_argument("--pair-plan",required=True)
    cmip.add_argument("--compiled-root",required=True)
    cmip.add_argument("--output",required=True)
    cmip.add_argument("--windows-per-shard",type=int,default=64)
    cmip.add_argument("--minimum-overlap-fraction",type=float,default=0.80)
    cmip.set_defaults(func=_cmd_chromatin_materialize_intervention_pairs)

    ct = sub.add_parser("chromatin-train",help="resume memory-bounded long-range chromatin training")
    ct.add_argument("--config",default="configs/chromatin-local-12gb.json")
    ct.add_argument(
        "--shards",
        required=True,
        action="append",
        help="compiled .npz glob; repeat for a frozen multi-study stage",
    )
    ct.add_argument("--checkpoint-dir",default="models/chromatin/checkpoints")
    ct.add_argument("--fasta",default=None)
    ct.add_argument("--device",default=None,help="override config: cpu, cuda, or auto")
    ct.add_argument("--no-resume",action="store_true")
    ct.add_argument("--min-free-vram-gb",type=float,default=8.0)
    ct.add_argument("--allow-low-vram",action="store_true")
    ct.add_argument(
        "--stage",
        choices=[
            "healthy_prior",
            "progression_state_residual",
            "signed_intervention_residual",
            "human_state_adaptation",
        ],
        default="human_state_adaptation",
    )
    ct.add_argument(
        "--initialize-from",
        default=None,
        help="load model weights only from a completed prior-stage checkpoint",
    )
    ct.add_argument(
        "--seed",
        type=int,
        default=None,
        help="override the frozen profile seed for an explicitly registered campaign run",
    )
    ct.add_argument(
        "--minimum-replicate-quality",
        type=float,
        default=None,
        help="filter complete track manifests by the frozen assay quality channel before hashing",
    )
    ct.add_argument(
        "--validation-study",
        action="append",
        default=None,
        help=(
            "restrict checkpoint selection to a validation-only study; repeatable and excluded "
            "from gradients (registered for human-state adaptation)"
        ),
    )
    ct.set_defaults(func=_cmd_chromatin_train)

    cp = sub.add_parser(
        "chromatin-predict",
        help="export checkpoint-locked candidate predictions without truth labels",
    )
    cp.add_argument("--config",required=True)
    cp.add_argument("--checkpoint",required=True)
    cp.add_argument("--shards",required=True)
    cp.add_argument("--out",required=True)
    cp.add_argument("--track-mapping",default="chromatin_registry.json")
    cp.add_argument("--fasta",default=None)
    cp.add_argument("--device",default="auto")
    cp.add_argument(
        "--seed",
        type=int,
        default=None,
        help="select the exact registered campaign-seed checkpoint configuration",
    )
    cp.add_argument(
        "--component",
        default="mean",
        choices=[
            "mean",
            "baseline",
            "state_residual",
            "perturbation_residual",
            "residual",
            "circuit_factors",
            "intervention_factors",
            "intervention_axis_potentials",
        ],
    )
    cp.add_argument("--crop-bins",type=int,default=896)
    cp.add_argument("--no-reverse-complement",action="store_true")
    cp.add_argument("--min-free-vram-gb",type=float,default=8.0)
    cp.add_argument("--allow-low-vram",action="store_true")
    cp.add_argument(
        "--ablate-state-residual",
        action="store_true",
        help="counterfactually remove the learned PDAC-state residual at inference",
    )
    cp.add_argument(
        "--ablate-intervention-residual",
        action="store_true",
        help="counterfactually remove the signed intervention residual at inference",
    )
    cp.add_argument(
        "--example-ids-from",
        default=None,
        help="restrict inference to an exact label-free evaluation cohort",
    )
    cp.set_defaults(func=_cmd_chromatin_predict)

    cca = sub.add_parser(
        "chromatin-circuit-audit",
        help="gate latent circuit interpretation on rotation-invariant multi-seed stability",
    )
    cca.add_argument(
        "--inputs",
        required=True,
        help="glob for raw circuit_factors or intervention_factors prediction bundles",
    )
    cca.add_argument("--registry",default="chromatin_registry.json")
    cca.add_argument(
        "--out",default="results/frozen/chromatin-circuit-stability-audit.json"
    )
    cca.set_defaults(func=_cmd_chromatin_circuit_audit)

    cb = sub.add_parser("chromatin-benchmark",help="paired independent-group comparison with frozen Enformer")
    cb.add_argument(
        "--candidate-root",
        required=True,
        help="directory containing <axis>.npz and <axis>.provenance.json candidate pairs",
    )
    cb.add_argument(
        "--baseline-root",
        required=True,
        help="directory containing <axis>.npz and <axis>.provenance.json baseline pairs",
    )
    cb.add_argument("--registry",default="chromatin_registry.json")
    cb.add_argument("--out",default="results/chromatin_enformer_benchmark.json")
    cb.add_argument("--bootstrap",type=int,default=10000)
    cb.add_argument(
        "--comparison-role",
        choices=[
            "headline_enformer",
            "diagnostic_enformer_adapter",
            "secondary_borzoi",
        ],
        default="headline_enformer",
    )
    cb.add_argument(
        "--candidate-seed-root",
        action="append",
        default=[],
        help="repeat once per registered seed; each root contains per-axis bundles/provenance",
    )
    cb.set_defaults(func=_cmd_chromatin_benchmark)

    cbs = sub.add_parser(
        "chromatin-benchmark-suite",
        help="gate the official Enformer claim on the condition-aware adapter and identity equality",
    )
    cbs.add_argument("--headline",required=True)
    cbs.add_argument("--adapter",required=True)
    cbs.add_argument("--borzoi",required=True)
    cbs.add_argument("--registry",default="chromatin_registry.json")
    cbs.add_argument(
        "--out",default="results/chromatin_enformer_claim_suite.json"
    )
    cbs.set_defaults(func=_cmd_chromatin_benchmark_suite)

    return p

def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv if argv is not None else sys.argv[1:])
    return int(args.func(args) or 0)

if __name__ == "__main__":
    raise SystemExit(main())
