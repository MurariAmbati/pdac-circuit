from __future__ import annotations

import json

import numpy as np

from ..core.audit import AuditChain
from ..core.contract import OutputEnvelope, Verdict
from ..core.governance import guard_emission
from ..core.paths import MODELS, ROOT
from ..core.provenance import REAL

def _norm(x, lo, hi):
    return float(np.clip((x - lo) / (hi - lo + 1e-9), 0.0, 1.0))

def _build_deep_circuit(beta_tf, beta_syn, beta_rep, *, k_rep=0.5, extra_tf=None, extra_beta_tf=None):
    from ..circuit import Circuit, Promoter

    c=Circuit()
    syn_inputs=["TF"] + (["TF2"] if extra_tf else [])
    c.add_gene("TF", basal=0.1, degradation=1.0, promoter=Promoter("pTF", strength=beta_tf, inputs=[], K=k_rep))
    if extra_tf:
        c.add_gene("TF2", basal=0.1, degradation=1.0, promoter=Promoter("pTF2", strength=extra_beta_tf, inputs=[], K=k_rep))
    c.add_gene("SynProm", basal=0.03, degradation=1.0, promoter=Promoter("pSyn", strength=beta_syn, inputs=syn_inputs, logic="AND"))
    c.add_gene("Repressor", basal=0.03, degradation=1.0, promoter=Promoter("pRep", strength=beta_rep, inputs=["SynProm"]))
    c.add_edge("TF", "SynProm", +1)
    if extra_tf:
        c.add_edge("TF2", "SynProm", +1)
    c.add_edge("SynProm", "Repressor", +1)
    c.add_edge("Repressor", "TF", -1)
    return c

def _simulate_deep(circuit, *, sweep_n=24, seed=20260620):
    from ..circuit import parameter_sweep, steady_state_within_tol
    from ..circuit.ode import ODEModel

    ode=ODEModel.from_circuit(circuit)
    sim=ode.simulate(t_span=(0.0, 80.0), n_points=400)
    xss=sim["x"][:, -1]
    tf_idx=ode.index["TF"]
    settle_ok=steady_state_within_tol(ode, tol=0.10)
    stable=ode.is_stable(xss)
    tf_free=circuit.gene("TF").promoter.strength / max(circuit.gene("TF").tf.degradation, 1e-6) + circuit.gene("TF").tf.basal
    knockdown=float(np.clip(1.0 - xss[tf_idx] / (tf_free + 1e-9), 0.0, 1.0))
    sweep=parameter_sweep(circuit, n=sweep_n, rel=0.4, seed=seed)
    return {"tf_steady_state": float(xss[tf_idx]), "knockdown": knockdown,
            "steady_state_ok": bool(settle_ok), "stable": bool(stable),
            "robustness": float(sweep["robustness"])}

def run_deep_design(*, subtype="classical", targets=None, max_targets=None, multi_top=30, sweep_n=24,
                    seed=20260620, out="results/deep.json") -> int:
    from ..data.genes import crispri_window, promoter_window
    from ..data.reference import fetch_sequence
    from ..data.tracks import load_atac_peaks, load_h3k27ac_peaks
    from ..grna.efficiency_model import GRNAModel
    from ..parts.repressor import select_repressor
    from ..parts.select import load_enhancer_model, load_promoter_model, score_promoters
    from ..scoring import CircuitScore, build_subscores, pareto_rank
    from ..scoring.immuno import immunogenicity_risk
    from ..targeting import build_afm, prioritize_targets
    from .orchestrator import _enhancer_candidates, _synthetic_promoter_library

    audit=AuditChain()
    caveats: list[str]=[]

    afm=build_afm()
    afm_out, env1=prioritize_targets(top_k=10, afm=afm, seed=seed)
    audit.add("targeting", inputs=["tcga-paad", "gtex-pancreas", "intogen-pdac"], payload=env1.payload["numbers"])
    caveats += afm_out.caveats
    table=afm_out.table
    candidates=list(table.sort_values("composite", ascending=False).index)
    if targets:
        requested=list(dict.fromkeys(targets))
        candidates=[t for t in requested if t in table.index]
        missing=[t for t in requested if t not in table.index]
        if missing:
            caveats.append(
                f"{len(missing)} requested target(s) absent from the Module I feature matrix and "
                f"cannot be designed: {', '.join(missing[:8])}"
            )
        if not candidates:
            return _finish(
                OutputEnvelope.abstain("no requested target is present in the AFM"),
                out, subtype, audit, caveats, 0, 0,
            )
    if max_targets:
        candidates=candidates[:max_targets]

    prom_model=load_promoter_model()
    enh_model=load_enhancer_model()
    grna_model=GRNAModel.load(MODELS / "grna_ontarget.pt") if (MODELS / "grna_ontarget.pt").exists() else None
    if not (prom_model and enh_model and grna_model):
        return _finish(OutputEnvelope.abstain("models missing; run `pdac train --all`"), out, subtype, audit, caveats, 0, 0)
    atac, h3k=load_atac_peaks(), load_h3k27ac_peaks()

    gan_lib=_synthetic_promoter_library(prom_model, top=1, n_gen=300, seed=seed)
    gan_best=gan_lib[0]["strength"] if gan_lib else 0.0

    snp_index=None
    try:
        from ..data import snp as snpmod

        if snpmod.available():
            wins=[crispri_window(tf) for tf in candidates]
            wins=[w for w in wins if w]
            snp_index=snpmod.extract_snps_for_regions(wins)
            caveats.append(f"dbSNP: {len(snp_index)} common SNPs in {len(wins)} CRISPRi windows; guides SNP-aware.")
    except Exception as e:
        caveats.append(f"dbSNP unavailable ({type(e).__name__}); guides not SNP-checked.")

    expr_lo, expr_hi=float(table["tumor_mean_log"].min()), float(table["tumor_mean_log"].max())
    circuit_scores: list[CircuitScore]=[]
    details: dict={}
    n_simulated=0
    parts_cache: dict={}

    def _get_parts(tf):
        if tf in parts_cache:
            return parts_cache[tf]
        res=None
        pw=promoter_window(tf)
        if pw is not None:
            prom_seq=fetch_sequence(pw["chrom"], pw["start"], pw["end"], pw["strand"])
            native=score_promoters(prom_model, [prom_seq])[0]["strength"]
            enh=_enhancer_candidates(tf, atac, h3k, enh_model)
            if enh is not None:
                rep=select_repressor(tf, model=grna_model, atac_index=atac, snp_index=snp_index, top_k=1)
                if rep.verdict == Verdict.OK:
                    guide=rep.payload["guides"][0]
                    row=table.loc[tf]
                    from ..signal import chromatin_state

                    res={
                        "prom_strength": max(native, gan_best),
                        "prom_source": "gan-generated" if gan_best > native else "native",
                        "enh": enh, "guide": guide,
                        "beta_tf": 0.3 + 0.6 * _norm(row["tumor_mean_log"], expr_lo, expr_hi),
                        "subtype_r": float(abs(row["basal_r"] if subtype == "basal" else row["classical_r"])),
                        "immuno": immunogenicity_risk(dna_seq=prom_seq[:300]).risk,
                        "chromatin": chromatin_state(tf),
                    }
        parts_cache[tf]=res
        return res

    def _assemble(tf, extra_tf=None):
        nonlocal n_simulated
        pa=_get_parts(tf)
        if pa is None:
            return None
        pb=_get_parts(extra_tf) if extra_tf else None
        if extra_tf and pb is None:
            return None
        beta_syn=0.4 + 1.6 * (pa["prom_strength"] * pa["enh"]["activity"])
        beta_rep=0.3 + 1.7 * (pa["guide"]["on_target"] or 0.0)
        circ=_build_deep_circuit(pa["beta_tf"], beta_syn, beta_rep,
                                   extra_tf=extra_tf, extra_beta_tf=pb["beta_tf"] if pb else None)
        dyn=_simulate_deep(circ, sweep_n=sweep_n, seed=seed)
        n_simulated += 1
        from ..signal.chromatin import activity_unit

        chrom=pa.get("chromatin")
        guide_open=1.0 if pa["guide"]["in_open_chromatin"] else 0.3
        if chrom:
            chrom_overlap=float(0.5 * guide_open + 0.5 * activity_unit(chrom.get("activity_score", 0.0)))
        else:
            chrom_overlap=guide_open
        sub=build_subscores(
            promoter_strength=pa["prom_strength"], enhancer_activity=pa["enh"]["activity"],
            grna_on_target=pa["guide"]["on_target"] or 0.0, subtype_expr_likelihood=pa["subtype_r"],
            chromatin_overlap=chrom_overlap,
            p_correct_under_perturbation=dyn["robustness"], off_target_risk=pa["guide"]["off_risk"],
            immuno_risk=pa["immuno"], integration_risk=0.1,
        )
        cid=tf if not extra_tf else f"{tf}+{extra_tf}"
        details[cid]={"target": cid, "logic": "neg-feedback" if not extra_tf else "AND(TF1,TF2)->repress",
                        "promoter_source": pa["prom_source"],
                        "beta": {"TF": round(pa["beta_tf"], 2), "SynProm": round(beta_syn, 2), "Repressor": round(beta_rep, 2)},
                        "dynamics": dyn,
                        "chromatin": {k: chrom[k] for k in ("state", "activity_score", "bivalency", "ctcf_occupancy")} if chrom else None,
                        "guide": {"on_target": pa["guide"]["on_target"], "off_risk": pa["guide"]["off_risk"],
                                  "snp": pa["guide"].get("overlaps_common_snp")}}
        return CircuitScore(circuit_id=cid, sub=sub, composite=0.0, pareto_rank=-1, crowding=0.0, acceptable=True, dominated_by=[])

    for tf in candidates:
        cs=_assemble(tf)
        if cs is not None:
            circuit_scores.append(cs)

    leaders=[tf for tf in candidates if tf in parts_cache and parts_cache[tf] is not None][:multi_top]
    for i, a in enumerate(leaders):
        for b in leaders[i + 1 :]:
            cs=_assemble(a, extra_tf=b)
            if cs is not None:
                circuit_scores.append(cs)

    audit.add("simulate", inputs=[subtype], payload={"n_simulated": n_simulated})
    if not circuit_scores:
        return _finish(OutputEnvelope.abstain(f"no circuit simulated for {subtype}"), out, subtype, audit, caveats, 0, 0)

    ranked=pareto_rank(circuit_scores)
    ranked.sort(key=lambda c: (c.pareto_rank, -c.composite))
    front0=[c for c in ranked if c.pareto_rank == 0]
    payload={
        "subtype": subtype,
        "n_circuits": len(circuit_scores),
        "n_individually_simulated": n_simulated,
        "n_single_tf": sum(1 for c in circuit_scores if "+" not in c.circuit_id),
        "n_multi_tf": sum(1 for c in circuit_scores if "+" in c.circuit_id),
        "n_pareto_front0": len(front0),
        "module_I": env1.payload["numbers"],
        "top_circuits": [
            {"circuit": c.circuit_id, "logic": details[c.circuit_id]["logic"], "pareto_rank": c.pareto_rank,
             "composite": round(c.composite, 4), "efficacy": round(c.sub.efficacy, 3),
             "specificity": round(c.sub.specificity, 3), "robustness": round(c.sub.robustness, 3),
             "safety": round(c.sub.safety, 3),
             "tf_knockdown": round(details[c.circuit_id]["dynamics"]["knockdown"], 3),
             "stable": details[c.circuit_id]["dynamics"]["stable"],
             "promoter_source": details[c.circuit_id]["promoter_source"],
             "chromatin_state": (details[c.circuit_id].get("chromatin") or {}).get("state")}
            for c in ranked[:40]
        ],
    }
    env=OutputEnvelope.ok(payload, data_classes=[REAL], cert="real", caveats=caveats)
    return _finish(env, out, subtype, audit, caveats, len(circuit_scores), n_simulated)

def _finish(env, out, subtype, audit, caveats, n, n_sim) -> int:
    env.audit=audit.to_dict()
    env.caveats=list(dict.fromkeys(list(env.caveats) + caveats))
    guarded=guard_emission(env, rendered_text=json.dumps(env.payload or {})[:5000])
    art={"schema": "pdac-circuit.deep-run/1", "ruo_banner": env.ruo_banner, "subtype": subtype, **guarded.to_dict()}
    from pathlib import Path

    p=Path(out) if str(out)[1:3] == ":\\" or str(out).startswith("/") else ROOT / out
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(art, indent=2, default=float), encoding="utf-8")
    print(f"[deep] subtype={subtype} verdict={guarded.verdict.value} circuits={n} individually-simulated={n_sim} "
          f"front0={(guarded.payload or {}).get('n_pareto_front0','?')} -> {out}")
    return 0 if guarded.verdict != Verdict.REFUSE else 1
