from __future__ import annotations

import json

from ..core.audit import AuditChain
from ..core.contract import OutputEnvelope,Verdict
from ..core.governance import guard_emission
from ..core.paths import MODELS
from ..core.provenance import REAL

def _safe(fn,default=None):
    try:
        return fn()
    except Exception as e:
        return ("__error__",f"{type(e).__name__}: {e}",default)

def _build_circuit(tf: str):
    from ..circuit import Circuit,assess

    c = Circuit()
    c.add_gene("TF",basal=0.4,degradation=1.0)
    c.add_gene("SynProm",basal=0.05,degradation=1.0)
    c.add_gene("Repressor",basal=0.05,degradation=1.0)
    c.add_edge("TF","SynProm",+1)
    c.add_edge("SynProm","Repressor",+1)
    c.add_edge("Repressor","TF",-1)
    return c,assess(c)

def _enhancer_candidates(tf: str,atac,h3k,enh_model,max_cand: int = 12):
    from ..data.genes import gene_locus
    from ..data.reference import fetch_sequence
    from ..parts.select import score_enhancers

    g = gene_locus(tf)
    if g is None:
        return None
    tss = g["tss"]
    near = [iv for iv in atac.overlaps(g["chrom"],tss - 50000,tss + 50000)
            if h3k.any_overlap(iv.chrom,iv.start,iv.end)]
    if not near:
        return None
    near = sorted(near,key=lambda iv: -iv.signal)[:max_cand]
    seqs = [fetch_sequence(iv.chrom,(iv.start + iv.end) // 2 - 1000,(iv.start + iv.end) // 2 + 1000) for iv in near]
    scores = score_enhancers(enh_model,seqs)
    best = max(range(len(scores)),key=lambda i: scores[i]["activity"])
    return {"activity": scores[best]["activity"],"signal": scores[best]["signal"],
            "locus": {"chrom": near[best].chrom,"start": near[best].start,"end": near[best].end}}

def _synthetic_promoter_library(prom_model,*,top: int = 8,n_gen: int = 200,seed: int = 20260620) -> list[dict]:
    from ..generate.promoter_gan import PromoterGAN
    from ..parts.select import score_promoters

    p = MODELS / "promoter_gan.pt"
    if not p.exists() or prom_model is None:
        return []
    try:
        gan = PromoterGAN.load(p)
        seqs = gan.generate(n_gen,seed=seed)
        scores = score_promoters(prom_model,seqs)
        ranked = sorted(
            ({"strength": s["strength"],"conformal": s["conformal"],"seq": seqs[i]} for i,s in enumerate(scores)),
            key=lambda d: -d["strength"],
        )
        return ranked[:top]
    except Exception:
        return []

def _optimize_promoter_seq(seq: str) -> dict:
    from ..seqopt import OptConfig,gc_content,optimize_circuit
    from ..seqopt.types import CircuitSeq,Feature

    cs = CircuitSeq(seq=seq,features=[Feature("promoter",0,len(seq),1,"promoter")])
    gc_before = gc_content(seq)
    try:
        opt,report = optimize_circuit(cs,OptConfig.from_registry())
        return {"gc_before": float(gc_before),"gc_after": float(gc_content(opt.seq)),
                "n_edits": len(opt.edit_log),"unsatisfied": list(report.unsatisfied)}
    except Exception as e:
        return {"gc_before": float(gc_before),"error": f"{type(e).__name__}: {e}"}

def run_pipeline(*,subtype: str = "basal",top_k: int = 10,only_targets=None,
                 seed: int = 20260620,out: str = "results/run.json") -> int:
    from ..data.tracks import load_atac_peaks,load_h3k27ac_peaks
    from ..grna.efficiency_model import GRNAModel
    from ..parts.repressor import select_repressor
    from ..parts.select import load_enhancer_model,load_promoter_model,score_promoters
    from ..scoring import CircuitScore,build_subscores,pareto_rank,score_circuits
    from ..scoring.immuno import immunogenicity_risk
    from ..targeting import build_afm,prioritize_targets
    from ..data.genes import promoter_window
    from ..data.reference import fetch_sequence

    audit = AuditChain()
    caveats: list[str] = []
    data_classes = [REAL]

    afm = build_afm()
    afm_out,env1 = prioritize_targets(top_k=top_k,afm=afm,seed=seed)
    audit.add("targeting",inputs=["lambert-tf","tcga-paad","gtex-pancreas","intogen-pdac"],payload=env1.payload["numbers"])
    caveats += afm_out.caveats
    targets = env1.payload["top_per_subtype"].get(subtype,[])[:top_k]
    if only_targets:
        requested = list(dict.fromkeys(only_targets))
        targets = [t for t in requested if t in afm_out.table.index]
        missing = [t for t in requested if t not in afm_out.table.index]
        if missing:
            caveats.append(
                f"{len(missing)} requested target(s) absent from the Module I feature matrix: "
                f"{', '.join(missing[:8])}"
            )
        if not targets:
            return _finish(OutputEnvelope.abstain("no requested target is present in the AFM"),
                           out,subtype,seed,audit,caveats)
    if env1.verdict != Verdict.OK:
        caveats.append(f"Module I did not certify-real (cert={env1.cert}); proceeding with diagnostic target list.")

    prom_model = load_promoter_model()
    enh_model = load_enhancer_model()
    grna_model = GRNAModel.load(MODELS / "grna_ontarget.pt") if (MODELS / "grna_ontarget.pt").exists() else None
    if prom_model is None or enh_model is None or grna_model is None:
        return _finish(OutputEnvelope.abstain("trained models missing; run `pdac train --all`"),out,subtype,seed,audit,caveats)
    atac = load_atac_peaks()
    h3k = load_h3k27ac_peaks()

    synth_library = _synthetic_promoter_library(prom_model,top=8,seed=seed)
    if synth_library:
        caveats.append(f"Module VII: {len(synth_library)} de-novo GAN promoters available; best generated strength={synth_library[0]['strength']:.2f}.")

    circuit_scores: list[CircuitScore] = []
    details: list[dict] = []
    for tf in targets:
        row = afm_out.table.loc[tf] if tf in afm_out.table.index else None
        pw = promoter_window(tf)
        if pw is None:
            caveats.append(f"{tf}: no GENCODE TSS; skipped.")
            continue
        core = promoter_window(tf,up=500,down=500)
        prom_seq = fetch_sequence(core["chrom"],core["start"],core["end"],core["strand"])
        prom = score_promoters(prom_model,[prom_seq])[0]
        prom["source"] = "native"
        prom["seq"] = prom_seq
        if synth_library and synth_library[0]["strength"] > prom["strength"]:
            best = synth_library[0]
            prom = {"strength": best["strength"],"conformal": best["conformal"],
                    "source": "gan-generated","seq": best["seq"]}
        enh = _enhancer_candidates(tf,atac,h3k,enh_model)
        if enh is None:
            caveats.append(f"{tf}: no active enhancer (ATAC∩H3K27ac) near TSS; skipped.")
            continue
        rep_env = select_repressor(tf,model=grna_model,atac_index=atac,top_k=3)
        if rep_env.verdict != Verdict.OK:
            caveats.append(f"{tf}: repressor design abstained ({rep_env.caveats[0] if rep_env.caveats else '?'}).")
            continue
        guide = rep_env.payload["guides"][0]
        circ,assessment = _build_circuit(tf)
        part_seq = prom.get("seq",prom_seq)
        seqopt = _optimize_promoter_seq(part_seq)
        immuno = immunogenicity_risk(dna_seq=part_seq[:300])

        subtype_r = float(abs(row["basal_r"] if subtype == "basal" else row["classical_r"])) if row is not None else 0.5
        sub = build_subscores(
            promoter_strength=prom["strength"],
            enhancer_activity=enh["activity"],
            grna_on_target=guide["on_target"] or 0.0,
            subtype_expr_likelihood=subtype_r,
            chromatin_overlap=1.0 if guide["in_open_chromatin"] else 0.3,
            p_correct_under_perturbation=assessment["robustness"],
            off_target_risk=guide["off_risk"],
            immuno_risk=immuno.risk,
            integration_risk=0.1,
        )
        seq_ok = not seqopt.get("unsatisfied")
        cs = CircuitScore(circuit_id=tf,sub=sub,composite=0.0,pareto_rank=-1,crowding=0.0,acceptable=seq_ok,dominated_by=[])
        circuit_scores.append(cs)
        details.append({
            "target_tf": tf,"subtype": subtype,
            "promoter": {"strength": prom["strength"],"conformal": prom["conformal"],
                         "source": prom.get("source","native")},
            "acceptable": seq_ok,
            "enhancer": enh,
            "repressor_guide": guide,
            "circuit": {"cert": assessment["cert"],"robustness": assessment["robustness"],
                        "steady_state_ok": assessment["steady_state_ok"],"stability_ok": assessment["stability_ok"]},
            "sequence_optimization": seqopt,
            "immunogenicity": {"risk": immuno.risk,"interval": [immuno.lo,immuno.hi],"caveat": immuno.caveat},
        })

    if not circuit_scores:
        return _finish(OutputEnvelope.abstain(f"no viable circuit assembled for subtype={subtype}",cert="certified-negative"),
                       out,subtype,seed,audit,caveats)

    vi_env = score_circuits(circuit_scores,top_k=min(top_k,len(circuit_scores)),seed=seed,data_classes=data_classes)
    audit.add("scoring",inputs=[d["target_tf"] for d in details],payload={"n": len(circuit_scores)})

    ranked = {c.circuit_id: c for c in pareto_rank(circuit_scores)}
    for d in details:
        c = ranked.get(d["target_tf"])
        if c is not None:
            d["sub_scores"] = {"efficacy": c.sub.efficacy,"specificity": c.sub.specificity,
                               "robustness": c.sub.robustness,"safety": c.sub.safety}
            d["pareto_rank"] = c.pareto_rank
            d["composite"] = c.composite
            d["acceptable"] = c.acceptable
    details.sort(key=lambda d: (d.get("pareto_rank",99),-d.get("composite",0)))

    payload = {
        "subtype": subtype,
        "module_I": env1.payload["numbers"],
        "ranked_circuits": details,
        "module_VI": vi_env.payload if vi_env.verdict == Verdict.OK else {"verdict": vi_env.verdict.value,"caveats": vi_env.caveats},
    }
    env = OutputEnvelope.ok(payload,data_classes=data_classes,cert="real",caveats=caveats)
    return _finish(env,out,subtype,seed,audit,caveats)

def _finish(env: OutputEnvelope,out: str,subtype: str,seed: int,audit: AuditChain,caveats: list[str]) -> int:
    from ..core.paths import ROOT

    env.audit = audit.to_dict()
    env.caveats = list(dict.fromkeys(list(env.caveats) + caveats))
    guarded = guard_emission(env,rendered_text=json.dumps(env.payload or {})[:5000])
    artifact = {
        "schema": "pdac-circuit.run/1",
        "ruo_banner": env.ruo_banner,
        "subtype": subtype,
        "seed": seed,
        **guarded.to_dict(),
    }
    out_path = (ROOT / out) if not str(out).startswith(("/","C:","c:")) else out
    from pathlib import Path

    p = Path(out_path)
    p.parent.mkdir(parents=True,exist_ok=True)
    p.write_text(json.dumps(artifact,indent=2,default=float),encoding="utf-8")
    n = len((guarded.payload or {}).get("ranked_circuits",[]))
    print(f"[pipeline] subtype={subtype} verdict={guarded.verdict.value} cert={guarded.cert} circuits={n} -> {out}")
    return 0 if guarded.verdict != Verdict.REFUSE else 1
