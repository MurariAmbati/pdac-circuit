from __future__ import annotations

import gzip
import json
import sys
import time
from itertools import combinations
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import numpy as np

from pdac_circuit.core.paths import MODELS, RESULTS
from pdac_circuit.core.seeds import set_seeds
from pdac_circuit.data.genes import crispri_window, gene_locus, promoter_window
from pdac_circuit.data.reference import fetch_sequence
from pdac_circuit.data.tracks import load_atac_peaks, load_h3k27ac_peaks
from pdac_circuit.grna.design import predict_on_target
from pdac_circuit.grna.efficiency_model import GRNAModel
from pdac_circuit.grna.genome_offtarget import genome_wide_offtargets
from pdac_circuit.grna.offtarget import score_guide_offtargets
from pdac_circuit.grna.scan import enumerate_protospacers
from pdac_circuit.parts.select import (
    load_enhancer_model,
    load_promoter_model,
    score_enhancers,
    score_promoters,
)
from pdac_circuit.pipeline.deep import _build_deep_circuit, _norm, _simulate_deep
from pdac_circuit.scoring import CircuitScore, build_subscores, pareto_rank
from pdac_circuit.scoring.immuno import immunogenicity_risk
from pdac_circuit.scoring.objectives import composite as composite_score
from pdac_circuit.scoring.pareto import _load_floors
from pdac_circuit.signal import chromatin_state
from pdac_circuit.signal.chromatin import activity_unit
from pdac_circuit.targeting import build_afm, prioritize_targets

SEED = 20260620
SUBTYPE = "classical"
N_TARGETS = 48
N_ENHANCERS = 6
N_GUIDES = 4
N_PROMOTERS = 8
GAN_POOL = 4000
PAIR_TOP = 40
PAIR_LOGICS = ("AND", "OR")
SWEEP_N = 24
MAX_MM = 4
PRESHORT = 48
MIN_LOCUS_SEP = 500
CHROMS = None
TOP_KEEP = 200


def enhancer_library(tf, atac, h3k, model, k):
    g = gene_locus(tf)
    if g is None:
        return []
    tss = g["tss"]
    raw = [iv for iv in atac.overlaps(g["chrom"], tss - 50000, tss + 50000)
           if h3k.any_overlap(iv.chrom, iv.start, iv.end)]
    if not raw:
        return []
    uniq = {}
    for iv in raw:
        key = (iv.chrom, iv.start, iv.end)
        if key not in uniq or iv.signal > uniq[key].signal:
            uniq[key] = iv
    near = []
    for iv in sorted(uniq.values(), key=lambda x: -x.signal):
        mid = (iv.start + iv.end) // 2
        if all(abs(mid - (o.start + o.end) // 2) >= MIN_LOCUS_SEP for o in near):
            near.append(iv)
        if len(near) >= 12:
            break
    seqs = [fetch_sequence(iv.chrom, (iv.start + iv.end) // 2 - 1000, (iv.start + iv.end) // 2 + 1000)
            for iv in near]
    sc = score_enhancers(model, seqs)
    order = sorted(range(len(sc)), key=lambda i: -sc[i]["activity"])[:k]
    return [{"activity": float(sc[i]["activity"]), "signal": float(sc[i]["signal"]),
             "locus": {"chrom": near[i].chrom, "start": int(near[i].start), "end": int(near[i].end)}}
            for i in order]


def promoter_library(model, n_gen, k, seed):
    from pdac_circuit.generate.promoter_gan import PromoterGAN

    gan = PromoterGAN.load(MODELS / "promoter_gan.pt")
    seqs = gan.generate(n_gen, seed=seed)
    sc = score_promoters(model, seqs)
    ranked = sorted(range(len(seqs)), key=lambda i: -sc[i]["strength"])
    span = ranked[: max(k, len(ranked) // 2)]
    pick = np.unique(np.linspace(0, len(span) - 1, k).round().astype(int))
    return [{"id": f"gan{j:02d}", "source": "gan-generated",
             "strength": float(sc[span[i]]["strength"]), "conformal": sc[span[i]]["conformal"],
             "seq": seqs[span[i]]} for j, i in enumerate(pick)]


def guide_library(tf, model, atac, k):
    win = crispri_window(tf)
    if win is None:
        return []
    cands = enumerate_protospacers(win["chrom"], win["start"], win["end"])
    if not cands:
        return []
    on = predict_on_target(model, [c.context for c in cands])
    for c, s in zip(cands, on):
        c.on_target = float(s)
        c.in_open_chromatin = atac.any_overlap(c.chrom, c.start, c.end)
    cands.sort(key=lambda c: -(c.on_target or 0.0))
    short = cands[:PRESHORT]
    neigh = [(win["chrom"], fetch_sequence(win["chrom"], max(0, win["start"] - 5000),
                                          win["end"] + 5000))]
    for c in short:
        local = score_guide_offtargets(c.protospacer, neigh, max_mm=MAX_MM)
        c.cfd_specificity = local["cfd_specificity"]
    short.sort(key=lambda c: -((c.on_target or 0.0) * (c.cfd_specificity or 0.0)
                               * (1.1 if c.in_open_chromatin else 1.0)))
    return short[:k]


def main():
    t0 = time.time()
    set_seeds(SEED)

    afm_out, env1 = prioritize_targets(top_k=10, afm=build_afm(), seed=SEED)
    table = afm_out.table
    ranked_tf = list(table.sort_values("composite", ascending=False).index)
    prom_model, enh_model = load_promoter_model(), load_enhancer_model()
    grna_model = GRNAModel.load(MODELS / "grna_ontarget.pt")
    atac, h3k = load_atac_peaks(), load_h3k27ac_peaks()
    expr_lo, expr_hi = float(table["tumor_mean_log"].min()), float(table["tumor_mean_log"].max())

    promoters = promoter_library(prom_model, GAN_POOL, N_PROMOTERS, SEED)
    print(f"[{time.time()-t0:6.1f}s] GAN library {len(promoters)} of {GAN_POOL} generated, "
          f"strength {min(p['strength'] for p in promoters):.3f}–{max(p['strength'] for p in promoters):.3f}")

    parts, skipped = {}, {}
    for tf in ranked_tf:
        if len(parts) >= N_TARGETS:
            break
        pw = promoter_window(tf)
        if pw is None:
            skipped[tf] = "no promoter window"
            continue
        enh = enhancer_library(tf, atac, h3k, enh_model, N_ENHANCERS)
        if not enh:
            skipped[tf] = "no ATAC peak with H3K27ac within 50 kb of TSS"
            continue
        gds = guide_library(tf, grna_model, atac, N_GUIDES)
        if not gds:
            skipped[tf] = "no NGG protospacer in CRISPRi window"
            continue
        pseq = fetch_sequence(pw["chrom"], pw["start"], pw["end"], pw["strand"])
        native = score_promoters(prom_model, [pseq])[0]
        row = table.loc[tf]
        parts[tf] = {
            "native": {"id": "native", "source": "native", "strength": float(native["strength"]),
                       "conformal": native["conformal"], "seq": pseq},
            "enh": enh, "guides": gds,
            "beta_tf": 0.3 + 0.6 * _norm(row["tumor_mean_log"], expr_lo, expr_hi),
            "subtype_r": float(abs(row["basal_r"] if SUBTYPE == "basal" else row["classical_r"])),
            "chromatin": chromatin_state(tf),
        }
        if len(parts) % 8 == 0:
            print(f"[{time.time()-t0:6.1f}s] parts for {len(parts)} targets")

    print(f"[{time.time()-t0:6.1f}s] {len(parts)} targets with complete parts, {len(skipped)} skipped")

    protos = sorted({c.protospacer[:20].upper() for p in parts.values() for c in p["guides"]})
    print(f"[{time.time()-t0:6.1f}s] genome-wide off-target for {len(protos)} guides at <={MAX_MM} mm")
    gw = genome_wide_offtargets(protos, max_mm=MAX_MM, chroms=CHROMS)
    for p in parts.values():
        for c in p["guides"]:
            r = gw[c.protospacer[:20].upper()]
            c.cfd_specificity = r["cfd_specificity"]
            c.mit_specificity = r["mit_specificity"]
            c.off_targets = r["off_targets"]
    print(f"[{time.time()-t0:6.1f}s] off-target done, "
          f"{sum(1 for p in protos if gw[p]['n_off_targets'] == 0)} guides with zero off-targets")

    immuno = {p["id"]: float(immunogenicity_risk(dna_seq=p["seq"][:300]).risk) for p in promoters}
    for tf, p in parts.items():
        immuno[f"native:{tf}"] = float(immunogenicity_risk(dna_seq=p["native"]["seq"][:300]).risk)

    scores, records = [], {}
    n_sim = 0

    def assemble(tf, prom, pid, enh, ei, gd, gi, logic, tf2=None):
        nonlocal n_sim
        p = parts[tf]
        beta_syn = 0.4 + 1.6 * (prom["strength"] * enh["activity"])
        beta_rep = 0.3 + 1.7 * (gd.on_target or 0.0)
        extra = parts[tf2]["beta_tf"] if tf2 else None
        circ = _build_deep_circuit(p["beta_tf"], beta_syn, beta_rep,
                                   extra_tf=tf2, extra_beta_tf=extra)
        circ.gene("SynProm").promoter.logic = logic
        dyn = _simulate_deep(circ, sweep_n=SWEEP_N, seed=SEED)
        n_sim += 1
        chrom = p.get("chromatin")
        gopen = 1.0 if gd.in_open_chromatin else 0.3
        overlap = (0.5 * gopen + 0.5 * activity_unit(chrom.get("activity_score", 0.0))) if chrom else gopen
        sub = build_subscores(
            promoter_strength=prom["strength"], enhancer_activity=enh["activity"],
            grna_on_target=gd.on_target or 0.0, subtype_expr_likelihood=p["subtype_r"],
            chromatin_overlap=overlap, p_correct_under_perturbation=dyn["robustness"],
            off_target_risk=gd.off_risk, immuno_risk=immuno[pid], integration_risk=0.1,
        )
        cid = f"{tf}+{tf2}" if tf2 else tf
        cid = f"{cid}|{pid}|e{ei}|g{gi}|{logic}"
        records[cid] = {
            "id": cid, "target": tf, "partner": tf2, "logic": logic,
            "promoter": pid, "promoter_source": prom["source"],
            "promoter_strength": round(prom["strength"], 4),
            "enhancer_rank": ei, "enhancer_activity": round(enh["activity"], 4),
            "enhancer_locus": f"{enh['locus']['chrom']}:{enh['locus']['start']}-{enh['locus']['end']}",
            "guide_rank": gi, "protospacer": gd.protospacer[:20].upper(),
            "on_target": round(gd.on_target or 0.0, 4),
            "cfd_specificity": round(gd.cfd_specificity or 0.0, 4),
            "n_off_targets": len(gd.off_targets or []),
            "knockdown": round(dyn["knockdown"], 4), "stable": dyn["stable"],
            "steady_state_ok": dyn["steady_state_ok"], "robustness": round(dyn["robustness"], 4),
            "efficacy": round(sub.efficacy, 4), "specificity": round(sub.specificity, 4),
            "safety": round(sub.safety, 4), "immuno_risk": round(immuno[pid], 4),
            "chromatin_state": (chrom or {}).get("state"),
        }
        return CircuitScore(circuit_id=cid, sub=sub, composite=0.0, pareto_rank=-1,
                            crowding=0.0, acceptable=True, dominated_by=[])

    total = sum((N_PROMOTERS + 1) * len(p["enh"]) * len(p["guides"]) for p in parts.values())
    print(f"[{time.time()-t0:6.1f}s] enumerating {total} single-target circuits")
    for n, (tf, p) in enumerate(parts.items(), 1):
        lib = [(p["native"], f"native:{tf}")] + [(q, q["id"]) for q in promoters]
        for prom, pid in lib:
            for ei, enh in enumerate(p["enh"]):
                for gi, gd in enumerate(p["guides"]):
                    scores.append(assemble(tf, prom, pid, enh, ei, gd, gi, "AND"))
        if n % 8 == 0:
            print(f"[{time.time()-t0:6.1f}s] {n}/{len(parts)} targets, {len(scores)} circuits")

    best = {}
    for c in scores:
        tf = records[c.circuit_id]["target"]
        if tf not in best or c.sub.efficacy > best[tf][1]:
            best[tf] = (c.circuit_id, c.sub.efficacy)
    order = sorted(best, key=lambda t: -best[t][1])[:PAIR_TOP]
    print(f"[{time.time()-t0:6.1f}s] pairing top {len(order)} targets over {len(PAIR_LOGICS)} logics")
    for a, b in combinations(order, 2):
        pa = parts[a]
        rid = best[a][0]
        pid = records[rid]["promoter"]
        prom = pa["native"] if pid.startswith("native") else next(q for q in promoters if q["id"] == pid)
        ei, gi = records[rid]["enhancer_rank"], records[rid]["guide_rank"]
        for logic in PAIR_LOGICS:
            scores.append(assemble(a, prom, pid, pa["enh"][ei], ei, pa["guides"][gi], gi, logic, tf2=b))

    floors = _load_floors()
    for c in scores:
        c.composite = composite_score(c.sub)
    print(f"[{time.time()-t0:6.1f}s] {len(scores)} circuits simulated, ranking "
          f"(safety floor {floors['safety_floor']}, efficacy floor {floors['efficacy_floor']})")
    ranked = pareto_rank(scores, safety_floor=floors["safety_floor"])
    ranked.sort(key=lambda c: (c.pareto_rank, -c.composite))
    for c in ranked:
        records[c.circuit_id]["composite"] = round(c.composite, 4)
        records[c.circuit_id]["pareto_rank"] = c.pareto_rank
        records[c.circuit_id]["acceptable"] = bool(c.acceptable)

    front0 = [c for c in ranked if c.pareto_rank == 0]
    allrec = [records[c.circuit_id] for c in ranked]
    comp = np.array([r["composite"] for r in allrec])
    kd = np.array([r["knockdown"] for r in allrec])

    gz = RESULTS / "circuit_design_campaign_all.jsonl.gz"
    with gzip.open(gz, "wt", encoding="utf-8") as fh:
        for r in allrec:
            fh.write(json.dumps(r, separators=(",", ":")) + "\n")

    out = {
        "schema": "pdac-circuit.circuit-design-campaign/1",
        "data_class": "REAL",
        "subtype": SUBTYPE,
        "seed": SEED,
        "design_space": {
            "targets_considered": len(ranked_tf), "targets_designed": len(parts),
            "targets_skipped": len(skipped),
            "promoters_per_target": N_PROMOTERS + 1, "gan_pool": GAN_POOL,
            "enhancers_per_target": N_ENHANCERS, "guides_per_target": N_GUIDES,
            "pair_top": PAIR_TOP, "pair_logics": list(PAIR_LOGICS),
            "guide_preshortlist": PRESHORT, "min_locus_separation_bp": MIN_LOCUS_SEP,
        },
        "n_circuits": len(allrec),
        "n_single_target": sum(1 for r in allrec if r["partner"] is None),
        "n_multi_target": sum(1 for r in allrec if r["partner"] is not None),
        "n_simulated": n_sim,
        "n_pareto_front0": len(front0),
        "n_stable": sum(1 for r in allrec if r["stable"]),
        "n_acceptable": sum(1 for r in allrec if r.get("acceptable")),
        "floors": floors,
        "n_passing_floors": sum(1 for r in allrec
                                if r["safety"] >= floors["safety_floor"]
                                and r["efficacy"] >= floors["efficacy_floor"]),
        "n_guides_scanned": len(protos),
        "n_guides_zero_offtarget": sum(1 for p in protos if gw[p]["n_off_targets"] == 0),
        "composite": {"min": float(comp.min()), "median": float(np.median(comp)),
                      "max": float(comp.max()), "mean": float(comp.mean())},
        "knockdown": {"min": float(kd.min()), "median": float(np.median(kd)),
                      "max": float(kd.max())},
        "module_I": env1.payload["numbers"],
        "skipped_targets": skipped,
        "top_circuits": allrec[:TOP_KEEP],
        "all_circuits_file": gz.name,
        "runtime_s": round(time.time() - t0, 1),
    }
    (RESULTS / "circuit_design_campaign.json").write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(f"[{time.time()-t0:6.1f}s] wrote results/circuit_design_campaign.json "
          f"({len(allrec)} circuits, {len(front0)} on front 0) and {gz.name}")


if __name__ == "__main__":
    main()
