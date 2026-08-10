from __future__ import annotations

import json
import re

import numpy as np
import pandas as pd

from pdac_circuit.core.paths import DEPMAP_CRISPR, RESULTS

OUT = RESULTS / "subtype_resolved_targets.json"
EXPR = r"C:\Users\murar\aurora-research\discoveries\data\OmicsExpression.csv"

def _sym(c):
    m = re.match(r"^(.*?)\s*\(\d+\)$", c)
    return (m.group(1) if m else c).strip()

MARGIN = 0.20

def _assign(b, c, margin=MARGIN):
    if max(b, c) <= 0:
        return "intermediate"
    if abs(b - c) < margin:
        return "ambiguous"
    return "basal" if b > c else "classical"

def subtype_score_lines():
    from pdac_circuit.data.misc import _pdac_model_ids
    from pdac_circuit.data.tf import subtype_signature_genes

    sig = subtype_signature_genes()
    want = {g for v in sig.values() for g in v}
    header = pd.read_csv(EXPR, nrows=0).columns.tolist()
    idc = header[0]
    s2c = {}
    for c in header[1:]:
        s2c.setdefault(_sym(c), c)
    use = [idc] + [s2c[g] for g in sorted(want & set(s2c))]
    df = pd.read_csv(EXPR, usecols=use, index_col=0)
    df.columns = [_sym(c) for c in df.columns]
    pdac = set(_pdac_model_ids())
    df = df[df.index.isin(pdac)]
    z = (df - df.mean(0)) / df.std(0).replace(0, 1)
    out = {}
    for name, genes in sig.items():
        cols = [g for g in genes if g in z.columns]
        out[name] = z[cols].mean(axis=1) if cols else pd.Series(0.0, index=z.index)
    scores = pd.DataFrame(out)
    scores["assignment"] = [_assign(b, c) for b, c in zip(scores["basal"], scores["classical"])]
    return scores

def panc1_subtype():
    from pdac_circuit.data.tf import subtype_signature_genes

    sig = subtype_signature_genes()
    want = {g for v in sig.values() for g in v}
    header = pd.read_csv(EXPR, nrows=0).columns.tolist()
    idc = header[0]
    s2c = {}
    for c in header[1:]:
        s2c.setdefault(_sym(c), c)
    use = [idc] + [s2c[g] for g in sorted(want & set(s2c))]
    df = pd.read_csv(EXPR, usecols=use, index_col=0)
    df.columns = [_sym(c) for c in df.columns]
    try:
        mdl = pd.read_csv(RESULTS.parent / "data" / "raw" / "depmap-crispr" / "Model.csv", index_col=0)
        name_col = next((c for c in mdl.columns if c.lower() in ("celllinename", "strippedcelllinename")), None)
        hits = mdl.index[mdl[name_col].astype(str).str.upper().str.replace("-", "") == "PANC1"] if name_col else []
    except Exception:
        hits = []
    if len(hits) == 0 or hits[0] not in df.index:
        return {"found": False, "note": "PANC-1 not resolvable in the DepMap expression panel"}
    z = (df - df.mean(0)) / df.std(0).replace(0, 1)
    row = z.loc[hits[0]]
    b = float(row[[g for g in sig["basal"] if g in z.columns]].mean())
    c = float(row[[g for g in sig["classical"] if g in z.columns]].mean())
    a = _assign(b, c)
    note = ("PANC-1 sits BELOW the panel mean on both programmes, so an argmax call would invent a "
            "subtype it does not express. Held out as intermediate. This is consistent with the "
            "reviewer's quasimesenchymal description, and it means PANC-1 is not a faithful "
            "substrate for either a basal or a classical target set."
            if a == "intermediate" else
            f"PANC-1 is called {a} on this signature.")
    return {"found": True, "depmap_id": str(hits[0]), "basal_score": round(b, 4),
            "classical_score": round(c, 4), "argmax_would_say": "basal" if b > c else "classical",
            "assignment": a, "note": note}

def main():
    from scipy.stats import mannwhitneyu

    gated = json.loads((RESULTS / "gated_target_ranking.json").read_text())
    cands = [r["gene"] for r in gated["ranked_candidates"]]
    quar = [r["gene"] for r in gated["quarantined"]]
    genes = cands + quar
    scores = subtype_score_lines()
    basal = list(scores.index[scores["assignment"] == "basal"])
    classical = list(scores.index[scores["assignment"] == "classical"])
    held = {k: int((scores["assignment"] == k).sum()) for k in ("intermediate", "ambiguous")}
    print(f"DepMap PDAC lines: {len(basal)} basal / {len(classical)} classical | "
          f"held out {held['intermediate']} intermediate + {held['ambiguous']} ambiguous", flush=True)

    header = pd.read_csv(DEPMAP_CRISPR, nrows=0).columns.tolist()
    idc = header[0]
    s2c = {}
    for c in header[1:]:
        s2c.setdefault(_sym(c), c)
    cols = {g: s2c[g] for g in genes if g in s2c}
    ge = pd.read_csv(DEPMAP_CRISPR, usecols=[idc] + list(cols.values()), index_col=0)
    ge.columns = [_sym(c) for c in ge.columns]

    rows = []
    for g in genes:
        if g not in ge.columns:
            continue
        b = ge[g].reindex(basal).dropna()
        c = ge[g].reindex(classical).dropna()
        if len(b) < 3 or len(c) < 3:
            continue
        u, p = mannwhitneyu(b, c, alternative="two-sided")
        diff = float(b.mean() - c.mean())
        rows.append({
            "gene": g,
            "in_gated_candidates": g in cands,
            "chronos_basal_mean": round(float(b.mean()), 4), "n_basal": len(b),
            "chronos_classical_mean": round(float(c.mean()), 4), "n_classical": len(c),
            "basal_minus_classical": round(diff, 4),
            "mannwhitney_p": round(float(p), 5),
            "more_essential_in": ("basal" if diff < 0 else "classical") if p < 0.05 else "neither (n.s.)",
        })
    rows.sort(key=lambda r: r["mannwhitney_p"])

    m = len(rows)
    ps = np.array([r["mannwhitney_p"] for r in rows])
    order = np.argsort(ps)
    bh = np.empty(m)
    bh[order] = np.minimum.accumulate((ps[order] * m / np.arange(1, m + 1))[::-1])[::-1]
    for r, q in zip(rows, np.clip(bh, 0, 1)):
        r["bh_q"] = round(float(q), 5)
        r["subtype_differential"] = bool(q < 0.05)
        if not r["subtype_differential"]:
            r["more_essential_in"] = "neither (n.s. after BH)"

    sens = []
    for mg in (0.0, 0.1, 0.2, 0.3, 0.4):
        asg = [_assign(b, c, mg) for b, c in zip(scores["basal"], scores["classical"])]
        s = pd.Series(asg, index=scores.index)
        bl, cl = list(s.index[s == "basal"]), list(s.index[s == "classical"])
        row = {"margin": mg, "n_basal": len(bl), "n_classical": len(cl)}
        for g in cands:
            if g not in ge.columns:
                continue
            b, c = ge[g].reindex(bl).dropna(), ge[g].reindex(cl).dropna()
            row[g] = round(float(mannwhitneyu(b, c, alternative="two-sided")[1]), 4) \
                if len(b) >= 3 and len(c) >= 3 else None
        sens.append(row)

    p1 = panc1_subtype()

    rep = {
        "schema": "pdac-circuit.subtype-resolved/1", "data_class": "REAL",
        "sealed_studies_touched": False,
        "method": ("Moffitt basal/classical signature (25+25 genes), mean z-scored expression across "
                   "DepMap PDAC lines. A line is called only if it is above the panel mean on the "
                   "winning programme and the two scores differ by > 0.20; otherwise it is held out "
                   "as intermediate/ambiguous rather than forced by argmax. Per-gene differential "
                   "Chronos by Mann-Whitney, Benjamini-Hochberg across the 8 genes tested. Dynamics "
                   "are not refit; attractor-collapse carries no weight (claim retracted)."),
        "line_split": {"n_basal": len(basal), "n_classical": len(classical),
                       "n_held_out": held, "margin": MARGIN},
        "panc1_substrate_check": p1,
        "margin_sensitivity": sens,
        "per_gene": rows,
    }
    OUT.write_text(json.dumps(rep, indent=2), encoding="utf-8")

    print("\n=== PANC-1 substrate check ===")
    print(f"  basal {p1.get('basal_score')} vs classical {p1.get('classical_score')} "
          f"-> {p1.get('assignment')} (argmax would have said {p1.get('argmax_would_say')})")
    print(f"  {p1.get('note')}")
    print("\n=== subtype-differential CRISPR effect ===")
    for r in rows:
        tag = "CANDIDATE" if r["in_gated_candidates"] else "quarantined"
        print(f"  {r['gene']:9} {tag:11} basal {r['chronos_basal_mean']:+.3f} vs classical "
              f"{r['chronos_classical_mean']:+.3f}  diff {r['basal_minus_classical']:+.3f}  "
              f"p={r['mannwhitney_p']:.4f} q={r['bh_q']:.4f}  -> {r['more_essential_in']}")
    print("\n=== margin sensitivity (p for each candidate) ===")
    for s in sens:
        cs = "  ".join(f"{g}={s[g]:.3f}" for g in cands if s.get(g) is not None)
        print(f"  margin {s['margin']:.1f}: {s['n_basal']:2}b/{s['n_classical']:2}c   {cs}")
    print(f"\nwrote {OUT}")

if __name__ == "__main__":
    main()
