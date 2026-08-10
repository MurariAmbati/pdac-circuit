from __future__ import annotations

import json
from dataclasses import replace

import numpy as np

from pdac_circuit.circuit import golden
from pdac_circuit.circuit.boolean import BooleanModel
from pdac_circuit.circuit.ode import ODEModel
from pdac_circuit.circuit.stability import _latin_hypercube, parameter_sweep, steady_state_within_tol
from pdac_circuit.core.paths import RESULTS

OUT=RESULTS / "robustness_metric_diagnostic.json"
N=256
SEED=20260620

def dead_circuit(base):
    c=base
    genes=[replace(g, beta=0.0) for g in c.to_ode().genes]
    return ODEModel(nodes=list(c.to_ode().nodes), genes=genes)

def per_gene_sweep(base_ode, boolean_ref, n=N, rel=0.5, seed=SEED, require_correct=True):
    rng=np.random.default_rng(seed)
    ngene=len(base_ode.genes)
    d=4 * ngene
    lhs=_latin_hypercube(n, d, rng)
    lo, hi = np.log(1.0 / (1.0 + rel)), np.log(1.0 + rel)
    fac=np.exp(lo + lhs * (hi - lo)).reshape(n, ngene, 4)

    ok_settle=0
    ok_correct=0
    for draw in fac:
        genes=[
            replace(g, beta=g.beta * f[0], gamma=g.gamma * f[1], K=g.K * f[2],
                    n=max(0.5, g.n * f[3]))
            for g, f in zip(base_ode.genes, draw)
        ]
        m=ODEModel(nodes=list(base_ode.nodes), genes=genes)
        try:
            settled=steady_state_within_tol(m)
        except Exception:
            settled=False
        ok_settle += int(settled)
        if not require_correct or boolean_ref is None:
            continue
        if not settled:
            continue
        try:
            sol=m.simulate(t_span=(0.0, 80.0), n_points=600)
            ss=sol["x"][:, -1]
            ceil=np.array([max(g.beta / max(g.gamma, 1e-9), 1e-9) for g in m.genes])
            hi_lo=tuple(bool(v) for v in (ss > 0.5 * ceil))
            ok_correct += int(hi_lo in boolean_ref)
        except Exception:
            pass
    return {"settle_fraction": ok_settle / n,
            "correct_fraction": (ok_correct / n) if require_correct else None}

def boolean_fixed_points(circuit):
    try:
        fps=BooleanModel.from_circuit(circuit).fixed_points()
        return {tuple(bool(v) for v in s) for s in fps} or None
    except Exception:
        return None

def main():
    cases={
        "robust_circuit (golden)": golden.robust_circuit(),
        "fragile_circuit (golden)": golden.fragile_circuit(),
        "monostable_circuit (golden)": golden.monostable_circuit(),
        "toggle_switch (golden)": golden.toggle_switch(),
        "repressilator (oscillates: must fail)": golden.repressilator(),
    }
    rows=[]
    for name, circ in cases.items():
        cur=parameter_sweep(circ, n=N, seed=SEED)
        ref=boolean_fixed_points(circ)
        fixed=per_gene_sweep(circ.to_ode(), ref, n=N, seed=SEED)
        dead=dead_circuit(circ)
        dead_cur=0
        rng=np.random.default_rng(SEED)
        lhs=_latin_hypercube(N, 4, rng)
        lo, hi = np.log(1 / 1.5), np.log(1.5)
        gf=np.exp(lo + lhs * (hi - lo))
        for row in gf:
            genes=[replace(g, beta=g.beta * row[0], gamma=g.gamma * row[1],
                             K=g.K * row[2], n=max(0.5, g.n * row[3])) for g in dead.genes]
            try:
                dead_cur += int(steady_state_within_tol(ODEModel(nodes=list(dead.nodes), genes=genes)))
            except Exception:
                pass
        rows.append({
            "circuit": name,
            "current_metric_global_jitter_settle_only": round(float(cur["robustness"]), 4),
            "dead_version_same_metric": round(dead_cur / N, 4),
            "per_gene_lhs_settle_only": round(float(fixed["settle_fraction"]), 4),
            "per_gene_lhs_boolean_correct": (None if fixed["correct_fraction"] is None
                                             else round(float(fixed["correct_fraction"]), 4)),
            "boolean_reference_resolved": ref is not None,
        })
        r=rows[-1]
        print(f"  {name:38} current={r['current_metric_global_jitter_settle_only']:.3f}  "
              f"dead={r['dead_version_same_metric']:.3f}  "
              f"per-gene={r['per_gene_lhs_settle_only']:.3f}  "
              f"correct={r['per_gene_lhs_boolean_correct']}", flush=True)

    cur_vals=[r["current_metric_global_jitter_settle_only"] for r in rows]
    dead_vals=[r["dead_version_same_metric"] for r in rows]
    pg_vals=[r["per_gene_lhs_settle_only"] for r in rows]
    separates=bool(max(cur_vals) - min(cur_vals) >= 0.10)
    dead_passes=bool(max(dead_vals) >= 0.95)
    verdict={
        "current_metric_range": [min(cur_vals), max(cur_vals)],
        "current_metric_spread": round(max(cur_vals) - min(cur_vals), 4),
        "separates_fragile_from_robust": separates,
        "dead_circuits_score_max": round(float(max(dead_vals)), 4),
        "dead_circuits_also_pass": dead_passes,
        "per_gene_metric_spread": round(max(pg_vals) - min(pg_vals), 4),
    }
    parts=[]
    parts.append(
        f"the sweep DOES detect instability (spread {verdict['current_metric_spread']}: it fails "
        f"oscillators and fragile designs), so global-vs-per-gene jitter is not the problem"
        if separates else
        f"the sweep does not separate the fragile and robust fixtures (spread "
        f"{verdict['current_metric_spread']})")
    if dead_passes:
        parts.append(
            f"but a DEAD circuit -- every gene's output drive set to zero -- scores "
            f"{verdict['dead_circuits_score_max']:.3f}, the maximum: the criterion asks only whether "
            f"the system settles, and doing nothing settles perfectly. robustness=1.0 is therefore "
            f"not evidence that a circuit works, and every circuit the pipeline actually delivers "
            f"scores exactly 1.0, so the objective contributes no ranking information to NSGA-II")
    verdict["conclusion"]="; ".join(parts)
    rep={"schema": "pdac-circuit.robustness-diagnostic/1", "data_class": "REAL",
           "sealed_studies_touched": False, "n_draws": N, "seed": SEED,
           "design": ("negative controls (dead circuits) and the repository's own fragile/robust "
                      "golden fixtures are scored under the shipped metric (one global "
                      "multiplicative factor per axis, settle-only criterion) and under a "
                      "corrected one (independent per-gene LHS, agreement with the circuit's "
                      "compiled Boolean fixed point)"),
           "per_circuit": rows, "verdict": verdict}
    OUT.write_text(json.dumps(rep, indent=2), encoding="utf-8")
    print(f"\nVERDICT: {verdict['conclusion']}")
    print(f"  current spread {verdict['current_metric_spread']}  |  per-gene spread "
          f"{verdict['per_gene_metric_spread']}  |  dead circuits pass: {verdict['dead_circuits_also_pass']}")
    print(f"wrote {OUT}")

if __name__ == "__main__":
    main()
