from __future__ import annotations

import json
import time

import numpy as np
import torch

from pdac_circuit.core.paths import RESULTS

OUT=RESULTS / "dynamics_characterization.json"
PRIMARY_NODES, PRIMARY_TAU, SEED=400, 0.4, 20260620
T0=time.time()

def log(m):
    print(f"[{time.time()-T0:7.1f}s] {m}", flush=True)

def settle(W, b, gain, x0, iters=500, tol=1e-7):
    x=x0.clone()
    for _ in range(iters):
        xn=torch.sigmoid(gain * (x @ W.t() + b))
        if (xn - x).abs().max() < tol:
            return xn, True
        x=xn
    return x, False

def spectral_radius(W, b, gain, xstar):
    z=(xstar @ W.t() + b)
    sig=torch.sigmoid(gain * z)
    dsig=(sig * (1 - sig)).cpu().numpy()
    J=(gain * dsig)[:, None] * W.cpu().numpy()
    ev=np.linalg.eigvals(J)
    return float(np.max(np.abs(ev)))

def cluster_fixed_points(points, tol=0.05):
    reps, sizes, labels=[], [], []
    for p in points:
        placed=False
        for k, r in enumerate(reps):
            if np.max(np.abs(p - r)) < tol:
                labels.append(k)
                sizes[k] += 1
                placed=True
                break
        if not placed:
            reps.append(p)
            sizes.append(1)
            labels.append(len(reps) - 1)
    return reps, sizes, labels

def main():
    from pdac_circuit.attractor.dynamics import AttractorDynamics
    from pdac_circuit.attractor.graph import build_regulatory_graph

    log("building graph + fitting (CPU, reproduces §15 fit)")
    g=build_regulatory_graph(max_nodes=PRIMARY_NODES, coexpr_threshold=PRIMARY_TAU, motif_edges=False, seed=SEED)
    d=AttractorDynamics(g, gain=4.0, device="cpu")
    fr=d.fit(epochs=1500, motif_weight=0.0, seed=SEED)
    W, b, gain=d.W, d.b, d.gain
    n=g.n
    rng=np.random.default_rng(SEED)
    log(f"fit done: fp_err={fr.fixed_point_error:.5f} dead_activation={fr.dead_activation:.4f}")

    states=torch.tensor(g.states, dtype=torch.float32)
    mean_state=states.mean(0)

    inits={"mean_cell_state": mean_state}
    for c in (0.0, 0.02, 0.05, 0.2, 0.5, 0.8, 0.95):
        inits[f"const_{c}"]=torch.full((n,), float(c))
    endpoints, tags, converged=[], [], []
    for tag, x0 in inits.items():
        xs, ok=settle(W, b, gain, x0)
        endpoints.append(xs.cpu().numpy())
        tags.append(tag)
        converged.append(ok)
    for i, s in enumerate(states):
        xs, ok=settle(W, b, gain, s)
        endpoints.append(xs.cpu().numpy())
        tags.append(f"cell_{i}")
        converged.append(ok)
    for j in range(60):
        xs, ok=settle(W, b, gain, torch.tensor(rng.random(n), dtype=torch.float32))
        endpoints.append(xs.cpu().numpy())
        tags.append(f"rand_{j}")
        converged.append(ok)

    reps, sizes, labels=cluster_fixed_points(endpoints, tol=0.05)
    rep_means=[float(r.mean()) for r in reps]
    log(f"distinct fixed points: {len(reps)}; basin sizes {sizes}; means "
        f"{[round(m,3) for m in rep_means]}")

    viable_k=int(np.argmax(rep_means))
    dead_k=int(np.argmin(rep_means))
    viable_fp=torch.tensor(reps[viable_k], dtype=torch.float32)
    dead_fp=torch.tensor(reps[dead_k], dtype=torch.float32)
    dead_is_distinct=bool(np.max(np.abs(reps[viable_k] - reps[dead_k])) > 0.1)

    low_init_dest={}
    for c in (0.02, 0.05):
        xs, _=settle(W, b, gain, torch.full((n,), float(c)))
        m=float(xs.mean())
        to_viable=float(np.max(np.abs(xs.cpu().numpy() - reps[viable_k]))) < 0.1
        low_init_dest[f"const_{c}"]={"settled_mean": round(m, 4),
                                       "flows_to": "viable" if to_viable else ("dead" if dead_is_distinct else "other")}

    rho_viable=spectral_radius(W, b, gain, viable_fp)
    rho_dead=spectral_radius(W, b, gain, dead_fp) if dead_is_distinct else None
    log(f"spectral radius: viable rho={rho_viable:.4f}"
        + (f", dead rho={rho_dead:.4f}" if rho_dead is not None else " (no distinct dead FP)"))

    gains=[round(x, 2) for x in np.linspace(0.5, 8.0, 26)]
    hi0=torch.full((n,), 0.95)
    lo0=torch.full((n,), 0.05)
    sweep=[]
    for gg in gains:
        gt=torch.tensor(gg)
        hi, _=settle(W, b, gt, hi0)
        lo, _=settle(W, b, gt, lo0)
        hm, lm=float(hi.mean()), float(lo.mean())
        sweep.append({"gain": gg, "from_high_mean": round(hm, 4), "from_low_mean": round(lm, 4),
                      "hysteresis_gap": round(hm - lm, 4)})
    bistable_gains=[s["gain"] for s in sweep if s["hysteresis_gap"] > 0.05]
    g_crit=min(bistable_gains) if bistable_gains else None
    log(f"bistable (gap>0.05) at gains: {bistable_gains[:1]}..{bistable_gains[-1:]}; "
        f"g*={g_crit}; operating gain 4.0 margin={None if g_crit is None else round(4.0-g_crit,2)}")

    collapse=d.collapse_scores(per_line=True)
    order_c=np.argsort(-collapse)
    probe_nodes=list(order_c[:5]) + list(order_c[-5:])
    base=viable_fp
    dead_ref=reps[dead_k]
    node_probe=[]
    for i in probe_nodes:
        x=base.clone()
        for _ in range(500):
            xn=torch.sigmoid(gain * (x @ W.t() + b))
            xn[i]=0.02
            if (xn - x).abs().max() < 1e-7:
                break
            x=xn
        settled=x.cpu().numpy()
        dist_viable=float(np.max(np.abs(settled - reps[viable_k])))
        dist_dead=float(np.max(np.abs(settled - dead_ref)))
        node_probe.append({
            "node": g.nodes[i], "collapse_score": round(float(collapse[i]), 4),
            "settled_mean": round(float(settled.mean()), 4),
            "viable_mean": round(float(reps[viable_k].mean()), 4),
            "maxdist_to_viable": round(dist_viable, 4),
            "maxdist_to_dead": round(dist_dead, 4),
            "closer_to": "dead_basin" if dist_dead < dist_viable else "perturbed_viable",
        })
    n_reach_dead=sum(p["closer_to"] == "dead_basin" for p in node_probe)

    verdict=_verdict(len(reps), dead_is_distinct, rho_viable, rho_dead, g_crit, n_reach_dead, len(node_probe))
    rep={
        "schema": "pdac-circuit.dynamics-characterization/1", "data_class": "REAL",
        "sealed_studies_touched": False,
        "config": {"nodes": PRIMARY_NODES, "tau": PRIMARY_TAU, "seed": SEED, "gain": 4.0,
                   "device": "cpu", "fixed_point_error": round(fr.fixed_point_error, 5),
                   "dead_activation_at_fit": round(fr.dead_activation, 4)},
        "probe1_fixed_points": {
            "n_distinct": len(reps), "basin_sizes": sizes,
            "means": [round(m, 4) for m in rep_means],
            "viable_mean": round(rep_means[viable_k], 4),
            "dead_mean": round(rep_means[dead_k], 4),
            "dead_is_distinct_from_viable": dead_is_distinct},
        "probe2_low_init_destination": low_init_dest,
        "probe3_spectral_radius": {"viable": round(rho_viable, 4),
                                   "dead": None if rho_dead is None else round(rho_dead, 4),
                                   "both_stable": bool(rho_viable < 1 and (rho_dead is None or rho_dead < 1))},
        "probe4_bifurcation": {"critical_gain": g_crit,
                               "operating_gain": 4.0,
                               "margin_above_critical": None if g_crit is None else round(4.0 - g_crit, 3),
                               "sweep": sweep},
        "probe5_collapse_mechanism": {
            "n_probed": len(node_probe), "n_reaching_dead_basin": n_reach_dead,
            "interpretation": ("collapse measures a basin transition to the dead attractor"
                               if n_reach_dead > len(node_probe) // 2 else
                               "collapse measures local downstream suppression at a perturbed viable "
                               "fixed point, NOT a transition to the dead attractor"),
            "per_node": node_probe},
        "verdict": verdict,
    }
    OUT.write_text(json.dumps(rep, indent=2), encoding="utf-8")

    print("\n" + "=" * 70)
    print(f"distinct fixed points: {len(reps)} | basin sizes {sizes} | means {[round(m,3) for m in rep_means]}")
    print(f"dead distinct from viable: {dead_is_distinct} | low inits -> {low_init_dest}")
    print(f"spectral radius viable {rho_viable:.4f}" + (f" dead {rho_dead:.4f}" if rho_dead else ""))
    print(f"bifurcation g* = {g_crit} | operating gain 4.0 | margin {None if g_crit is None else round(4.0-g_crit,2)}")
    print(f"collapse reaches dead basin for {n_reach_dead}/{len(node_probe)} probed nodes")
    print(f"\nVERDICT: {verdict}")
    print(f"wrote {OUT}")

def _verdict(n_fp, dead_distinct, rho_v, rho_d, g_crit, n_dead, n_probe):
    parts=[f"{n_fp} distinct fixed points",
             "distinct dead attractor exists" if dead_distinct else "NO distinct dead attractor (system is effectively monostable)",
             f"viable rho={rho_v:.3f} ({'stable' if rho_v < 1 else 'UNSTABLE'})",
             (f"dead rho={rho_d:.3f}" if rho_d is not None else "no dead FP to test"),
             (f"bifurcation g*={g_crit}, operating 4.0 {'above' if (g_crit and 4.0>g_crit) else 'BELOW/at'} it" if g_crit else "no bifurcation in [0.5,8]: contractive throughout"),
             f"clamping reaches dead basin for {n_dead}/{n_probe} nodes"]
    if not dead_distinct or (g_crit is None) or (g_crit and 4.0 <= g_crit):
        head="BISTABILITY NOT DEMONSTRATED: the dead attractor is imposed at fit time but is not a stable basin the operating dynamics reach"
    elif n_dead <= n_probe // 2:
        head="BISTABLE, BUT COLLAPSE != BASIN TRANSITION: the map is bistable, yet the essentiality score measures local suppression, not travel to the dead attractor"
    else:
        head="BISTABLE AND USED: distinct stable dead attractor, operating gain above the bifurcation, and clamping essential nodes reaches the dead basin"
    return head + " | " + "; ".join(parts)

if __name__ == "__main__":
    main()
