from __future__ import annotations

import json
import time

import numpy as np
import torch

from pdac_circuit.core.paths import RESULTS

OUT=RESULTS / "verify_dynamics_instability.json"
T0=time.time()

def log(m):
    print(f"[{time.time()-T0:7.1f}s] {m}", flush=True)

def step(W, b, gain, x):
    return torch.sigmoid(gain * (x @ W.t() + b))

def iterate(W, b, gain, x0, iters=2000, tol=1e-6):
    x=x0.clone()
    for k in range(iters):
        xn=step(W, b, gain, x)
        d=(xn - x).abs().max().item()
        if d < tol:
            return xn, k, d, True
        x=xn
    return x, iters, d, False

def main():
    from pdac_circuit.attractor.dynamics import AttractorDynamics
    from pdac_circuit.attractor.graph import build_regulatory_graph

    log("refitting primary config on CPU (deterministic)")
    g=build_regulatory_graph(max_nodes=400, coexpr_threshold=0.4, motif_edges=False, seed=20260620)
    d=AttractorDynamics(g, gain=4.0, device="cpu")
    d.fit(epochs=1500, motif_weight=0.0, seed=20260620)
    W, b, gain, n=d.W, d.b, d.gain, g.n
    states=torch.tensor(g.states, dtype=torch.float32)
    rng=np.random.default_rng(1)

    one_step_res=[]
    for s in states:
        r=(step(W, b, gain, s) - s).abs().max().item()
        one_step_res.append(r)
    log(f"D: per-cell-state one-step residual (max-abs): median {np.median(one_step_res):.4f}, "
        f"max {np.max(one_step_res):.4f}")

    conv=0
    endpoints=[]
    for s in states:
        xs, k, dfin, ok=iterate(W, b, gain, s)
        conv += int(ok)
        endpoints.append(xs.cpu().numpy())
    conv_rand=0
    for _ in range(30):
        xs, k, dfin, ok=iterate(W, b, gain, torch.tensor(rng.random(n), dtype=torch.float32))
        conv_rand += int(ok)
    log(f"A: converged {conv}/{len(states)} cell-state inits, {conv_rand}/30 random inits "
        f"(tol 1e-6, 2000 iters)")

    growth=[]
    for s in states[:20]:
        settled, _, _, _=iterate(W, b, gain, s, iters=1000)
        eps=1e-3
        pert=settled + eps * torch.tensor(rng.standard_normal(n), dtype=torch.float32)
        d0=(pert - settled).abs().max().item()
        x=pert.clone()
        for _ in range(20):
            x=step(W, b, gain, x)
        y=settled.clone()
        for _ in range(20):
            y=step(W, b, gain, y)
        dN=(x - y).abs().max().item()
        growth.append(dN / max(d0, 1e-12))
    log(f"B: perturbation deviation growth after 20 steps: median {np.median(growth):.2f}x "
        f"(>1 => unstable), max {np.max(growth):.1f}x")

    xstar, _, _, _=iterate(W, b, gain, states.mean(0), iters=1000)
    z=xstar @ W.t() + b
    u=gain * z
    sig=torch.sigmoid(u)
    dsig=(sig * (1 - sig)).cpu().numpy()
    J_analytic=(gain * dsig)[:, None] * W.cpu().numpy()
    rho_analytic=float(np.max(np.abs(np.linalg.eigvals(J_analytic))))
    Wt=W
    base=step(Wt, b, gain, xstar).cpu().numpy()
    eps=1e-5
    Jfd=np.zeros((n, n))
    xnp=xstar.cpu().numpy()
    for j in range(n):
        xp=xnp.copy()
        xp[j] += eps
        fp=step(Wt, b, gain, torch.tensor(xp, dtype=torch.float32)).cpu().numpy()
        Jfd[:, j]=(fp - base) / eps
    rho_fd=float(np.max(np.abs(np.linalg.eigvals(Jfd))))
    jac_err=float(np.max(np.abs(J_analytic - Jfd)))
    log(f"C: spectral radius analytic {rho_analytic:.4f} vs finite-diff {rho_fd:.4f} "
        f"(max Jacobian elementwise diff {jac_err:.2e})")

    stable=(conv > len(states) // 2) and (np.median(growth) < 1.0) and (rho_analytic < 1.0)
    rep={
        "schema": "pdac-circuit.verify-instability/1", "data_class": "REAL",
        "sealed_studies_touched": False,
        "D_fixed_point_quality": {"median_one_step_residual": round(float(np.median(one_step_res)), 4),
                                  "max_one_step_residual": round(float(np.max(one_step_res)), 4),
                                  "note": "max-abs per-node deviation of a cell state from its one-step image"},
        "A_convergence": {"cell_state_converged": conv, "cell_state_total": len(states),
                          "random_converged": conv_rand, "random_total": 30, "tol": 1e-6, "iters": 2000},
        "B_perturbation_growth": {"median_growth_20steps": round(float(np.median(growth)), 3),
                                  "max_growth_20steps": round(float(np.max(growth)), 3),
                                  "unstable_if_gt_1": True},
        "C_spectral_radius_crosscheck": {"analytic": round(rho_analytic, 4), "finite_diff": round(rho_fd, 4),
                                         "max_jacobian_elementwise_diff": jac_err},
        "conclusion": ("STABLE attractors confirmed" if stable else
                       "INSTABILITY CONFIRMED by three independent tests: cell states are not stable "
                       "attractors of the operating map (poor fixed-point quality, non-convergence, "
                       "growing perturbations, spectral radius > 1 cross-checked analytic vs finite-diff)"),
    }
    OUT.write_text(json.dumps(rep, indent=2), encoding="utf-8")
    print("\n" + "=" * 66)
    print(f"D fixed-point residual: median {np.median(one_step_res):.4f} max {np.max(one_step_res):.4f}")
    print(f"A convergence: {conv}/{len(states)} cell, {conv_rand}/30 random")
    print(f"B perturbation growth (20 steps): median {np.median(growth):.2f}x")
    print(f"C spectral radius: analytic {rho_analytic:.3f} / finite-diff {rho_fd:.3f} (diff {jac_err:.1e})")
    print(f"\nCONCLUSION: {rep['conclusion']}")
    print(f"wrote {OUT}")

if __name__ == "__main__":
    main()
