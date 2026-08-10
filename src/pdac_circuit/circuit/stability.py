from __future__ import annotations

import json
from dataclasses import dataclass, replace
from typing import TYPE_CHECKING

import numpy as np

from ..core.paths import REGISTRY_JSON
from ..stats import classify_cert, wilson_ci
from .ode import ODEModel

if TYPE_CHECKING:
    from .topology import Circuit

def _module_iii_prereg() -> dict:
    blob = json.loads(REGISTRY_JSON.read_text(encoding="utf-8"))
    return blob["prereg"]["module_III"]

_PRE = _module_iii_prereg()
STEADY_STATE_TOL: float = float(_PRE["steady_state_tol"])
ROBUSTNESS_MIN: float = float(_PRE["robustness_min"])
SWEEP_N: int = int(_PRE["sweep_n"])

def steady_state_within_tol(
    model: ODEModel,
    tol: float = STEADY_STATE_TOL,
    window_frac: float = 0.2,
    t_span: tuple[float, float] = (0.0, 80.0),
    x0: np.ndarray | None = None,
    n_points: int = 600,
) -> bool:
    sol = model.simulate(t_span=t_span, x0=x0, n_points=n_points)
    x = sol["x"]
    n = x.shape[1]
    w = max(2, int(round(window_frac * n)))
    tail = x[:, -w:]
    means = tail.mean(axis=1)
    floor = np.maximum(np.abs(means), 1e-3)
    dev = np.max(np.abs(tail - means[:, None]), axis=1)
    return bool(np.all(dev <= tol * floor))

@dataclass
class _SweepParams:

    beta: float
    gamma: float
    K: float
    n: float

def _perturbed_model(base: ODEModel, factors: _SweepParams) -> ODEModel:
    new_genes = []
    for g in base.genes:
        new_genes.append(
            replace(
                g,
                beta=g.beta * factors.beta,
                gamma=g.gamma * factors.gamma,
                K=g.K * factors.K,
                n=max(0.5, g.n * factors.n),
            )
        )
    return ODEModel(nodes=list(base.nodes), genes=new_genes)

def _latin_hypercube(n: int, d: int, rng: np.random.Generator) -> np.ndarray:
    samples = np.empty((n, d), dtype=float)
    cut = np.linspace(0.0, 1.0, n + 1)
    u = (cut[:-1] + cut[1:]) / 2.0
    for j in range(d):
        samples[:, j] = rng.permutation(u)
    return samples

def parameter_sweep(
    circuit: "Circuit",
    n: int = SWEEP_N,
    rel: float = 0.5,
    seed: int = 20260620,
    tol: float = STEADY_STATE_TOL,
) -> dict:
    base = circuit.to_ode()
    rng = np.random.default_rng(seed)
    lhs = _latin_hypercube(n, 4, rng)
    lo, hi = np.log(1.0 / (1.0 + rel)), np.log(1.0 + rel)
    log_factors = lo + lhs * (hi - lo)
    factors = np.exp(log_factors)

    successes = 0
    for row in factors:
        fac = _SweepParams(beta=row[0], gamma=row[1], K=row[2], n=row[3])
        model = _perturbed_model(base, fac)
        try:
            ok = steady_state_within_tol(model, tol=tol)
        except Exception:
            ok = False
        successes += int(ok)

    ci = wilson_ci(successes, n, level=0.95)
    return {
        "robustness": successes / n if n else float("nan"),
        "successes": int(successes),
        "n": int(n),
        "wilson_ci": ci,
        "seed": int(seed),
        "rel": float(rel),
    }

def noise_robustness(
    model: ODEModel,
    sigma: float = 0.05,
    reps: int = 16,
    t_end: float = 60.0,
    dt: float = 0.01,
    burn_frac: float = 0.5,
    seed: int = 20260620,
    x0: np.ndarray | None = None,
) -> dict:
    rng = np.random.default_rng(seed)
    x0 = model._default_x0() if x0 is None else np.asarray(x0, dtype=float)
    n_steps = int(round(t_end / dt))
    burn = int(round(burn_frac * n_steps))
    sqrt_dt = np.sqrt(dt)
    cvs: list[float] = []
    for _ in range(reps):
        x = x0.copy()
        traj = np.empty((n_steps - burn, model.dim), dtype=float)
        for k in range(n_steps):
            drift = model.rhs(0.0, x)
            x = x + drift * dt + sigma * sqrt_dt * rng.standard_normal(model.dim)
            x = np.clip(x, 0.0, None)
            if k >= burn:
                traj[k - burn] = x
        mean = traj.mean(axis=0)
        std = traj.std(axis=0)
        cv = std / np.maximum(np.abs(mean), 1e-6)
        cvs.append(float(np.mean(cv)))
    return {
        "mean_cv": float(np.mean(cvs)),
        "max_cv": float(np.max(cvs)),
        "sigma": float(sigma),
        "reps": int(reps),
    }

def _randomize_wiring(circuit: "Circuit", rng: np.random.Generator) -> "Circuit":
    from .topology import Circuit

    new = Circuit(name=circuit.name + "_rand")
    new.enhancers = dict(circuit.enhancers)
    for node in circuit.nodes:
        gene = circuit.gene(node)
        new.graph.add_node(node, gene=gene)
    for u, v, d in circuit.graph.edges(data=True):
        sign = int(rng.choice([-1, 1]))
        new.add_edge(u, v, sign=sign)
    return new

def _viability_passrate(
    circuit: "Circuit",
    *,
    tol: float = STEADY_STATE_TOL,
    sweep_n: int = 32,
    rel: float = 0.5,
    seed: int = 20260620,
) -> float:
    try:
        res = parameter_sweep(circuit, n=sweep_n, rel=rel, seed=seed, tol=tol)
        return float(res["robustness"])
    except Exception:
        return 0.0

def viability_perm_null(
    circuit: "Circuit",
    B: int = 200,
    seed: int = 20260620,
    tol: float = STEADY_STATE_TOL,
    sweep_n: int = 32,
    rel: float = 0.5,
) -> dict:
    obs = _viability_passrate(circuit, tol=tol, sweep_n=sweep_n, rel=rel, seed=seed)
    rng = np.random.default_rng(seed)
    null = np.empty(B, dtype=float)
    for b in range(B):
        null[b] = _viability_passrate(
            _randomize_wiring(circuit, rng), tol=tol, sweep_n=sweep_n, rel=rel, seed=seed
        )
    ge = int(np.sum(null >= obs))
    p = (1 + ge) / (B + 1)
    return {
        "obs_passrate": float(obs),
        "null_mean": float(null.mean()),
        "null_ge_frac": float(np.mean(null >= obs)),
        "p": float(p),
        "p_floor": 1.0 / (B + 1),
        "B": int(B),
    }

def assess(
    circuit: "Circuit",
    *,
    sweep_n: int = SWEEP_N,
    rel: float = 0.5,
    perm_B: int = 200,
    seed: int = 20260620,
    tol: float = STEADY_STATE_TOL,
    robustness_min: float = ROBUSTNESS_MIN,
    alpha: float = 0.05,
) -> dict:
    circuit.validate()
    model = circuit.to_ode()

    steady_ok = steady_state_within_tol(model, tol=tol)
    x_ss = model.steady_state()
    stable = model.is_stable(x_ss)
    knife = model.is_knife_edge(x_ss)

    sweep = parameter_sweep(circuit, n=sweep_n, rel=rel, seed=seed, tol=tol)
    robustness = sweep["robustness"]

    perm = viability_perm_null(circuit, B=perm_B, seed=seed, tol=tol)
    perm_p = perm["p"]

    n_edges = circuit.graph.number_of_edges()
    has_cycle = bool(_has_directed_cycle(circuit))
    perm_powered = bool(has_cycle and (2 ** n_edges) >= int(round(1.0 / alpha)))

    descriptive_only = bool(knife)
    beats_random = (perm_p < alpha) if perm_powered else True
    positive_significant = bool(steady_ok and stable and beats_random)
    exceeds_margin = bool(robustness >= robustness_min)
    cert = classify_cert(
        gate_ok=steady_ok,
        descriptive_only=descriptive_only,
        positive_significant=positive_significant,
        exceeds_margin=exceeds_margin,
        powered=True,
    )

    caveats: list[str] = []
    if knife:
        caveats.append("knife-edge stability: eigenvalue on the imaginary axis (marginal)")
    if not exceeds_margin:
        caveats.append(
            f"robustness {robustness:.3f} below pre-registered floor {robustness_min:.2f}"
        )
    if not steady_ok:
        caveats.append("nominal model does not settle within steady-state tolerance")
    if not perm_powered:
        caveats.append(
            "wiring permutation null underpowered (too few edges / acyclic); "
            "viability certified on stability + robustness only"
        )

    return {
        "steady_state_ok": bool(steady_ok),
        "stability_ok": bool(stable),
        "knife_edge": bool(knife),
        "x_steady_state": [float(v) for v in x_ss],
        "eigenvalues_real": [float(v) for v in model.eigenvalues(x_ss).real],
        "robustness": float(robustness),
        "robustness_min": float(robustness_min),
        "wilson_ci": sweep["wilson_ci"],
        "perm_p": float(perm_p),
        "perm_powered": bool(perm_powered),
        "cert": cert,
        "caveats": caveats,
    }

def _has_directed_cycle(circuit: "Circuit") -> bool:
    import networkx as nx

    g = circuit.graph
    if any(u == v for u, v in g.edges()):
        return True
    return not nx.is_directed_acyclic_graph(g)
