from __future__ import annotations

import json
from typing import Sequence

import numpy as np

from pdac_circuit.core.contract import OutputEnvelope
from pdac_circuit.core.paths import REGISTRY_JSON
from pdac_circuit.stats import classify_cert

from .objectives import composite as composite_score
from .types import OBJECTIVES,CircuitScore,SubScores

_DEFAULT_SIGMA=0.02

def _load_floors() -> dict:
    reg=json.loads(REGISTRY_JSON.read_text(encoding="utf-8"))
    m=reg["prereg"]["module_VI"]
    return {
        "safety_floor": float(m["safety_floor"]),
        "efficacy_floor": float(m["efficacy_floor"]),
        "front_membership_min": float(m["front_membership_min"]),
    }

def dominates(a: SubScores,b: SubScores) -> bool:
    av=np.asarray(a.as_vector(),dtype=float)
    bv=np.asarray(b.as_vector(),dtype=float)
    return bool(np.all(av >= bv) and np.any(av > bv))

def fast_nondominated_sort(scores: Sequence[SubScores]) -> list[list[int]]:
    n=len(scores)
    if n == 0:
        return []

    S: list[list[int]] = [[] for _ in range(n)]
    n_dom=np.zeros(n,dtype=int)

    for p in range(n):
        for q in range(p + 1,n):
            if dominates(scores[p],scores[q]):
                S[p].append(q)
                n_dom[q] += 1
            elif dominates(scores[q],scores[p]):
                S[q].append(p)
                n_dom[p] += 1

    fronts: list[list[int]] = []
    current=[i for i in range(n) if n_dom[i] == 0]
    while current:
        fronts.append(sorted(current))
        nxt: list[int] = []
        for p in current:
            for q in S[p]:
                n_dom[q] -= 1
                if n_dom[q] == 0:
                    nxt.append(q)
        current=nxt
    return fronts

def crowding_distance(
    front_indices: Sequence[int],scores: Sequence[SubScores]
) -> np.ndarray:
    m=len(front_indices)
    dist=np.zeros(m,dtype=float)
    if m == 0:
        return dist
    if m <= 2:
        dist[:]=np.inf
        return dist

    vals=np.array([scores[i].as_vector() for i in front_indices],dtype=float)
    for obj in range(len(OBJECTIVES)):
        col=vals[:,obj]
        order=np.argsort(col,kind="stable")
        lo=col[order[0]]
        hi=col[order[-1]]
        span=hi - lo
        dist[order[0]] = np.inf
        dist[order[-1]] = np.inf
        if span <= 0.0:
            continue
        for k in range(1,m - 1):
            prev_v=col[order[k - 1]]
            next_v=col[order[k + 1]]
            dist[order[k]] += (next_v - prev_v) / span
    return dist

def pareto_rank(
    circuits: Sequence[CircuitScore],
    *,
    safety_floor: float | None = None,
) -> list[CircuitScore]:
    if safety_floor is None:
        safety_floor=_load_floors()["safety_floor"]

    n=len(circuits)
    for c in circuits:
        c.acceptable=bool(c.sub.safety >= safety_floor)
        c.dominated_by=[]

    feasible=[i for i in range(n) if circuits[i].acceptable]
    infeasible=[i for i in range(n) if not circuits[i].acceptable]

    for i in range(n):
        doms=[]
        for j in range(n):
            if i != j and dominates(circuits[j].sub,circuits[i].sub):
                doms.append(circuits[j].circuit_id)
        circuits[i].dominated_by = doms

    rank_offset=0
    for group in (feasible,infeasible):
        if not group:
            continue
        sub_scores=[circuits[i].sub for i in group]
        fronts=fast_nondominated_sort(sub_scores)
        for f_idx,front in enumerate(fronts):
            global_idxs=[group[k] for k in front]
            cd=crowding_distance(front,sub_scores)
            for local_k,gi in zip(front,global_idxs):
                circuits[gi].pareto_rank = rank_offset + f_idx
                pos=front.index(local_k)
                circuits[gi].crowding = float(cd[pos])
        rank_offset += len(fronts)

    return list(circuits)

def select_top(
    circuits: Sequence[CircuitScore],
    k: int,
    preference: dict | None = None,
) -> list[CircuitScore]:
    front0=[c for c in circuits if c.acceptable and c.pareto_rank == 0]
    if not front0:
        return []

    if preference:
        weights=np.array([float(preference.get(o,0.0)) for o in OBJECTIVES],dtype=float)
        if weights.sum() <= 0.0:
            weights=np.ones(len(OBJECTIVES),dtype=float)
        ranked=sorted(
            front0,
            key=lambda c: composite_score(c.sub,weights),
            reverse=True,
        )
    else:
        ranked=sorted(front0,key=lambda c: (-c.crowding,c.circuit_id))
    return ranked[: max(0,int(k))]

def _objective_sigmas(sub: SubScores) -> np.ndarray:
    sig=np.full(len(OBJECTIVES),_DEFAULT_SIGMA,dtype=float)
    for j,obj in enumerate(OBJECTIVES):
        iv=sub.intervals.get(obj) if sub.intervals else None
        if iv is not None:
            lo,hi = float(iv[0]),float(iv[1])
            sig[j]=max(0.0,(hi - lo) / 2.0)
    return sig

def front_membership_probability(
    circuits: Sequence[CircuitScore],
    n_draws: int = 1000,
    seed: int = 20260620,
    *,
    safety_floor: float | None = None,
) -> dict[str,float]:
    if safety_floor is None:
        safety_floor=_load_floors()["safety_floor"]

    n=len(circuits)
    ids=[c.circuit_id for c in circuits]
    if n == 0:
        return {}

    base=np.array([c.sub.as_vector() for c in circuits],dtype=float)
    sig=np.array([_objective_sigmas(c.sub) for c in circuits],dtype=float)

    rng=np.random.default_rng(seed)
    counts=np.zeros(n,dtype=float)

    for _ in range(n_draws):
        noise=rng.normal(0.0,1.0,size=base.shape) * sig
        pert=np.clip(base + noise,0.0,1.0)
        safety_col=pert[:,OBJECTIVES.index("safety")]
        feasible=np.where(safety_col >= safety_floor)[0]
        if feasible.size == 0:
            continue
        sub_list=[
            SubScores(
                efficacy=pert[i,0],
                specificity=pert[i,1],
                robustness=pert[i,2],
                safety=pert[i,3],
            )
            for i in feasible
        ]
        fronts=fast_nondominated_sort(sub_list)
        if not fronts:
            continue
        for local in fronts[0]:
            counts[feasible[local]] += 1.0

    return {ids[i]: float(counts[i] / n_draws) for i in range(n)}

def score_circuits(
    circuits: Sequence[CircuitScore],
    *,
    weights: Sequence[float] | None = None,
    preference: dict | None = None,
    top_k: int = 3,
    membership_draws: int = 1000,
    seed: int = 20260620,
    data_classes: Sequence[str] = (),
) -> OutputEnvelope:
    floors=_load_floors()
    safety_floor=floors["safety_floor"]
    efficacy_floor=floors["efficacy_floor"]
    membership_min=floors["front_membership_min"]

    circuits=list(circuits)
    for c in circuits:
        c.composite=composite_score(c.sub,weights)

    pareto_rank(circuits,safety_floor=safety_floor)

    passes=[
        c
        for c in circuits
        if c.sub.safety >= safety_floor and c.sub.efficacy >= efficacy_floor
    ]

    if not passes:
        binding,miss = _closest_miss(circuits,safety_floor,efficacy_floor)
        cert=classify_cert(gate_ok=False)
        reason=(
            f"No circuit clears the Module VI floors (safety_floor={safety_floor:.2f}, "
            f"efficacy_floor={efficacy_floor:.2f}). Binding constraint: {binding}. "
            f"Closest miss: circuit {miss['circuit_id']!r} short by {miss['gap']:.4f} on "
            f"{miss['objective']} (value {miss['value']:.4f})."
        )
        env=OutputEnvelope.abstain(
            reason,
            cert="certified-negative",
            data_classes=data_classes,
            audit={
                "binding_constraint": binding,
                "closest_miss": miss,
                "floors": floors,
                "lattice_cert": cert,
                "n_circuits": len(circuits),
            },
        )
        return env

    membership=front_membership_probability(
        circuits,n_draws=membership_draws,seed=seed,safety_floor=safety_floor
    )
    selected=select_top(circuits,top_k,preference=preference)

    selection=[]
    for c in selected:
        p=membership.get(c.circuit_id,0.0)
        selection.append(
            {
                "circuit_id": c.circuit_id,
                "composite": c.composite,
                "pareto_rank": c.pareto_rank,
                "crowding": c.crowding,
                "sub_scores": dict(zip(OBJECTIVES,c.sub.as_vector())),
                "front_membership_prob": p,
                "robust": bool(p >= membership_min),
            }
        )

    payload={
        "n_circuits": len(circuits),
        "n_passing_floors": len(passes),
        "front0_ids": [c.circuit_id for c in circuits if c.acceptable and c.pareto_rank == 0],
        "selected": selection,
        "front_membership": membership,
        "floors": floors,
    }
    caveats=[
        "Pareto scoring is descriptive multi-objective ranking, not a clinical recommendation.",
        "Immunogenicity contributions to the safety objective are a coarse heuristic proxy.",
    ]
    return OutputEnvelope.ok(
        payload,data_classes=data_classes,cert="real",caveats=caveats,audit={"floors": floors}
    )

def _closest_miss(
    circuits: Sequence[CircuitScore],safety_floor: float,efficacy_floor: float
) -> tuple[str,dict]:
    best=None
    for c in circuits:
        safety_gap=max(0.0,safety_floor - c.sub.safety)
        eff_gap=max(0.0,efficacy_floor - c.sub.efficacy)
        for obj,gap,val,floor in (
            ("safety",safety_gap,c.sub.safety,safety_floor),
            ("efficacy",eff_gap,c.sub.efficacy,efficacy_floor),
        ):
            if gap <= 0.0:
                continue
            if best is None or gap < best[0]:
                best=(gap,obj,c,val,floor)

    if best is None:
        c=circuits[0]
        return "joint(safety,efficacy)",{
            "circuit_id": c.circuit_id,
            "objective": "joint",
            "gap": 0.0,
            "value": float(min(c.sub.safety,c.sub.efficacy)),
        }

    gap,obj,c,val,floor = best
    binding=f"{obj} >= {floor:.2f}"
    miss={
        "circuit_id": c.circuit_id,
        "objective": obj,
        "gap": float(gap),
        "value": float(val),
        "floor": float(floor),
    }
    return binding,miss
