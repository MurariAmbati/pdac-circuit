from __future__ import annotations

from dataclasses import dataclass,field
from typing import TYPE_CHECKING,Callable,Literal,Sequence

if TYPE_CHECKING:
    from .topology import Circuit

State = tuple[bool,...]
Mode = Literal["sync","async"]

def _make_rule(
    activators: list[str],
    repressors: list[str],
    logic: str,
    self_true: bool,
) -> Callable[[dict[str,bool]],bool]:

    def rule(state: dict[str,bool]) -> bool:
        if not activators and not repressors:
            return self_true
        act = [bool(state[a]) for a in activators]
        rep = [not bool(state[r]) for r in repressors]
        terms = act + rep
        if logic == "AND":
            return all(terms)
        if logic == "OR":
            return any(terms)
        if logic == "NAND":
            return not all(terms)
        raise ValueError(f"unknown logic {logic!r}")

    return rule

@dataclass
class BooleanModel:

    nodes: list[str]
    rules: dict[str,Callable[[dict[str,bool]],bool]]
    index: dict[str,int] = field(init=False)

    def __post_init__(self) -> None:
        self.index = {n: i for i,n in enumerate(self.nodes)}

    @classmethod
    def from_circuit(cls,circuit: "Circuit") -> "BooleanModel":
        circuit.validate()
        nodes = circuit.nodes
        rules: dict[str,Callable[[dict[str,bool]],bool]] = {}
        for n in nodes:
            gene = circuit.gene(n)
            activators = circuit.activators(n)
            repressors = circuit.repressors(n)
            logic = gene.promoter.logic if gene.promoter is not None else "OR"
            self_true = gene.tf.basal >= 0.5
            rules[n] = _make_rule(activators,repressors,logic,self_true)
        return cls(nodes=nodes,rules=rules)

    def _as_dict(self,state: State | Sequence[bool] | dict[str,bool]) -> dict[str,bool]:
        if isinstance(state,dict):
            return {n: bool(state[n]) for n in self.nodes}
        return {n: bool(state[i]) for i,n in enumerate(self.nodes)}

    def _as_tuple(self,state: dict[str,bool]) -> State:
        return tuple(bool(state[n]) for n in self.nodes)

    def step_sync(self,state: State | dict[str,bool]) -> State:
        cur = self._as_dict(state)
        nxt = {n: bool(self.rules[n](cur)) for n in self.nodes}
        return self._as_tuple(nxt)

    def step_async(
        self,state: State | dict[str,bool],order: Sequence[str]
    ) -> State:
        cur = self._as_dict(state)
        for n in order:
            cur[n] = bool(self.rules[n](cur))
        return self._as_tuple(cur)

    def trajectory(
        self,
        state0: State | dict[str,bool],
        steps: int,
        mode: Mode = "sync",
        order: Sequence[str] | None = None,
    ) -> list[State]:
        order = list(order) if order is not None else list(self.nodes)
        s = self._as_tuple(self._as_dict(state0))
        traj = [s]
        for _ in range(steps):
            s = self.step_sync(s) if mode == "sync" else self.step_async(s,order)
            traj.append(s)
        return traj

    def attractors(self,mode: Mode = "sync") -> list[dict]:
        n = len(self.nodes)
        if n > 20:
            raise ValueError(
                f"exhaustive attractor enumeration needs n<=20 (got {n}); "
                "use trajectory() from sampled initial states instead"
            )
        order = list(self.nodes)

        def advance(s: State) -> State:
            return self.step_sync(s) if mode == "sync" else self.step_async(s,order)

        def canon(cycle: list[State]) -> tuple[State,...]:
            rots = [tuple(cycle[i:] + cycle[:i]) for i in range(len(cycle))]
            return min(rots)

        attractor_of: dict[State,tuple[State,...]] = {}
        catalog: dict[tuple[State,...],dict] = {}

        for bits in range(2 ** n):
            s0: State = tuple(bool((bits >> i) & 1) for i in range(n))
            seen: dict[State,int] = {}
            path: list[State] = []
            s = s0
            while s not in seen and s not in attractor_of:
                seen[s] = len(path)
                path.append(s)
                s = advance(s)
            if s in attractor_of:
                key = attractor_of[s]
            else:
                start = seen[s]
                cycle = path[start:]
                key = canon(cycle)
                if key not in catalog:
                    catalog[key] = {
                        "states": [list(st) for st in key],
                        "length": len(key),
                        "basin": 0,
                        "is_fixed_point": len(key) == 1,
                    }
            for st in path:
                attractor_of[st] = key
            attractor_of[s0] = key
            catalog[key]["basin"] += 1

        out = sorted(
            catalog.values(),
            key=lambda r: (not r["is_fixed_point"],r["length"],tuple(map(tuple,r["states"]))),
        )
        return out

    def fixed_points(self,mode: Mode = "sync") -> list[State]:
        return [tuple(a["states"][0]) for a in self.attractors(mode=mode) if a["is_fixed_point"]]
