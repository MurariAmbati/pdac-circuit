from __future__ import annotations

from dataclasses import dataclass,field
from typing import TYPE_CHECKING,Literal

import networkx as nx

if TYPE_CHECKING:
    from .boolean import BooleanModel
    from .ode import ODEModel

Logic=Literal["AND","OR","NAND"]
Sign=Literal[-1,1]

@dataclass
class Promoter:

    id: str
    strength: float
    inputs: list[str]
    logic: Logic = "AND"
    n_hill: float = 2.0
    K: float = 0.5

@dataclass
class Enhancer:

    id: str
    tissue_score: float
    active_in: set[str] = field(default_factory=set)

@dataclass
class Repressor:

    id: str
    target_tf: str
    max_repression: float = 1.0

@dataclass
class TF:

    id: str
    basal: float = 0.1
    degradation: float = 1.0

@dataclass
class Gene:

    tf: TF
    promoter: Promoter | None = None

class Circuit:

    def __init__(self,name: str = "circuit") -> None:
        self.name=name
        self.graph: nx.DiGraph = nx.DiGraph()
        self.enhancers: dict[str,Enhancer] = {}

    def add_gene(
        self,
        node_id: str,
        *,
        basal: float = 0.1,
        degradation: float = 1.0,
        promoter: Promoter | None = None,
    ) -> "Circuit":
        tf=TF(id=node_id,basal=basal,degradation=degradation)
        self.graph.add_node(node_id,gene=Gene(tf=tf,promoter=promoter))
        return self

    def add_tf(self,tf: TF,promoter: Promoter | None = None) -> "Circuit":
        self.graph.add_node(tf.id,gene=Gene(tf=tf,promoter=promoter))
        return self

    def add_edge(self,src: str,dst: str,sign: Sign) -> "Circuit":
        if sign not in (-1,1):
            raise ValueError(f"edge sign must be +1 or -1, got {sign!r}")
        self.graph.add_edge(src,dst,sign=int(sign))
        return self

    def add_enhancer(self,enhancer: Enhancer) -> "Circuit":
        self.enhancers[enhancer.id]=enhancer
        return self

    @property
    def nodes(self) -> list[str]:
        return sorted(self.graph.nodes)

    def gene(self,node_id: str) -> Gene:
        return self.graph.nodes[node_id]["gene"]

    def activators(self,node_id: str) -> list[str]:
        return sorted(
            u for u,_,d in self.graph.in_edges(node_id,data=True) if d["sign"] == 1
        )

    def repressors(self,node_id: str) -> list[str]:
        return sorted(
            u for u,_,d in self.graph.in_edges(node_id,data=True) if d["sign"] == -1
        )

    def edge_sign(self,src: str,dst: str) -> int:
        return int(self.graph.edges[src,dst]["sign"])

    def validate(self) -> None:
        nodes=set(self.graph.nodes)
        for u,v,d in self.graph.edges(data=True):
            if u not in nodes:
                raise ValueError(f"edge {u}->{v} has undeclared source {u!r}")
            if v not in nodes:
                raise ValueError(f"edge {u}->{v} has undeclared target {v!r}")
            if d.get("sign") not in (-1,1):
                raise ValueError(f"edge {u}->{v} has invalid sign {d.get('sign')!r}")
        for n in nodes:
            gene=self.graph.nodes[n].get("gene")
            if gene is None:
                raise ValueError(f"node {n!r} has no Gene record")
            prom=gene.promoter
            if prom is not None:
                for inp in prom.inputs:
                    if inp not in nodes:
                        raise ValueError(
                            f"promoter {prom.id!r} on {n!r} references undeclared input {inp!r}"
                        )
                if prom.logic not in ("AND","OR","NAND"):
                    raise ValueError(f"promoter {prom.id!r} has invalid logic {prom.logic!r}")

    def is_valid(self) -> bool:
        try:
            self.validate()
            return True
        except ValueError:
            return False

    def to_boolean(self) -> "BooleanModel":
        from .boolean import BooleanModel

        return BooleanModel.from_circuit(self)

    def to_ode(self) -> "ODEModel":
        from .ode import ODEModel

        return ODEModel.from_circuit(self)

    def __repr__(self) -> str:
        return (
            f"Circuit(name={self.name!r}, n_nodes={self.graph.number_of_nodes()}, "
            f"n_edges={self.graph.number_of_edges()})"
        )
