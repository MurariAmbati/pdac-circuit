from __future__ import annotations

from .topology import Circuit,Enhancer,Promoter

def toggle_switch(
    *,
    beta: float = 4.0,
    gamma: float = 1.0,
    K: float = 0.5,
    n_hill: float = 3.0,
    basal: float = 0.05,
) -> Circuit:
    c = Circuit("toggle_switch")
    c.add_gene(
        "A",
        basal=basal,
        degradation=gamma,
        promoter=Promoter(id="pA",strength=beta,inputs=["B"],logic="AND",n_hill=n_hill,K=K),
    )
    c.add_gene(
        "B",
        basal=basal,
        degradation=gamma,
        promoter=Promoter(id="pB",strength=beta,inputs=["A"],logic="AND",n_hill=n_hill,K=K),
    )
    c.add_edge("A","B",sign=-1)
    c.add_edge("B","A",sign=-1)
    c.validate()
    return c

def repressilator(
    *,
    beta: float = 10.0,
    gamma: float = 1.0,
    K: float = 0.5,
    n_hill: float = 3.0,
    basal: float = 0.05,
) -> Circuit:
    c = Circuit("repressilator")
    for src,dst,name in (("C","A","pA"),("A","B","pB"),("B","C","pC")):
        node = dst
        c.add_gene(
            node,
            basal=basal,
            degradation=gamma,
            promoter=Promoter(id=name,strength=beta,inputs=[src],logic="AND",n_hill=n_hill,K=K),
        )
    c.add_edge("A","B",sign=-1)
    c.add_edge("B","C",sign=-1)
    c.add_edge("C","A",sign=-1)
    c.validate()
    return c

def negative_feedback(
    *,
    beta: float = 4.0,
    gamma: float = 1.0,
    K: float = 0.5,
    n_hill: float = 2.0,
    basal: float = 0.1,
) -> Circuit:
    c = Circuit("negative_feedback")
    c.add_gene(
        "A",
        basal=basal,
        degradation=gamma,
        promoter=Promoter(id="pA",strength=beta,inputs=["A"],logic="AND",n_hill=n_hill,K=K),
    )
    c.add_edge("A","A",sign=-1)
    c.validate()
    return c

def negative_feedback_two_node(
    *,
    beta: float = 4.0,
    gamma: float = 1.0,
    K: float = 0.5,
    n_hill: float = 2.0,
    basal: float = 0.1,
) -> Circuit:
    c = Circuit("negative_feedback_2node")
    c.add_gene(
        "A",
        basal=basal,
        degradation=gamma,
        promoter=Promoter(id="pA",strength=beta,inputs=["B"],logic="AND",n_hill=n_hill,K=K),
    )
    c.add_gene(
        "B",
        basal=basal,
        degradation=gamma,
        promoter=Promoter(id="pB",strength=beta,inputs=["A"],logic="AND",n_hill=n_hill,K=K),
    )
    c.add_edge("A","B",sign=+1)
    c.add_edge("B","A",sign=-1)
    c.validate()
    return c

def fragile_circuit() -> Circuit:
    return repressilator(beta=10.0,gamma=1.0,K=0.5,n_hill=3.0,basal=0.05)

def positive_loop(
    *,
    beta: float = 6.0,
    gamma: float = 1.0,
    K: float = 0.5,
    n_hill: float = 3.0,
    basal: float = 0.03,
) -> Circuit:
    c = Circuit("positive_loop")
    for src,dst,name in (("C","A","pA"),("A","B","pB"),("B","C","pC")):
        c.add_gene(
            dst,
            basal=basal,
            degradation=gamma,
            promoter=Promoter(id=name,strength=beta,inputs=[src],logic="AND",n_hill=n_hill,K=K),
        )
    c.add_edge("C","A",sign=+1)
    c.add_edge("A","B",sign=+1)
    c.add_edge("B","C",sign=+1)
    c.validate()
    return c

def robust_circuit() -> Circuit:
    return positive_loop(beta=6.0,gamma=1.0,K=0.5,n_hill=3.0,basal=0.03)

def monostable_circuit() -> Circuit:
    return negative_feedback_two_node(beta=5.0,gamma=1.0,K=0.5,n_hill=2.0,basal=0.2)

def add_pdac_enhancer(circuit: Circuit) -> Circuit:
    circuit.add_enhancer(Enhancer(id="pdac_enh",tissue_score=0.9,active_in={"PDAC"}))
    return circuit
