from __future__ import annotations

from dataclasses import dataclass,field

OBJECTIVES: tuple[str,...] = ("efficacy","specificity","robustness","safety")

@dataclass
class SubScores:

    efficacy: float
    specificity: float
    robustness: float
    safety: float
    components: dict = field(default_factory=dict)
    intervals: dict = field(default_factory=dict)

    def as_vector(self) -> tuple[float,float,float,float]:
        return (self.efficacy,self.specificity,self.robustness,self.safety)

    def get(self,objective: str) -> float:
        return float(getattr(self,objective))

@dataclass
class CircuitScore:

    circuit_id: str
    sub: SubScores
    composite: float = 0.0
    pareto_rank: int = -1
    crowding: float = 0.0
    acceptable: bool = True
    dominated_by: list[str] = field(default_factory=list)
