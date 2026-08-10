from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np
from scipy.integrate import solve_ivp

if TYPE_CHECKING:
    from .topology import Circuit

def hill_activate(x: np.ndarray | float, K: float, n: float) -> np.ndarray | float:
    xp=np.clip(x, 0.0, None) ** n
    return xp / (K ** n + xp)

def hill_repress(x: np.ndarray | float, K: float, n: float) -> np.ndarray | float:
    xp=np.clip(x, 0.0, None) ** n
    return (K ** n) / (K ** n + xp)

@dataclass
class _GeneKinetics:

    beta: float
    alpha: float
    gamma: float
    K: float
    n: float
    activators: list[int]
    repressors: list[int]

class ODEModel:

    def __init__(self, nodes: list[str], genes: list[_GeneKinetics]) -> None:
        self.nodes=nodes
        self.index={name: i for i, name in enumerate(nodes)}
        self.genes=genes
        self.dim=len(nodes)

    @classmethod
    def from_circuit(cls, circuit: "Circuit") -> "ODEModel":
        circuit.validate()
        nodes=circuit.nodes
        idx={name: i for i, name in enumerate(nodes)}
        genes: list[_GeneKinetics]=[]
        for name in nodes:
            gene=circuit.gene(name)
            prom=gene.promoter
            beta=float(prom.strength) if prom is not None else 1.0
            K=float(prom.K) if prom is not None else 0.5
            n_hill=float(prom.n_hill) if prom is not None else 2.0
            genes.append(
                _GeneKinetics(
                    beta=beta,
                    alpha=float(gene.tf.basal),
                    gamma=float(gene.tf.degradation),
                    K=K,
                    n=n_hill,
                    activators=[idx[a] for a in circuit.activators(name)],
                    repressors=[idx[r] for r in circuit.repressors(name)],
                )
            )
        return cls(nodes=nodes, genes=genes)

    def rhs(self, t: float, x: np.ndarray) -> np.ndarray:
        x=np.asarray(x, dtype=float)
        dx=np.empty(self.dim, dtype=float)
        for i, g in enumerate(self.genes):
            drive=g.beta
            for a in g.activators:
                drive *= hill_activate(x[a], g.K, g.n)
            for r in g.repressors:
                drive *= hill_repress(x[r], g.K, g.n)
            dx[i]=drive + g.alpha - g.gamma * x[i]
        return dx

    def _default_x0(self) -> np.ndarray:
        return 0.1 + 0.05 * np.arange(self.dim, dtype=float)

    def simulate(
        self,
        t_span: tuple[float, float] = (0.0, 50.0),
        x0: np.ndarray | None = None,
        n_points: int = 500,
    ) -> dict:
        x0=self._default_x0() if x0 is None else np.asarray(x0, dtype=float)
        t_eval=np.linspace(t_span[0], t_span[1], n_points)
        sol=solve_ivp(
            self.rhs,
            t_span,
            x0,
            method="LSODA",
            t_eval=t_eval,
            rtol=1e-6,
            atol=1e-9,
        )
        if not sol.success:
            raise RuntimeError(f"solve_ivp failed: {sol.message}")
        return {"t": sol.t, "x": sol.y, "success": bool(sol.success)}

    def steady_state(
        self,
        t_span: tuple[float, float] = (0.0, 200.0),
        x0: np.ndarray | None = None,
        n_points: int = 1000,
    ) -> np.ndarray:
        sol=self.simulate(t_span=t_span, x0=x0, n_points=n_points)
        return np.asarray(sol["x"][:, -1], dtype=float)

    def jacobian(self, x: np.ndarray, eps: float = 1e-6) -> np.ndarray:
        x=np.asarray(x, dtype=float)
        f0=self.rhs(0.0, x)
        J=np.empty((self.dim, self.dim), dtype=float)
        for j in range(self.dim):
            xp=x.copy()
            h=eps * max(1.0, abs(x[j]))
            xp[j] += h
            J[:, j]=(self.rhs(0.0, xp) - f0) / h
        return J

    def eigenvalues(self, x: np.ndarray) -> np.ndarray:
        return np.linalg.eigvals(self.jacobian(x))

    def is_stable(self, x: np.ndarray, tol: float = 1e-9) -> bool:
        ev=self.eigenvalues(x)
        return bool(np.all(ev.real < -tol))

    def is_knife_edge(self, x: np.ndarray, tol: float = 1e-6) -> bool:
        ev=self.eigenvalues(x)
        return bool(np.any(np.abs(ev.real) <= tol))
