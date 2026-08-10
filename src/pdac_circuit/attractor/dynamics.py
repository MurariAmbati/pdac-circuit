from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch

from .graph import RegulatoryGraph

@dataclass
class FitResult:
    W: np.ndarray
    b: np.ndarray
    fixed_point_error: float
    dead_activation: float
    gain: float
    epochs: int
    device: str

class AttractorDynamics:
    def __init__(self, graph: RegulatoryGraph, *, gain: float = 4.0, device: str | None = None):
        self.graph=graph
        self.gain=gain
        self.device=device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.W: torch.Tensor | None=None
        self.b: torch.Tensor | None=None
        self.fit_result: FitResult | None=None

    def _mask(self) -> torch.Tensor:
        return torch.tensor(self.graph.adjacency, device=self.device)

    def fit(self, *, epochs: int = 1800, lr: float = 0.03, motif_weight: float = 0.5,
            dead_weight: float = 0.5, l2: float = 1e-3, seed: int = 20260620) -> FitResult:
        torch.manual_seed(seed)
        g=self.graph
        n=g.n
        mask=self._mask()
        init=g.signs * g.adjacency * 0.4 * (1.0 + motif_weight * g.motif_support)
        A=torch.nn.Parameter(torch.tensor(init, dtype=torch.float32, device=self.device))
        b=torch.nn.Parameter(torch.full((n,), -2.0, device=self.device))
        states=torch.tensor(g.states, dtype=torch.float32, device=self.device)
        dead0=0.05 * torch.ones(n, device=self.device)
        opt=torch.optim.Adam([A, b], lr=lr)
        fp=torch.tensor(0.0)
        dead=torch.tensor(0.0)
        for _ in range(epochs):
            opt.zero_grad()
            W=A * mask
            pred=torch.sigmoid(self.gain * (states @ W.t() + b))
            fp=((pred - states) ** 2).mean()
            dead=torch.sigmoid(self.gain * (dead0 @ W.t() + b))
            loss=fp + dead_weight * torch.relu(dead - 0.25).mean() + l2 * (A * mask).pow(2).mean()
            loss.backward()
            opt.step()
        self.W=(A * mask).detach()
        self.b=b.detach()
        self.fit_result=FitResult(
            W=self.W.cpu().numpy(),
            b=self.b.cpu().numpy(),
            fixed_point_error=float(fp.item()),
            dead_activation=float(dead.mean().item()),
            gain=self.gain,
            epochs=epochs,
            device=self.device,
        )
        return self.fit_result

    def _settle(self, x0: torch.Tensor, iters: int = 250) -> torch.Tensor:
        x=x0.clone()
        for _ in range(iters):
            xn=torch.sigmoid(self.gain * (x @ self.W.t() + self.b))
            if (xn - x).abs().max() < 1e-6:
                return xn
            x=xn
        return x

    def _settle_batch(self, X0: torch.Tensor, clamp_diag: float | None = None,
                      iters: int = 250) -> torch.Tensor:
        X=X0.clone()
        n=X.shape[0]
        idx=torch.arange(n, device=self.device)
        for _ in range(iters):
            Xn=torch.sigmoid(self.gain * (X @ self.W.t() + self.b))
            if clamp_diag is not None:
                Xn[idx, idx]=clamp_diag
            if (Xn - X).abs().max() < 1e-6:
                X=Xn
                break
            X=Xn
        return X

    def collapse_scores(self, *, per_line: bool = True) -> np.ndarray:
        assert self.W is not None, "fit() first"
        n=self.graph.n
        starts=(
            [torch.tensor(s, dtype=torch.float32, device=self.device) for s in self.graph.states]
            if per_line
            else [torch.tensor(self.graph.states.mean(axis=0), dtype=torch.float32, device=self.device)]
        )
        total=np.zeros(n)
        for start in starts:
            base=self._settle(start)
            X0=base.unsqueeze(0).repeat(n, 1)
            ko=self._settle_batch(X0, clamp_diag=0.02)
            drop=torch.relu(base.unsqueeze(0) - ko)
            drop[torch.arange(n, device=self.device), torch.arange(n, device=self.device)]=0.0
            total += drop.sum(dim=1).cpu().numpy()
        return total / len(starts)

    def attractor(self) -> np.ndarray:
        base=self._settle(torch.tensor(self.graph.states.mean(axis=0), dtype=torch.float32, device=self.device))
        return base.cpu().numpy()

    def control_design(self, *, repressible_mask: np.ndarray, essential_mask: np.ndarray,
                       max_targets: int = 6) -> dict:
        assert self.W is not None, "fit() first"
        g=self.graph
        n=g.n
        healthy=torch.tensor(g.healthy_dir, dtype=torch.float32, device=self.device)
        ess=torch.tensor(essential_mask.astype(np.float32), device=self.device)
        base=self._settle(torch.tensor(g.states.mean(axis=0), dtype=torch.float32, device=self.device))
        candidates=[i for i in range(n) if repressible_mask[i] and not essential_mask[i]]
        chosen: list[int]=[]
        clamp={}
        steps=[]

        def settle_with(clamp_map):
            x=base.clone()
            idxs=list(clamp_map.keys())
            for _ in range(250):
                xn=torch.sigmoid(self.gain * (x @ self.W.t() + self.b))
                for i in idxs:
                    xn[i]=0.02
                if (xn - x).abs().max() < 1e-6:
                    return xn
                x=xn
            return x

        cur=base
        for _ in range(max_targets):
            best_i, best_gain, best_state=None, -1e9, None
            for i in candidates:
                if i in chosen:
                    continue
                trial=dict(clamp)
                trial[i]=0.02
                new=settle_with(trial)
                move=(new - base) * healthy
                efficacy=float(torch.relu(move).sum().item())
                safety_cost=float((torch.relu(base - new) * ess).sum().item())
                net=efficacy - 2.0 * safety_cost
                if net > best_gain:
                    best_i, best_gain, best_state=i, net, new
            if best_i is None:
                break
            chosen.append(best_i)
            clamp[best_i]=0.02
            move=(best_state - base) * healthy
            steps.append({
                "target": g.nodes[best_i],
                "step_net": round(best_gain, 4),
                "efficacy_toward_healthy": round(float(torch.relu(move).sum().item()), 4),
                "essential_collapse_cost": round(float((torch.relu(base - best_state) * ess).sum().item()), 4),
            })
            cur=best_state
        shift=float(((cur - base) * healthy).sum().item())
        return {
            "targets": [g.nodes[i] for i in chosen],
            "steps": steps,
            "net_healthy_shift": round(shift, 4),
            "baseline_attractor_mean": round(float(base.mean().item()), 4),
            "controlled_attractor_mean": round(float(cur.mean().item()), 4),
        }
