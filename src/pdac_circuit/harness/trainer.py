from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from ..core.seeds import set_seeds
from ..stats.metrics import auroc, expected_calibration_error, pearson, spearman

@dataclass
class TrainConfig:
    task: str = "regression"
    epochs: int = 40
    lr: float = 1e-3
    weight_decay: float = 1e-4
    batch_size: int = 64
    patience: int = 6
    seed: int = 20260620
    amp: bool = True
    grad_clip: float = 1.0
    multitask_lambda: float = 0.5
    device: str = "auto"

@dataclass
class TrainResult:
    best_metric: float
    history: list[dict] = field(default_factory=list)
    test_metrics: dict = field(default_factory=dict)
    epochs_run: int = 0

def _device(cfg: TrainConfig):
    import torch

    if cfg.device != "auto":
        return torch.device(cfg.device)
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")

def _to_tensor(x, device, dtype=None):
    import torch

    t=torch.from_numpy(np.asarray(x))
    if dtype is not None:
        t=t.to(dtype)
    return t.to(device)

def _eval_metrics(task, y_true, y_pred, y_pred2=None, y_true2=None) -> dict:
    if task == "binary":
        m={"auroc": auroc(y_true, y_pred), "ece": expected_calibration_error(y_true, y_pred)}
    elif task == "multitask":
        m={"auroc": auroc(y_true, y_pred)}
        if y_pred2 is not None and y_true2 is not None:
            m["signal_spearman"]=spearman(y_true2, y_pred2)
    else:
        m={"spearman": spearman(y_true, y_pred), "pearson": pearson(y_true, y_pred)}
    return m

def _primary(task, metrics) -> float:
    if task == "regression":
        return metrics.get("spearman", float("nan"))
    return metrics.get("auroc", float("nan"))

class Trainer:
    def __init__(self, cfg: TrainConfig):
        self.cfg=cfg

    def fit(self, model, train, val) -> TrainResult:
        import torch

        set_seeds(self.cfg.seed)
        dev=_device(self.cfg)
        model=model.to(dev)
        cfg=self.cfg

        Xtr=_to_tensor(train["X"], dev, torch.float32)
        atr=_to_tensor(train["aux"], dev, torch.float32) if train.get("aux") is not None else None
        ytr=_to_tensor(train["y"], dev, torch.float32)
        Xva=_to_tensor(val["X"], dev, torch.float32)
        ava=_to_tensor(val["aux"], dev, torch.float32) if val.get("aux") is not None else None
        yva_np=np.asarray(val["y"])

        opt=torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
        sched=torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=cfg.epochs)
        use_amp=cfg.amp and dev.type == "cuda"
        scaler=torch.amp.GradScaler("cuda", enabled=use_amp)

        n=Xtr.shape[0]
        best_metric, best_state, best_epoch = -np.inf, None, 0
        history=[]
        rng=np.random.default_rng(cfg.seed)

        for epoch in range(cfg.epochs):
            model.train()
            perm=rng.permutation(n)
            for i in range(0, n, cfg.batch_size):
                bi=perm[i : i + cfg.batch_size]
                xb=Xtr[bi]
                ab=atr[bi] if atr is not None else None
                yb=ytr[bi]
                opt.zero_grad(set_to_none=True)
                with torch.amp.autocast("cuda", enabled=use_amp, dtype=torch.bfloat16):
                    out=model(xb, ab) if ab is not None else model(xb)
                    loss=self._loss(out, yb)
                scaler.scale(loss).backward()
                if cfg.grad_clip:
                    scaler.unscale_(opt)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
                scaler.step(opt)
                scaler.update()
            sched.step()

            vm=self._evaluate(model, Xva, ava, yva_np, dev)
            primary=_primary(cfg.task, vm)
            history.append({"epoch": epoch, **vm})
            if np.isfinite(primary) and primary > best_metric:
                best_metric, best_epoch = primary, epoch
                best_state={k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            if epoch - best_epoch >= cfg.patience:
                break

        if best_state is not None:
            model.load_state_dict(best_state)
        return TrainResult(best_metric=float(best_metric), history=history, epochs_run=len(history))

    def _loss(self, out, yb):
        import torch.nn.functional as F

        task=self.cfg.task
        if task == "binary":
            return F.binary_cross_entropy_with_logits(out.squeeze(-1), yb)
        if task == "multitask":
            logit, signal = out[:, 0], out[:, 1]
            active, sig_true = yb[:, 0], yb[:, 1]
            return F.binary_cross_entropy_with_logits(logit, active) + self.cfg.multitask_lambda * F.smooth_l1_loss(signal, sig_true)
        return F.smooth_l1_loss(out.squeeze(-1), yb)

    def _evaluate(self, model, X, aux, y_np, dev) -> dict:
        import torch

        model.eval()
        preds=[]
        with torch.no_grad():
            for i in range(0, X.shape[0], 512):
                xb=X[i : i + 512]
                ab=aux[i : i + 512] if aux is not None else None
                with torch.amp.autocast("cuda", enabled=(self.cfg.amp and dev.type == "cuda"), dtype=torch.bfloat16):
                    out=model(xb, ab) if ab is not None else model(xb)
                preds.append(out.float().cpu().numpy())
        pred=np.concatenate(preds, axis=0)
        task=self.cfg.task
        if task == "binary":
            p=1 / (1 + np.exp(-pred.reshape(-1)))
            return _eval_metrics(task, y_np, p)
        if task == "multitask":
            p_active=1 / (1 + np.exp(-pred[:, 0]))
            return _eval_metrics(task, y_np[:, 0], p_active, y_pred2=pred[:, 1], y_true2=y_np[:, 1])
        return _eval_metrics(task, y_np, pred.reshape(-1))

    def predict(self, model, X, aux=None) -> np.ndarray:
        import torch

        dev=_device(self.cfg)
        model=model.to(dev).eval()
        Xt=_to_tensor(X, dev, torch.float32)
        at=_to_tensor(aux, dev, torch.float32) if aux is not None else None
        preds=[]
        with torch.no_grad():
            for i in range(0, Xt.shape[0], 512):
                xb=Xt[i : i + 512]
                ab=at[i : i + 512] if at is not None else None
                out=model(xb, ab) if ab is not None else model(xb)
                preds.append(out.float().cpu().numpy())
        return np.concatenate(preds, axis=0)
