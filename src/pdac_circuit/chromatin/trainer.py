from __future__ import annotations

from dataclasses import asdict
import hashlib
import json
from pathlib import Path

from .config import ChromatinModelConfig, ChromatinTrainConfig
from .losses import LossWeights, total_chromatin_loss

def _config_hash(
    model: ChromatinModelConfig,
    training: ChromatinTrainConfig,
    training_stage: str = "human_state_adaptation",
) -> str:
    payload=json.dumps(
        {
            "model": asdict(model),
            "training": asdict(training),
            "training_stage": training_stage,
        },
        sort_keys=True,
        separators=(",", ":"),
    )
    return hashlib.sha256(payload.encode()).hexdigest()

def _code_fingerprint() -> str:

    directory=Path(__file__).parent
    digest=hashlib.sha256()
    for name in (
        "config.py",
        "curriculum.py",
        "losses.py",
        "model.py",
        "streaming.py",
        "trainer.py",
    ):
        digest.update(name.encode())
        digest.update((directory / name).read_bytes())
    return digest.hexdigest()

def parameter_report(model) -> dict:
    parameters=sum(value.numel() for value in model.parameters())
    trainable=sum(value.numel() for value in model.parameters() if value.requires_grad)
    return {
        "parameters": parameters,
        "trainable_parameters": trainable,
        "estimated_adamw_state_gb": round(trainable * 16 / (1024**3), 3),
    }

def available_cuda_memory_gb() -> float | None:
    try:
        import torch

        if not torch.cuda.is_available():
            return None
        free, _ = torch.cuda.mem_get_info()
        return free / (1024**3)
    except Exception:
        return None

def load_weights_for_inference(
    model,
    model_config: ChromatinModelConfig,
    train_config: ChromatinTrainConfig,
    checkpoint_path: str | Path,
    *,
    device,
) -> dict:

    import torch

    state=torch.load(checkpoint_path, map_location=device, weights_only=False)
    if state.get("schema") != "pdac-circuit.chromatin-checkpoint/1":
        raise ValueError("invalid chromatin checkpoint schema")
    training_stage=state.get("training_stage", "human_state_adaptation")
    if state.get("config_hash") != _config_hash(
        model_config, train_config, training_stage
    ):
        raise ValueError("checkpoint/config hash mismatch; refusing inference")
    if state.get("code_fingerprint") != _code_fingerprint():
        raise ValueError("checkpoint/code fingerprint mismatch; refusing inference")
    model.load_state_dict(state["model"])
    model.to(device).eval()
    return {
        "global_step": int(state["global_step"]),
        "optimizer_step": int(state["optimizer_step"]),
        "epoch": int(state["epoch"]),
        "code_fingerprint": state["code_fingerprint"],
        "config_hash": state["config_hash"],
        "training_stage": training_stage,
        "data_fingerprint": state.get("data_fingerprint"),
        "initialization_provenance": state.get("initialization_provenance"),
    }

def load_weights_for_initialization(
    model,
    model_config: ChromatinModelConfig,
    checkpoint_path: str | Path,
    *,
    device,
) -> dict:

    import torch

    path=Path(checkpoint_path)
    state=torch.load(path, map_location=device, weights_only=False)
    if state.get("schema") != "pdac-circuit.chromatin-checkpoint/1":
        raise ValueError("invalid chromatin checkpoint schema")
    if state.get("model_config") != asdict(model_config):
        raise ValueError("initialization checkpoint architecture mismatch")
    if state.get("code_fingerprint") != _code_fingerprint():
        raise ValueError("initialization checkpoint code fingerprint mismatch")
    model.load_state_dict(state["model"])
    digest=hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return {
        "schema": "pdac-circuit.weight-initialization/1",
        "checkpoint": str(path),
        "checkpoint_sha256": digest.hexdigest(),
        "source_training_stage": state.get("training_stage"),
        "source_config_hash": state.get("config_hash"),
        "source_code_fingerprint": state.get("code_fingerprint"),
        "source_data_fingerprint": state.get("data_fingerprint"),
    }

class MemoryBoundedTrainer:
    def __init__(
        self,
        model,
        model_config: ChromatinModelConfig,
        train_config: ChromatinTrainConfig,
        *,
        loss_weights: LossWeights | None = None,
        training_stage: str = "human_state_adaptation",
        data_fingerprint: dict | None = None,
        initialization_provenance: dict | None = None,
    ):
        import torch

        model_config.validate()
        train_config.validate()
        self.model=model
        self.model_config=model_config
        self.train_config=train_config
        self.loss_weights=loss_weights or LossWeights(
            profile=train_config.loss_profile,
            correlation=train_config.loss_correlation,
            uncertainty=train_config.loss_uncertainty,
            residual_delta=train_config.loss_residual_delta,
            perturbation_delta=train_config.loss_perturbation_delta,
            healthy_zero=train_config.loss_healthy_zero,
            state_graph=train_config.loss_state_graph,
            domain_invariance=train_config.loss_domain_invariance,
        )
        self.training_stage=training_stage
        self.data_fingerprint=data_fingerprint
        self.initialization_provenance=initialization_provenance
        if train_config.device == "auto":
            self.device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device=torch.device(train_config.device)
        self.model.to(self.device)
        trainable_parameters=[
            parameter for parameter in self.model.parameters() if parameter.requires_grad
        ]
        if not trainable_parameters:
            raise ValueError("trainer has no trainable parameters")
        self.optimizer=torch.optim.AdamW(
            trainable_parameters,
            lr=train_config.learning_rate,
            weight_decay=train_config.weight_decay,
        )
        self.scheduler=torch.optim.lr_scheduler.LambdaLR(self.optimizer, self._lr_factor)
        use_scaler=self.device.type == "cuda" and train_config.amp_dtype == "float16"
        self.scaler=torch.amp.GradScaler("cuda", enabled=use_scaler)
        self.global_step=0
        self.optimizer_step=0
        self.epoch=0
        self.batch_in_epoch=0
        self._restored_gradients=False
        self.best_validation_loss=float("inf")
        self.validations_without_improvement=0
        self.last_validation=None

    def _lr_factor(self, step: int) -> float:
        import math

        cfg=self.train_config
        if step < cfg.warmup_steps:
            return max(step, 1) / max(cfg.warmup_steps, 1)
        progress=min(1.0, (step - cfg.warmup_steps) / max(1, cfg.max_steps - cfg.warmup_steps))
        return 0.05 + 0.95 * 0.5 * (1.0 + math.cos(math.pi * progress))

    def _autocast(self):
        import contextlib
        import torch

        if self.device.type != "cuda" or self.train_config.amp_dtype == "float32":
            return contextlib.nullcontext()
        dtype=torch.bfloat16 if self.train_config.amp_dtype == "bfloat16" else torch.float16
        return torch.amp.autocast("cuda", dtype=dtype)

    def _move(self, batch: dict) -> dict:
        moved={}
        for key, value in batch.items():
            moved[key]=value.to(self.device, non_blocking=True) if hasattr(value, "to") else value
        return moved

    def _forward_loss(self, batch: dict):
        output=self.model(
            batch["sequence"].float(),
            batch["assay_features"].float(),
            batch["state_features"].float(),
            batch["perturbation_features"].float(),
            batch["disease_mask"].float(),
        )
        return total_chromatin_loss(
            output,
            batch["target"].float(),
            weights=self.loss_weights,
            signal_mask=batch.get("signal_mask"),
            paired_delta=batch.get("paired_delta"),
            perturbation_delta=batch.get("perturbation_delta"),
            pair_mask=batch.get("pair_mask"),
            perturbation_mask=batch.get("perturbation_mask"),
            healthy_mask=batch.get("healthy_mask"),
            graph_edges=batch.get("graph_edges"),
        )

    def evaluate(self, loader) -> dict:

        import torch

        was_training=self.model.training
        self.model.eval()
        group_losses: dict[str, list[float]] = {}
        example_losses=[]
        objective_losses=[]
        part_totals: dict[str, float] = {}
        batches=0
        try:
            with torch.no_grad():
                for batch in loader:
                    batch=self._move(batch)
                    with self._autocast():
                        loss, parts = self._forward_loss(batch)
                    groups=[str(value) for value in batch.get("sample_group", [])]
                    if len(groups) != 1:
                        raise ValueError(
                            "validation loader must use batch size one for exact group aggregation"
                        )
                    selection_value=float(parts["profile"].detach().cpu())
                    objective_value=float(loss.detach().cpu())
                    example_losses.append(selection_value)
                    objective_losses.append(objective_value)
                    group_losses.setdefault(groups[0], []).append(selection_value)
                    for key, part in parts.items():
                        part_totals[key]=part_totals.get(key, 0.0) + float(
                            part.detach().cpu()
                        )
                    batches += 1
                    maximum=self.train_config.validation_max_batches
                    if maximum is not None and batches >= maximum:
                        break
        finally:
            if was_training:
                self.model.train()
        if not example_losses or not group_losses:
            raise RuntimeError("validation loader produced no eligible validation examples")
        group_means={
            group: sum(values) / len(values) for group, values in group_losses.items()
        }
        return {
            "schema": "pdac-circuit.chromatin-validation/1",
            "optimizer_step": self.optimizer_step,
            "examples": len(example_losses),
            "groups": len(group_means),
            "selection_metric": "independent_group_mean_log_profile_loss",
            "group_mean_loss": sum(group_means.values()) / len(group_means),
            "example_mean_loss": sum(example_losses) / len(example_losses),
            "objective_example_mean_loss": sum(objective_losses) / len(objective_losses),
            "parts": {key: value / batches for key, value in part_totals.items()},
            "truncated": self.train_config.validation_max_batches is not None,
        }
    def fit(
        self,
        loader,
        checkpoint_dir: str | Path,
        *,
        resume: bool = True,
        validation_loader=None,
    ) -> dict:
        import torch

        checkpoint_dir=Path(checkpoint_dir)
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        latest=checkpoint_dir / "latest.pt"
        if resume and latest.exists():
            self.load_checkpoint(latest)

        if self.optimizer_step >= self.train_config.max_steps:
            return {
                "global_step": self.global_step,
                "optimizer_step": self.optimizer_step,
                "epoch": self.epoch,
                "batch_in_epoch": self.batch_in_epoch,
                "last": None,
                "parameter_report": parameter_report(self.model),
                "checkpoint": str(latest),
                "already_complete": True,
                "best_validation_loss": self.best_validation_loss,
                "last_validation": self.last_validation,
            }

        cfg=self.train_config
        starting_global_step=self.global_step
        if not self._restored_gradients:
            self.optimizer.zero_grad(set_to_none=True)
        history=[]
        stop=False
        starting_epoch=self.epoch
        for epoch in range(starting_epoch, cfg.epochs):
            self.epoch=epoch
            skip_batches=self.batch_in_epoch if epoch == starting_epoch else 0
            dataset=getattr(loader, "dataset", None)
            if hasattr(dataset, "set_epoch"):
                dataset.set_epoch(epoch)
            self.model.train()
            epoch_completed=True
            for batch_index, batch in enumerate(loader):
                if batch_index < skip_batches:
                    continue
                batch=self._move(batch)
                with self._autocast():
                    loss, parts = self._forward_loss(batch)
                    scaled_loss=loss / cfg.gradient_accumulation
                self.scaler.scale(scaled_loss).backward()
                self.global_step += 1
                self.batch_in_epoch=batch_index + 1

                if self.global_step % cfg.gradient_accumulation == 0:
                    self.scaler.unscale_(self.optimizer)
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), cfg.grad_clip)
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                    self.optimizer.zero_grad(set_to_none=True)
                    self.scheduler.step()
                    self.optimizer_step += 1

                    record={
                        "global_step": self.global_step,
                        "optimizer_step": self.optimizer_step,
                        "epoch": epoch,
                        "loss": float(loss.detach().cpu()),
                        **{key: float(value.detach().cpu()) for key, value in parts.items()},
                    }
                    if (
                        validation_loader is not None
                        and self.optimizer_step % cfg.eval_every == 0
                    ):
                        validation=self.evaluate(validation_loader)
                        self.last_validation=validation
                        record["validation"]=validation
                        improved=(
                            validation["group_mean_loss"]
                            < self.best_validation_loss - cfg.minimum_validation_delta
                        )
                        if improved:
                            self.best_validation_loss=validation["group_mean_loss"]
                            self.validations_without_improvement=0
                            self.save_checkpoint(checkpoint_dir / "best.pt")
                        else:
                            self.validations_without_improvement += 1
                    history.append(record)
                    if self.optimizer_step % cfg.checkpoint_every == 0:
                        self.save_checkpoint(latest)
                        self.save_checkpoint(
                            checkpoint_dir / f"step-{self.optimizer_step:08d}.pt"
                        )
                    if self.optimizer_step >= cfg.max_steps:
                        stop=True
                        epoch_completed=False
                        break
                    if (
                        validation_loader is not None
                        and self.validations_without_improvement
                        >= cfg.early_stopping_patience
                    ):
                        stop=True
                        epoch_completed=False
                        break
            if epoch_completed:
                self.epoch=epoch + 1
                self.batch_in_epoch=0
            self.save_checkpoint(latest)
            if stop:
                break

        if self.global_step == starting_global_step:
            raise RuntimeError("training loader produced no eligible batches")

        return {
            "global_step": self.global_step,
            "optimizer_step": self.optimizer_step,
            "epoch": self.epoch,
            "batch_in_epoch": self.batch_in_epoch,
            "last": history[-1] if history else None,
            "parameter_report": parameter_report(self.model),
            "checkpoint": str(latest),
            "best_checkpoint": str(checkpoint_dir / "best.pt"),
            "best_validation_loss": self.best_validation_loss,
            "last_validation": self.last_validation,
            "validations_without_improvement": self.validations_without_improvement,
            "early_stopped": (
                validation_loader is not None
                and self.validations_without_improvement >= cfg.early_stopping_patience
            ),
        }

    def save_checkpoint(self, path: str | Path) -> None:
        import torch

        path=Path(path)
        temporary=path.with_suffix(path.suffix + ".tmp")
        state={
            "schema": "pdac-circuit.chromatin-checkpoint/1",
            "config_hash": _config_hash(
                self.model_config, self.train_config, self.training_stage
            ),
            "code_fingerprint": _code_fingerprint(),
            "training_stage": self.training_stage,
            "data_fingerprint": self.data_fingerprint,
            "initialization_provenance": self.initialization_provenance,
            "model_config": asdict(self.model_config),
            "train_config": asdict(self.train_config),
            "model": self.model.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "scheduler": self.scheduler.state_dict(),
            "scaler": self.scaler.state_dict(),
            "global_step": self.global_step,
            "optimizer_step": self.optimizer_step,
            "epoch": self.epoch,
            "batch_in_epoch": self.batch_in_epoch,
            "best_validation_loss": self.best_validation_loss,
            "validations_without_improvement": self.validations_without_improvement,
            "last_validation": self.last_validation,
            "gradients": {
                name: parameter.grad.detach().cpu()
                for name, parameter in self.model.named_parameters()
                if parameter.grad is not None
            },
        }
        torch.save(state, temporary)
        temporary.replace(path)

    def load_checkpoint(self, path: str | Path) -> None:
        import torch

        state=torch.load(path, map_location=self.device, weights_only=False)
        expected=_config_hash(
            self.model_config, self.train_config, self.training_stage
        )
        if state.get("config_hash") != expected:
            raise ValueError("checkpoint/config hash mismatch; refusing an ambiguous resume")
        if state.get("code_fingerprint") != _code_fingerprint():
            raise ValueError("checkpoint/code fingerprint mismatch; refusing an ambiguous resume")
        if state.get("data_fingerprint") != self.data_fingerprint:
            raise ValueError("checkpoint/data fingerprint mismatch; refusing an ambiguous resume")
        checkpoint_initialization=state.get("initialization_provenance")
        if (
            self.initialization_provenance is not None
            and checkpoint_initialization != self.initialization_provenance
        ):
            raise ValueError("checkpoint initialization lineage mismatch")
        self.initialization_provenance=checkpoint_initialization
        self.model.load_state_dict(state["model"])
        self.optimizer.load_state_dict(state["optimizer"])
        self.scheduler.load_state_dict(state["scheduler"])
        self.scaler.load_state_dict(state["scaler"])
        self.global_step=int(state["global_step"])
        self.optimizer_step=int(state["optimizer_step"])
        self.epoch=int(state["epoch"])
        self.batch_in_epoch=int(state.get("batch_in_epoch", 0))
        self.best_validation_loss=float(state.get("best_validation_loss", float("inf")))
        self.validations_without_improvement=int(
            state.get("validations_without_improvement", 0)
        )
        self.last_validation=state.get("last_validation")
        gradients=state.get("gradients", {})
        parameters=dict(self.model.named_parameters())
        for name, gradient in gradients.items():
            if name not in parameters:
                raise ValueError(f"checkpoint gradient references unknown parameter {name!r}")
            parameters[name].grad = gradient.to(self.device)
        self._restored_gradients=bool(gradients)
        if self.global_step % self.train_config.gradient_accumulation and not gradients:
            raise ValueError("checkpoint lost partial accumulation gradients; refusing resume")
