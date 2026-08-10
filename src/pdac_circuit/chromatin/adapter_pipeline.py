from __future__ import annotations

from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import random
import sys
import uuid

import numpy as np

from .baseline_adapter import EnformerStateAdapter,load_adapter_config
from .evaluation import save_raw_predictions
from .streaming import sha256_file

RAW_KEYS={"model","example_id","prediction","metadata"}
CONDITION_KEYS={
    "example_id",
    "group",
    "split",
    "assay_features",
    "state_features",
    "perturbation_features",
    "metadata",
}

def _validation_scope(config_path: str | Path) -> str:
    payload=json.loads(Path(config_path).read_text(encoding="utf-8"))
    scope=str(
        payload.get("training_policy",{}).get("validation_scope","group_disjoint")
    )
    allowed={"group_disjoint","locus_disjoint_same_groups_allowed"}
    if scope not in allowed:
        raise ValueError(f"unsupported adapter validation_scope {scope!r}")
    return scope

def _archive(path: str | Path,required: set[str]) -> dict:
    with np.load(path,allow_pickle=False) as source:
        missing=sorted(required - set(source.files))
        if missing:
            raise ValueError(f"{Path(path).name} missing keys: {missing}")
        return {key: source[key].copy() for key in source.files}

def _unique_ids(values,*,label: str) -> np.ndarray:
    ids=np.asarray(values).astype(str)
    if len(ids) == 0 or len(set(ids)) != len(ids):
        raise ValueError(f"{label} example IDs must be nonempty and unique")
    return ids

def _example_id_sha256(values) -> str:
    digest=hashlib.sha256()
    for value in sorted(str(item) for item in values):
        digest.update(value.encode("utf-8"))
        digest.update(b"\n")
    return digest.hexdigest()

def _load_conditions(path: str | Path) -> dict:
    payload=_archive(path,CONDITION_KEYS)
    unknown=sorted(set(payload) - CONDITION_KEYS)
    if unknown:
        raise ValueError(
            f"condition bundle contains forbidden or unknown arrays: {unknown}"
        )
    metadata=json.loads(str(payload["metadata"].item()))
    if metadata.get("schema") != "pdac-circuit.baseline-conditions/1":
        raise ValueError("invalid baseline-condition bundle schema")
    if metadata.get("contains_targets") is not False:
        raise ValueError("adapter condition bundle must explicitly contain no targets")
    if metadata.get("candidate_feature_access") is not False:
        raise ValueError("adapter condition bundle may not access candidate features")
    payload["example_id"]=_unique_ids(
        payload["example_id"],label="condition bundle"
    )
    payload["metadata_json"]=metadata
    return payload

def _load_raw_enformer(path: str | Path) -> dict:
    payload=_archive(path,RAW_KEYS)
    if str(payload["model"].item()) != "Enformer":
        raise ValueError("state adapter input must be frozen raw Enformer predictions")
    payload["example_id"]=_unique_ids(payload["example_id"],label="Enformer raw")
    if not np.isfinite(payload["prediction"]).all():
        raise ValueError("Enformer raw predictions contain non-finite values")
    payload["metadata_json"]=json.loads(str(payload["metadata"].item()))
    return payload

def _align_raw_conditions(raw: dict,conditions: dict,config) -> dict:
    raw_ids=raw["example_id"]
    condition_ids=conditions["example_id"]
    if set(raw_ids) != set(condition_ids) or len(raw_ids) != len(condition_ids):
        raise ValueError("Enformer raw and condition example-ID sets differ")
    cohort_sha=_example_id_sha256(condition_ids)
    if (
        raw["metadata_json"].get("example_id_sha256") != cohort_sha
        or conditions["metadata_json"].get("example_id_sha256") != cohort_sha
    ):
        raise ValueError("Enformer raw and conditions lack one matching cohort hash")
    raw_index={value: index for index,value in enumerate(raw_ids)}
    order=np.asarray([raw_index[value] for value in condition_ids])
    prediction=raw["prediction"][order].astype(np.float32)
    if prediction.shape != (len(condition_ids),config.bins):
        raise ValueError(
            f"Enformer predictions must have shape ({len(condition_ids)}, {config.bins})"
        )
    for key,expected in (
        ("assay_features",config.assay_features),
        ("state_features",config.state_features),
        ("perturbation_features",config.perturbation_features),
    ):
        if conditions[key].shape != (len(condition_ids),expected):
            raise ValueError(
                f"{key} must have shape ({len(condition_ids)}, {expected})"
            )
    return {
        "example_id": condition_ids,
        "prediction": prediction,
        "group": conditions["group"].astype(str),
        "split": conditions["split"].astype(str),
        "assay_features": conditions["assay_features"].astype(np.float32),
        "state_features": conditions["state_features"].astype(np.float32),
        "perturbation_features": conditions["perturbation_features"].astype(np.float32),
        "conditions_metadata": conditions["metadata_json"],
        "raw_metadata": raw["metadata_json"],
    }

def _validate_target_blind_sampling(
    config_path: str | Path,surface: dict,*,split: str
) -> dict | None:
    payload=json.loads(Path(config_path).read_text(encoding="utf-8"))
    expected=payload.get("training_policy",{}).get("target_blind_sampling")
    actual=surface["conditions_metadata"].get("label_free_sampling")
    if expected is None:
        return actual
    cap_key=(
        "maximum_train_examples_per_condition_group"
        if split == "train"
        else "maximum_validation_examples_per_condition_group"
    )
    if (
        not isinstance(actual,dict)
        or actual.get("method") != expected.get("method")
        or actual.get("seed") != expected.get("seed")
        or actual.get("max_examples_per_condition_group") != expected.get(cap_key)
        or actual.get("signal_access") is not False
    ):
        raise ValueError(f"adapter {split} target-blind sampling policy drifted")
    return actual

def _load_training_surface(
    raw_path: str | Path,
    truth_path: str | Path,
    conditions_path: str | Path,
    *,
    expected_split: str,
    config,
) -> dict:
    raw=_load_raw_enformer(raw_path)
    conditions=_load_conditions(conditions_path)
    aligned=_align_raw_conditions(raw,conditions,config)
    truth=_archive(truth_path,{"example_id","target","group","split"})
    truth_ids=_unique_ids(truth["example_id"],label="adapter truth")
    if set(truth_ids) != set(aligned["example_id"]) or len(truth_ids) != len(
        aligned["example_id"]
    ):
        raise ValueError("adapter truth, raw, and condition example-ID sets differ")
    condition_index={
        value: index for index,value in enumerate(aligned["example_id"])
    }
    order=np.asarray([condition_index[value] for value in truth_ids])
    aligned={
        key: value[order] if isinstance(value,np.ndarray) else value
        for key,value in aligned.items()
    }
    truth_group=truth["group"].astype(str)
    truth_split=truth["split"].astype(str)
    if not np.array_equal(truth_group,aligned["group"]):
        raise ValueError("adapter truth and condition group labels differ")
    if not np.array_equal(truth_split,aligned["split"]):
        raise ValueError("adapter truth and condition split labels differ")
    if set(truth_split) != {expected_split}:
        raise ValueError(
            f"adapter surface must contain only {expected_split!r}, got {sorted(set(truth_split))}"
        )
    target=truth["target"].astype(np.float32)
    if target.shape != aligned["prediction"].shape or not np.isfinite(target).all():
        raise ValueError("adapter target shape/values differ from Enformer predictions")
    mask=(
        truth["mask"].astype(bool)
        if "mask" in truth
        else np.ones_like(target,dtype=bool)
    )
    if mask.shape != target.shape or not mask.any():
        raise ValueError("adapter truth mask is empty or has the wrong shape")
    return {**aligned,"target": target,"mask": mask}

def _masked_log_loss(prediction,target,mask):
    import torch.nn.functional as F

    error=F.smooth_l1_loss(
        prediction.clamp_min(0).log1p(),
        target.clamp_min(0).log1p(),
        reduction="none",
    )
    weights=mask.to(error.dtype)
    return (error * weights).sum() / weights.sum().clamp_min(1.0)

def _group_validation_loss(model,surface: dict,device,batch_size: int) -> float:
    import torch

    row_scores=[]
    model.eval()
    with torch.no_grad():
        for start in range(0,len(surface["example_id"]),batch_size):
            stop=min(len(surface["example_id"]),start + batch_size)
            prediction=model(
                torch.from_numpy(surface["prediction"][start:stop]).to(device),
                torch.from_numpy(surface["assay_features"][start:stop]).to(device),
                torch.from_numpy(surface["state_features"][start:stop]).to(device),
                torch.from_numpy(surface["perturbation_features"][start:stop]).to(device),
            )
            target=torch.from_numpy(surface["target"][start:stop]).to(device)
            mask=torch.from_numpy(surface["mask"][start:stop]).to(device)
            absolute=(
                prediction.clamp_min(0).log1p() - target.clamp_min(0).log1p()
            ).abs()
            numerator=(absolute * mask).sum(dim=1)
            denominator=mask.sum(dim=1).clamp_min(1)
            row_scores.extend((numerator / denominator).cpu().numpy().tolist())
    row_scores=np.asarray(row_scores)
    groups=surface["group"]
    return float(
        np.mean([row_scores[groups == group].mean() for group in sorted(set(groups))])
    )

def train_enformer_adapter(
    config_path: str | Path,
    *,
    train_raw: str | Path,
    train_truth: str | Path,
    train_conditions: str | Path,
    validation_raw: str | Path,
    validation_truth: str | Path,
    validation_conditions: str | Path,
    out: str | Path,
    epochs: int = 30,
    batch_size: int = 16,
    learning_rate: float = 2e-4,
    weight_decay: float = 1e-4,
    seed: int = 20_260_620,
    device: str = "cpu",
) -> dict:

    import torch

    if epochs < 1 or batch_size < 1 or learning_rate <= 0 or weight_decay < 0:
        raise ValueError("invalid adapter optimization settings")
    config=load_adapter_config(config_path)
    validation_scope=_validation_scope(config_path)
    train=_load_training_surface(
        train_raw,
        train_truth,
        train_conditions,
        expected_split="train",
        config=config,
    )
    validation=_load_training_surface(
        validation_raw,
        validation_truth,
        validation_conditions,
        expected_split="validation",
        config=config,
    )
    train_sampling=_validate_target_blind_sampling(
        config_path,train,split="train"
    )
    validation_sampling=_validate_target_blind_sampling(
        config_path,validation,split="validation"
    )
    overlap=set(train["group"]) & set(validation["group"])
    example_overlap=set(train["example_id"]) & set(validation["example_id"])
    if example_overlap:
        raise ValueError("adapter train and validation example IDs overlap")
    if overlap and validation_scope == "group_disjoint":
        raise ValueError(f"adapter validation groups overlap training groups: {sorted(overlap)}")
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    actual_device=torch.device(
        "cuda" if device == "auto" and torch.cuda.is_available() else device
    )
    model=EnformerStateAdapter(config).to(actual_device)
    optimizer=torch.optim.AdamW(
        model.parameters(),lr=learning_rate,weight_decay=weight_decay
    )
    generator=torch.Generator().manual_seed(seed)
    tensors=torch.utils.data.TensorDataset(
        torch.from_numpy(train["prediction"]),
        torch.from_numpy(train["assay_features"]),
        torch.from_numpy(train["state_features"]),
        torch.from_numpy(train["perturbation_features"]),
        torch.from_numpy(train["target"]),
        torch.from_numpy(train["mask"]),
    )
    loader=torch.utils.data.DataLoader(
        tensors,
        batch_size=batch_size,
        shuffle=True,
        generator=generator,
        num_workers=0,
    )
    history=[]
    best_loss=float("inf")
    best_epoch=-1
    best_state=None
    for epoch in range(epochs):
        model.train()
        train_losses=[]
        for batch in loader:
            base,assay,state,perturbation,target,mask = [
                value.to(actual_device) for value in batch
            ]
            optimizer.zero_grad(set_to_none=True)
            prediction=model(base,assay,state,perturbation)
            loss=_masked_log_loss(prediction,target,mask)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(),1.0)
            optimizer.step()
            train_losses.append(float(loss.detach().cpu()))
        validation_loss=_group_validation_loss(
            model,validation,actual_device,batch_size
        )
        history.append(
            {
                "epoch": epoch + 1,
                "train_loss": float(np.mean(train_losses)),
                "group_validation_log_mae": validation_loss,
            }
        )
        if validation_loss < best_loss:
            best_loss=validation_loss
            best_epoch=epoch + 1
            best_state={
                key: value.detach().cpu().clone()
                for key,value in model.state_dict().items()
            }
    if best_state is None:
        raise RuntimeError("adapter training produced no checkpoint")
    checkpoint={
        "schema": "pdac-circuit.enformer-state-adapter-checkpoint/1",
        "config": asdict(config),
        "state_dict": best_state,
        "best_epoch": best_epoch,
        "best_group_validation_log_mae": best_loss,
        "history": history,
        "training_policy": {
            "train_split": "train",
            "validation_split": "validation",
            "train_groups": sorted(set(train["group"])),
            "validation_groups": sorted(set(validation["group"])),
            "validation_scope": validation_scope,
            "overlapping_validation_groups": sorted(overlap),
            "exact_example_id_disjoint": True,
            "target_blind_sampling": {
                "train": train_sampling,
                "validation": validation_sampling,
            },
            "candidate_feature_access": False,
            "test_label_access": False,
            "device": str(actual_device),
            "epochs": epochs,
            "batch_size": batch_size,
            "learning_rate": learning_rate,
            "weight_decay": weight_decay,
            "seed": seed,
        },
        "source_sha256": {
            "config": sha256_file(config_path),
            "train_raw": sha256_file(train_raw),
            "train_truth": sha256_file(train_truth),
            "train_conditions": sha256_file(train_conditions),
            "validation_raw": sha256_file(validation_raw),
            "validation_truth": sha256_file(validation_truth),
            "validation_conditions": sha256_file(validation_conditions),
            "adapter_code": sha256_file(Path(__file__)),
            "model_code": sha256_file(Path(__file__).with_name("baseline_adapter.py")),
        },
        "command": " ".join(sys.argv),
    }
    destination=Path(out)
    destination.parent.mkdir(parents=True,exist_ok=True)
    temporary=destination.parent / f".{destination.name}.partial-{uuid.uuid4().hex}"
    torch.save(checkpoint,temporary)
    temporary.replace(destination)
    return {
        "out": str(destination),
        "sha256": sha256_file(destination),
        "best_epoch": best_epoch,
        "best_group_validation_log_mae": best_loss,
        "train_examples": len(train["example_id"]),
        "validation_examples": len(validation["example_id"]),
        "train_groups": len(set(train["group"])),
        "validation_groups": len(set(validation["group"])),
        "validation_scope": validation_scope,
        "overlapping_validation_groups": len(overlap),
        "candidate_feature_access": False,
        "test_label_access": False,
    }

def predict_enformer_adapter(
    config_path: str | Path,
    *,
    checkpoint_path: str | Path,
    raw_path: str | Path,
    conditions_path: str | Path,
    out: str | Path,
    batch_size: int = 32,
    device: str = "cpu",
    ablate_intervention_residual: bool = False,
) -> dict:

    import torch

    if batch_size < 1:
        raise ValueError("adapter prediction batch_size must be positive")
    config=load_adapter_config(config_path)
    raw=_load_raw_enformer(raw_path)
    conditions=_load_conditions(conditions_path)
    surface=_align_raw_conditions(raw,conditions,config)
    if ablate_intervention_residual:
        surface["perturbation_features"]=np.zeros_like(
            surface["perturbation_features"],dtype=np.float32
        )
    actual_device=torch.device(
        "cuda" if device == "auto" and torch.cuda.is_available() else device
    )
    checkpoint=torch.load(checkpoint_path,map_location="cpu",weights_only=False)
    if checkpoint.get("schema") != "pdac-circuit.enformer-state-adapter-checkpoint/1":
        raise ValueError("invalid Enformer adapter checkpoint schema")
    if checkpoint.get("config") != asdict(config):
        raise ValueError("adapter checkpoint/config mismatch")
    if checkpoint.get("training_policy",{}).get("test_label_access") is not False:
        raise ValueError("adapter checkpoint does not certify zero test-label access")
    model=EnformerStateAdapter(config).to(actual_device)
    model.load_state_dict(checkpoint["state_dict"])
    model.eval()
    predictions=[]
    with torch.no_grad():
        for start in range(0,len(surface["example_id"]),batch_size):
            stop=min(len(surface["example_id"]),start + batch_size)
            predictions.append(
                model(
                    torch.from_numpy(surface["prediction"][start:stop]).to(actual_device),
                    torch.from_numpy(surface["assay_features"][start:stop]).to(actual_device),
                    torch.from_numpy(surface["state_features"][start:stop]).to(actual_device),
                    torch.from_numpy(surface["perturbation_features"][start:stop]).to(
                        actual_device
                    ),
                )
                .cpu()
                .numpy()
            )
    prediction=np.concatenate(predictions,axis=0)
    raw_metadata=raw["metadata_json"]
    checkpoint_sha=sha256_file(checkpoint_path)
    metadata={
        "schema": "pdac-circuit.raw-predictions/1",
        "model": "Enformer + grouped PDAC state adapter",
        "model_version": f"adapter-sha256:{checkpoint_sha}",
        "weights_sha256": checkpoint_sha,
        "parent_enformer_weights_sha256": raw_metadata.get("weights_sha256"),
        "track_mapping_sha256": raw_metadata.get("track_mapping_sha256"),
        "raw_enformer_sha256": sha256_file(raw_path),
        "conditions_sha256": sha256_file(conditions_path),
        "candidate_feature_access": False,
        "test_label_access": False,
        "ablate_state_residual": False,
        "ablate_intervention_residual": bool(ablate_intervention_residual),
        "intervention_reference_construction": (
            "exact_zero_registered_perturbation_vector"
            if ablate_intervention_residual
            else "registered_treatment_condition"
        ),
        "command": " ".join(sys.argv),
    }
    save_raw_predictions(
        out,
        model=metadata["model"],
        example_id=surface["example_id"],
        prediction=prediction,
        metadata=metadata,
    )
    return {
        "out": str(out),
        "examples": len(surface["example_id"]),
        "shape": list(prediction.shape),
        "sha256": sha256_file(out),
        **metadata,
    }
