from __future__ import annotations

import json
import sys
import time
from pathlib import Path

from pdac_circuit.chromatin.cli import run_train
from pdac_circuit.core.paths import RESULTS

OUT=RESULTS / "chromatin_ensemble.json"
CFG="configs/chromatin-ensemble.json"
SHARDS="data/processed/chromatin_encode_healthy_full_v1/*/*.npz"
FASTA="data/raw/hg38-ref/hg38.fa"
N_SEEDS=int(sys.argv[1]) if len(sys.argv) > 1 else 24
T0=time.time()

def final_metrics(ckpt_dir: str) -> dict:
    import torch

    latest=Path(ckpt_dir) / "latest.pt"
    if not latest.exists():
        return {}
    c=torch.load(latest,map_location="cpu",weights_only=False)
    lv=c.get("last_validation") or {}
    parts=lv.get("parts",{}) if isinstance(lv,dict) else {}
    return {
        "optimizer_step": c.get("optimizer_step"),
        "best_validation_loss": (None if c.get("best_validation_loss") in (None,float("inf"))
                                 else round(float(c["best_validation_loss"]),6)),
        "val_profile_loss": round(float(parts.get("profile")),6) if "profile" in parts else None,
        "val_profile_correlation": round(float(parts.get("correlation")),6) if "correlation" in parts else None,
    }

def main():
    if OUT.exists():
        state=json.loads(OUT.read_text())
        state["n_seeds_requested"]=N_SEEDS
    else:
        state={"schema": "pdac-circuit.chromatin-ensemble/1","data_class": "REAL",
                 "n_seeds_requested": N_SEEDS,"shards": SHARDS,"members": []}
    completed={m["seed"] for m in state["members"] if m.get("val_profile_correlation") is not None}
    state["members"]=[m for m in state["members"] if m.get("val_profile_correlation") is not None]
    OUT.write_text(json.dumps(state,indent=2))
    print(f"resuming: {len(completed)} seeds already complete",flush=True)
    for i in range(N_SEEDS):
        seed=20260620 + i * 101
        if seed in completed:
            continue
        ckpt=f"models/chromatin/ensemble/seed_{seed}"
        t=time.time()
        m=None
        for attempt in range(2):
            try:
                run_train(
                    config_path=CFG,
                    shards_glob=SHARDS,
                    checkpoint_dir=ckpt,
                    fasta_path=FASTA,
                    device="cuda",
                    resume=False,
                    minimum_free_gb=8.0,
                    allow_low_vram=True,
                    stage="healthy_prior",
                    seed=seed,
                )
                m=final_metrics(ckpt)
                break
            except PermissionError as e:
                print(f"[{time.time()-T0:8.1f}s] seed {seed} file-lock race (attempt {attempt+1}): {e}",
                      flush=True)
                time.sleep(5)
                m=final_metrics(ckpt) or None
                if m:
                    break
            except Exception as e:
                state["members"].append({"seed": seed,"error": f"{type(e).__name__}: {e}"})
                print(f"[{time.time()-T0:8.1f}s] seed {seed} ERROR: {e}",flush=True)
                m=None
                break
        if m and m.get("val_profile_correlation") is not None:
            m["seed"]=seed
            m["seconds"]=round(time.time() - t,1)
            state["members"].append(m)
            print(f"[{time.time()-T0:8.1f}s] seed {seed}: profile_corr={m.get('val_profile_correlation')} "
                  f"profile_loss={m.get('val_profile_loss')} step={m.get('optimizer_step')} [{m['seconds']}s]",
                  flush=True)
        corrs=[m["val_profile_correlation"] for m in state["members"]
                 if m.get("val_profile_correlation") is not None]
        if corrs:
            import numpy as np

            state["aggregate"]={
                "n_completed": len(corrs),
                "profile_correlation_mean": round(float(np.mean(corrs)),4),
                "profile_correlation_std": round(float(np.std(corrs)),4),
                "profile_correlation_min": round(float(np.min(corrs)),4),
                "profile_correlation_max": round(float(np.max(corrs)),4),
            }
        state["elapsed_seconds"]=round(time.time() - T0,1)
        OUT.write_text(json.dumps(state,indent=2))
    print(f"ensemble complete: {state.get('aggregate')} in {state['elapsed_seconds']}s",flush=True)

if __name__ == "__main__":
    main()
