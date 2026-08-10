import glob
import json
import sys
from pathlib import Path

def main(ckpt_dir: str,out: str) -> None:
    import torch

    files = sorted(glob.glob(str(Path(ckpt_dir) / "step-*.pt")))
    curve = []
    for f in files:
        try:
            ck = torch.load(f,map_location="cpu",weights_only=False)
        except Exception:
            continue
        lv = ck.get("last_validation")
        row = {
            "global_step": ck.get("global_step"),
            "optimizer_step": ck.get("optimizer_step"),
            "epoch": ck.get("epoch"),
            "best_validation_loss": (None if ck.get("best_validation_loss") in (None,float("inf"))
                                     else round(float(ck["best_validation_loss"]),6)),
        }
        if isinstance(lv,dict):
            for k,v in lv.items():
                if isinstance(v,(int,float)):
                    row[f"val_{k}"] = round(float(v),6)
        curve.append(row)
    report = {
        "schema": "pdac-circuit.chromatin-training-curve/1",
        "checkpoint_dir": ckpt_dir,
        "n_checkpoints": len(curve),
        "final_step": curve[-1]["global_step"] if curve else 0,
        "best_validation_loss": min((r["best_validation_loss"] for r in curve
                                     if r["best_validation_loss"] is not None),default=None),
        "curve": curve,
    }
    Path(out).write_text(json.dumps(report,indent=2))
    print(f"wrote {out}: {len(curve)} checkpoints, final step {report['final_step']}, "
          f"best val loss {report['best_validation_loss']}")

if __name__ == "__main__":
    main(sys.argv[1],sys.argv[2])
