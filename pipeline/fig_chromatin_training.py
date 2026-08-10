import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
curve_p = ROOT / "results" / "chromatin_training_curve.json"
ens_p = ROOT / "results" / "chromatin_ensemble.json"
out = ROOT / "figures" / "fig_chromatin_training.png"

fig, ax = plt.subplots(1, 2, figsize=(11, 4.2))

if curve_p.exists():
    c = json.loads(curve_p.read_text())["curve"]
    steps = [r["optimizer_step"] for r in c if r.get("best_validation_loss") is not None]
    loss = [r["best_validation_loss"] for r in c if r.get("best_validation_loss") is not None]
    ax[0].plot(steps, loss, "-o", ms=3, color="#5b4bc4")
    ax[0].set_xlabel("optimizer step")
    ax[0].set_ylabel("best held-out profile loss")
    ax[0].set_title("Healthy-prior training (196,608 bp)\n33,156 real ENCODE shards")
    ax[0].grid(alpha=0.3)

if ens_p.exists():
    e = json.loads(ens_p.read_text())
    corrs = [m["val_profile_correlation"] for m in e.get("members", [])
             if m.get("val_profile_correlation") is not None]
    if corrs:
        ax[1].bar(range(1, len(corrs) + 1), corrs, color="#0f766e")
        mu, sd = float(np.mean(corrs)), float(np.std(corrs))
        ax[1].axhline(mu, color="#b45309", lw=2, label=f"mean {mu:.4f} ± {sd:.4f}")
        ax[1].set_ylim(0.5, 0.8)
        ax[1].set_xlabel("training seed (independent run)")
        ax[1].set_ylabel("held-out profile correlation")
        ax[1].set_title(f"Multi-seed reproducibility (n={len(corrs)})")
        ax[1].legend(fontsize=8)
        ax[1].grid(alpha=0.3, axis="y")

fig.suptitle("Long-range PDACircuitFormer — first real training on healthy pancreas chromatin", fontsize=11)
fig.tight_layout(rect=(0, 0, 1, 0.94))
out.parent.mkdir(parents=True, exist_ok=True)
fig.savefig(out, dpi=130)
print("wrote", out)
