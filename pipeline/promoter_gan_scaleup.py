from __future__ import annotations

import json

import numpy as np
import torch

from src.pdac_circuit.core.paths import MODELS, REGISTRY_JSON, RESULTS
from src.pdac_circuit.core.seeds import set_seeds, sha256_text, write_model_manifest
from src.pdac_circuit.generate.datamodule import build_real_promoter_onehot
from src.pdac_circuit.generate.evaluate import evaluate_gan, js_divergence, kmer_spectrum
from src.pdac_circuit.generate.promoter_gan import (
    PromoterGAN,
    build_critic,
    build_generator,
    samples_to_seqs,
)
from src.pdac_circuit.generate.training import _gradient_penalty
from src.pdac_circuit.parts.select import load_promoter_model
from src.pdac_circuit.stats import classify_cert

SEED=20260620
SEQ_LEN=1024
N_UNCAP=60000
GEN_ITERS=2500
BS=64
N_CRITIC=5
LAM=10.0


def _train(real: np.ndarray, dev):
    G=build_generator(128, SEQ_LEN).to(dev)
    D=build_critic(SEQ_LEN).to(dev)
    optG=torch.optim.Adam(G.parameters(), lr=1e-4, betas=(0.5, 0.9))
    optD=torch.optim.Adam(D.parameters(), lr=1e-4, betas=(0.5, 0.9))
    realT=torch.from_numpy(real).float().to(dev)
    n=realT.shape[0]
    rng=np.random.default_rng(SEED)
    g_torch=torch.Generator(device=dev)
    g_torch.manual_seed(SEED)
    real_spec=kmer_spectrum(samples_to_seqs(real[: min(1500, n)]))
    best_js, best_state = float("inf"), None

    def gen_js(n_eval=256):
        G.eval()
        with torch.no_grad():
            z=torch.randn(n_eval, 128, generator=g_torch, device=dev)
            oh=G(z, tau=0.5, hard=True).cpu().numpy()
        G.train()
        return js_divergence(kmer_spectrum(samples_to_seqs(oh)), real_spec)

    for it in range(GEN_ITERS):
        for _ in range(N_CRITIC):
            idx=torch.from_numpy(rng.integers(0, n, BS)).to(dev)
            real_b=realT[idx]
            z=torch.randn(BS, 128, generator=g_torch, device=dev)
            fake=G(z, tau=1.0).detach()
            gp=_gradient_penalty(D, real_b, fake, dev)
            d_loss=D(fake).mean() - D(real_b).mean() + LAM * gp
            optD.zero_grad(set_to_none=True)
            d_loss.backward()
            optD.step()
        z=torch.randn(BS, 128, generator=g_torch, device=dev)
        g_loss=-D(G(z, tau=1.0)).mean()
        optG.zero_grad(set_to_none=True)
        g_loss.backward()
        optG.step()
        if it >= 100 and it % 50 == 0:
            js=gen_js()
            if js < best_js:
                best_js=js
                best_state={k: v.detach().cpu().clone() for k, v in G.state_dict().items()}

    if best_state is not None:
        G.load_state_dict(best_state)
    gan=PromoterGAN(z_dim=128, seq_len=SEQ_LEN)
    gan.gen=G
    return gan, float(best_js)


def main() -> None:
    pre=json.loads(REGISTRY_JSON.read_text(encoding="utf-8"))["prereg"]["module_VII"]
    set_seeds(SEED)
    dev=torch.device("cuda" if torch.cuda.is_available() else "cpu")
    prom=load_promoter_model()
    real_ref=samples_to_seqs(build_real_promoter_onehot(n=4000, seq_len=SEQ_LEN, seed=SEED)[:2000])

    shipped=PromoterGAN.load(MODELS / "promoter_gan.pt")
    base=evaluate_gan(shipped, real_ref, prom, n=1500, seed=SEED)
    print(f"BASELINE (shipped 12k) js_gen {base['js_gen_vs_real']:.5f} | js_rand {base['js_random_vs_real']:.5f} "
          f"| p90 {base['pred_strength_gen_p90']:.4f} | uplift {base['strength_uplift']:.4f}")

    real=build_real_promoter_onehot(n=N_UNCAP, seq_len=SEQ_LEN, seed=SEED)
    print(f"uncapped real promoters: {len(real)} (shipped cap 12000, top-quartile pool)")
    gan, best_js = _train(real, dev)
    gan.gen=gan.gen.to("cpu")
    new=evaluate_gan(gan, real_ref, prom, n=1500, seed=SEED)
    print(f"SCALEUP  ({len(real)}) js_gen {new['js_gen_vs_real']:.5f} | js_rand {new['js_random_vs_real']:.5f} "
          f"| p90 {new['pred_strength_gen_p90']:.4f} | uplift {new['strength_uplift']:.4f}")

    beats_random=bool(new["realism_beats_random"])
    p90_ok=new["pred_strength_gen_p90"] >= pre["strength_p90_min"]
    certified=beats_random and (new["js_gen_vs_real"] <= pre["realism_js_max"]) and p90_ok
    stronger_tail=new["pred_strength_gen_p90"] >= base["pred_strength_gen_p90"]
    improved=certified and stronger_tail

    out={
        "schema": "pdac-circuit.gan-scaleup/1",
        "data_class": "REAL",
        "eval": "fixed real reference (2000 top-quartile FANTOM5 promoters); shipped + scaleup scored identically with the current promoter model",
        "n_train_uncapped": len(real),
        "shipped_cap": 12000,
        "baseline_js_gen_vs_real": base["js_gen_vs_real"],
        "baseline_js_random_vs_real": base["js_random_vs_real"],
        "baseline_p90": base["pred_strength_gen_p90"],
        "baseline_strength_uplift": base["strength_uplift"],
        "scaleup_js_gen_vs_real": new["js_gen_vs_real"],
        "scaleup_js_random_vs_real": new["js_random_vs_real"],
        "scaleup_p90": new["pred_strength_gen_p90"],
        "scaleup_strength_uplift": new["strength_uplift"],
        "delta_js_gen": float(new["js_gen_vs_real"] - base["js_gen_vs_real"]),
        "deployed": bool(improved),
        "gate": "certified-real (js_gen<=0.05, beats random, p90>=0.7) AND selectable tail p90 >= shipped; the pipeline selects the strongest generated promoter",
    }

    if improved:
        gan.save(MODELS / "promoter_gan.pt")
        fix_seqs=gan.generate(8, seed=0)
        (MODELS / "promoter_gan.fixture.json").write_text(
            json.dumps({"schema": "pdac-circuit.fixture/1", "model_key": "promoter_gan",
                        "seqs_sha256": sha256_text("".join(fix_seqs)), "n": 8, "seq_len": SEQ_LEN}, indent=2),
            encoding="utf-8",
        )
        realism_ok=(new["js_gen_vs_real"] <= pre["realism_js_max"]) and beats_random
        cert=classify_cert(positive_significant=realism_ok and p90_ok,
                             exceeds_margin=realism_ok and p90_ok, powered=True)
        write_model_manifest(MODELS / "promoter_gan.model.json", model_key="promoter_gan", module="VII", arch="wgan-gp",
                             weight_path=MODELS / "promoter_gan.pt", metrics=new,
                             data_lineage=["fantom5-cage", "hg38-ref"], seed=SEED, extra={"cert": cert})
        print(f"DEPLOYED: js_gen {base['js_gen_vs_real']:.5f} -> {new['js_gen_vs_real']:.5f} "
              f"({out['delta_js_gen']:+.5f}); p90 {new['pred_strength_gen_p90']:.4f}")
    else:
        print(f"NOT DEPLOYED: scaleup js {new['js_gen_vs_real']:.5f} vs shipped {base['js_gen_vs_real']:.5f}, "
              f"p90 {new['pred_strength_gen_p90']:.4f}; shipped kept.")

    RESULTS.mkdir(parents=True, exist_ok=True)
    (RESULTS / "promoter_gan_scaleup.json").write_text(json.dumps(out, indent=2), encoding="utf-8")
    print("wrote results/promoter_gan_scaleup.json")


if __name__ == "__main__":
    main()
