from __future__ import annotations

import json

import numpy as np

from ..core.paths import MODELS,REGISTRY_JSON,RESULTS
from ..core.seeds import set_seeds,sha256_text,write_model_manifest
from ..stats import classify_cert

def _gradient_penalty(critic,real,fake,device):
    import torch

    b=real.shape[0]
    alpha=torch.rand(b,1,1,device=device)
    interp=(alpha * real + (1 - alpha) * fake).requires_grad_(True)
    d=critic(interp)
    grads=torch.autograd.grad(d.sum(),interp,create_graph=True)[0]
    return ((grads.flatten(1).norm(2,dim=1) - 1) ** 2).mean()

def train_promoter_gan(*,quick: bool = False,seed: int = 20260620) -> dict:
    import torch

    pre=json.loads(REGISTRY_JSON.read_text(encoding="utf-8"))["prereg"]["module_VII"]
    seq_len=int(pre["gan_seq_len"])
    set_seeds(seed)
    from .datamodule import build_real_promoter_onehot
    from .evaluate import evaluate_gan
    from .promoter_gan import PromoterGAN,build_critic,build_generator

    real=build_real_promoter_onehot(n=3000 if quick else 12000,seq_len=seq_len,seed=seed)
    real_seqs_for_eval=None
    from .promoter_gan import samples_to_seqs

    real_seqs_for_eval=samples_to_seqs(real[: min(2000,len(real))])

    dev=torch.device("cuda" if torch.cuda.is_available() else "cpu")
    G=build_generator(128,seq_len).to(dev)
    D=build_critic(seq_len).to(dev)
    optG=torch.optim.Adam(G.parameters(),lr=1e-4,betas=(0.5,0.9))
    optD=torch.optim.Adam(D.parameters(),lr=1e-4,betas=(0.5,0.9))

    realT=torch.from_numpy(real).float().to(dev)
    n=realT.shape[0]
    bs=64
    n_critic=5
    gen_iters=400 if quick else 2500
    lam=10.0
    rng=np.random.default_rng(seed)
    g_torch=torch.Generator(device=dev)
    g_torch.manual_seed(seed)

    from .evaluate import js_divergence,kmer_spectrum
    from .promoter_gan import samples_to_seqs

    real_spec=kmer_spectrum(samples_to_seqs(real[: min(1500,n)]))
    eval_every=50
    warmup=100
    best_js,best_state = float("inf"),None

    def _gen_js(n_eval=256):
        G.eval()
        with torch.no_grad():
            z=torch.randn(n_eval,128,generator=g_torch,device=dev)
            oh=G(z,tau=0.5,hard=True).cpu().numpy()
        G.train()
        return js_divergence(kmer_spectrum(samples_to_seqs(oh)),real_spec)

    for it in range(gen_iters):
        for _ in range(n_critic):
            idx=torch.from_numpy(rng.integers(0,n,bs)).to(dev)
            real_b=realT[idx]
            z=torch.randn(bs,128,generator=g_torch,device=dev)
            fake=G(z,tau=1.0).detach()
            d_real=D(real_b).mean()
            d_fake=D(fake).mean()
            gp=_gradient_penalty(D,real_b,fake,dev)
            d_loss=d_fake - d_real + lam * gp
            optD.zero_grad(set_to_none=True)
            d_loss.backward()
            optD.step()
        z=torch.randn(bs,128,generator=g_torch,device=dev)
        fake=G(z,tau=1.0)
        g_loss=-D(fake).mean()
        optG.zero_grad(set_to_none=True)
        g_loss.backward()
        optG.step()
        if it >= warmup and it % eval_every == 0:
            js=_gen_js()
            if js < best_js:
                best_js=js
                best_state={k: v.detach().cpu().clone() for k,v in G.state_dict().items()}

    if best_state is not None:
        G.load_state_dict(best_state)
    gan=PromoterGAN(z_dim=128,seq_len=seq_len)
    gan.gen=G

    from ..parts.select import load_promoter_model

    prom_model=load_promoter_model()
    metrics=evaluate_gan(gan,real_seqs_for_eval,prom_model,n=500 if quick else 1500,seed=seed)

    realism_ok=(metrics["js_gen_vs_real"] <= pre["realism_js_max"]) and metrics["realism_beats_random"]
    p90=metrics.get("pred_strength_gen_p90",0.0)
    usable_ok=p90 >= pre["strength_p90_min"]
    cert=classify_cert(positive_significant=realism_ok and usable_ok,
                         exceeds_margin=realism_ok and usable_ok,powered=n >= 500)

    gan_path=MODELS / "promoter_gan.pt"
    gan.save(gan_path)
    gan.gen=gan.gen.to("cpu")
    fix_seqs=gan.generate(8,seed=0)
    (MODELS / "promoter_gan.fixture.json").write_text(
        json.dumps({"schema": "pdac-circuit.fixture/1","model_key": "promoter_gan",
                    "seqs_sha256": sha256_text("".join(fix_seqs)),"n": 8,"seq_len": seq_len},indent=2),
        encoding="utf-8",
    )
    write_model_manifest(MODELS / "promoter_gan.model.json",model_key="promoter_gan",module="VII",arch="wgan-gp",
                         weight_path=gan_path,metrics=metrics,data_lineage=["fantom5-cage","hg38-ref"],seed=seed,extra={"cert": cert})
    out={"model": "promoter_gan","cert": cert,"n_train": int(n),"gen_iters": gen_iters,**{k: v for k,v in metrics.items() if isinstance(v,(int,float,bool))}}
    RESULTS.mkdir(parents=True,exist_ok=True)
    (RESULTS / "promoter_gan.train.json").write_text(json.dumps(out,indent=2,default=float),encoding="utf-8")
    return out

def recertify_gan(*,seed: int = 20260620) -> dict:
    pre=json.loads(REGISTRY_JSON.read_text(encoding="utf-8"))["prereg"]["module_VII"]
    from .datamodule import build_real_promoter_onehot
    from .evaluate import evaluate_gan
    from .promoter_gan import PromoterGAN,samples_to_seqs

    gan=PromoterGAN.load(MODELS / "promoter_gan.pt")
    real=build_real_promoter_onehot(n=4000,seq_len=gan.seq_len,seed=seed)
    real_seqs=samples_to_seqs(real[: min(2000,len(real))])
    from ..parts.select import load_promoter_model

    metrics=evaluate_gan(gan,real_seqs,load_promoter_model(),n=1500,seed=seed)
    realism_ok=(metrics["js_gen_vs_real"] <= pre["realism_js_max"]) and metrics["realism_beats_random"]
    usable_ok=metrics.get("pred_strength_gen_p90",0.0) >= pre["strength_p90_min"]
    cert=classify_cert(positive_significant=realism_ok and usable_ok,exceeds_margin=realism_ok and usable_ok,powered=True)
    fix_seqs=gan.generate(8,seed=0)
    (MODELS / "promoter_gan.fixture.json").write_text(
        json.dumps({"schema": "pdac-circuit.fixture/1","model_key": "promoter_gan",
                    "seqs_sha256": sha256_text("".join(fix_seqs)),"n": 8,"seq_len": gan.seq_len},indent=2),
        encoding="utf-8",
    )
    write_model_manifest(MODELS / "promoter_gan.model.json",model_key="promoter_gan",module="VII",arch="wgan-gp",
                         weight_path=MODELS / "promoter_gan.pt",metrics=metrics,data_lineage=["fantom5-cage","hg38-ref"],seed=seed,extra={"cert": cert})
    out={"model": "promoter_gan","cert": cert,**{k: v for k,v in metrics.items() if isinstance(v,(int,float,bool))}}
    (RESULTS / "promoter_gan.train.json").write_text(json.dumps(out,indent=2,default=float),encoding="utf-8")
    return out
