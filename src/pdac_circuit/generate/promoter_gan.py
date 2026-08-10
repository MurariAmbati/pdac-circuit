from __future__ import annotations

import json
from pathlib import Path

import numpy as np

def build_generator(z_dim: int = 128, seq_len: int = 1024, base_ch: int = 256):
    import torch.nn as nn

    n_up=int(np.log2(seq_len // 16))

    class Generator(nn.Module):
        def __init__(self):
            super().__init__()
            self.init_len=16
            self.fc=nn.Linear(z_dim, base_ch * self.init_len)
            chs=[base_ch]
            blocks=[]
            ch=base_ch
            for i in range(n_up):
                out=max(32, ch // 2)
                blocks += [
                    nn.ConvTranspose1d(ch, out, 4, stride=2, padding=1),
                    nn.BatchNorm1d(out),
                    nn.ReLU(inplace=True),
                ]
                ch=out
                chs.append(ch)
            self.up=nn.Sequential(*blocks)
            self.to_logits=nn.Conv1d(ch, 4, 7, padding=3)

        def logits_of(self, z):
            h=self.fc(z).view(z.shape[0], -1, self.init_len)
            h=self.up(h)
            return self.to_logits(h)

        def forward(self, z, tau: float = 1.0, hard: bool = False):
            import torch.nn.functional as F

            return F.gumbel_softmax(self.logits_of(z), tau=tau, hard=hard, dim=1)

    return Generator()

def build_critic(seq_len: int = 1024, base_ch: int = 32):
    import torch.nn as nn

    n_down=int(np.log2(seq_len // 16))

    class Critic(nn.Module):
        def __init__(self):
            super().__init__()
            blocks=[]
            ch_in=4
            ch=base_ch
            for i in range(n_down):
                blocks += [
                    nn.Conv1d(ch_in, ch, 4, stride=2, padding=1),
                    nn.InstanceNorm1d(ch, affine=True),
                    nn.LeakyReLU(0.2, inplace=True),
                ]
                ch_in=ch
                ch=min(256, ch * 2)
            self.down=nn.Sequential(*blocks)
            self.fc=nn.Linear(ch_in * 16, 1)

        def forward(self, x):
            h=self.down(x)
            return self.fc(h.flatten(1)).squeeze(-1)

    return Critic()

_BASES="ACGT"

def samples_to_seqs(soft_onehot: np.ndarray) -> list[str]:
    idx=np.argmax(soft_onehot, axis=1)
    return ["".join(_BASES[i] for i in row) for row in idx]

class PromoterGAN:
    def __init__(self, z_dim: int = 128, seq_len: int = 1024):
        self.z_dim=z_dim
        self.seq_len=seq_len
        self.gen=None

    def generate(self, n: int, *, seed: int = 0, batch: int = 256) -> list[str]:
        import torch

        self.gen.eval().to("cpu")
        g=torch.Generator().manual_seed(seed)
        seqs: list[str] = []
        with torch.no_grad():
            for i in range(0, n, batch):
                b=min(batch, n - i)
                z=torch.randn(b, self.z_dim, generator=g)
                logits=self.gen.logits_of(z)
                u=torch.rand(logits.shape, generator=g).clamp_(1e-9, 1 - 1e-9)
                gumbel=-torch.log(-torch.log(u))
                idx=torch.argmax(logits + gumbel, dim=1).numpy()
                seqs += ["".join(_BASES[k] for k in row) for row in idx]
        return seqs

    def save(self, path: Path):
        import torch

        torch.save(self.gen.state_dict(), path)
        Path(str(path) + ".meta.json").write_text(json.dumps({"z_dim": self.z_dim, "seq_len": self.seq_len}), encoding="utf-8")

    @classmethod
    def load(cls, path: Path) -> "PromoterGAN":
        import torch

        meta=json.loads(Path(str(path) + ".meta.json").read_text(encoding="utf-8"))
        m=cls(z_dim=meta["z_dim"], seq_len=meta["seq_len"])
        m.gen=build_generator(meta["z_dim"], meta["seq_len"])
        m.gen.load_state_dict(torch.load(path, map_location="cpu"))
        m.gen.eval()
        return m
