from __future__ import annotations

import json
from pathlib import Path

def build_enhancer_cnn(seq_len: int = 2000):
    import torch
    import torch.nn as nn

    class EnhancerCNN(nn.Module):
        def __init__(self):
            super().__init__()
            self.conv=nn.Sequential(
                nn.Conv1d(4,128,19,padding=9),nn.BatchNorm1d(128),nn.ReLU(),nn.MaxPool1d(4),
                nn.Conv1d(128,128,11,padding=10,dilation=2),nn.BatchNorm1d(128),nn.ReLU(),nn.MaxPool1d(4),
                nn.Conv1d(128,256,7,padding=12,dilation=4),nn.BatchNorm1d(256),nn.ReLU(),nn.MaxPool1d(4),
                nn.Conv1d(256,256,5,padding=16,dilation=8),nn.BatchNorm1d(256),nn.ReLU(),
            )
            self.pool=nn.AdaptiveAvgPool1d(1)
            self.trunk=nn.Sequential(nn.Linear(256,128),nn.ReLU(),nn.Dropout(0.3))
            self.cls_head=nn.Linear(128,1)
            self.sig_head=nn.Linear(128,1)

        def forward(self,x,aux=None):
            h=self.pool(self.conv(x)).squeeze(-1)
            h=self.trunk(h)
            return torch.cat([self.cls_head(h),self.sig_head(h)],dim=1)

    return EnhancerCNN()

class EnhancerModel:
    def __init__(self,seq_len: int = 2000):
        self.seq_len=seq_len
        self.cnn=None

    def save(self,path: Path):
        import torch

        torch.save(self.cnn.state_dict(),path)
        Path(str(path) + ".meta.json").write_text(json.dumps({"seq_len": self.seq_len}),encoding="utf-8")

    @classmethod
    def load(cls,path: Path) -> "EnhancerModel":
        import torch

        meta=json.loads(Path(str(path) + ".meta.json").read_text(encoding="utf-8"))
        m=cls(seq_len=meta["seq_len"])
        m.cnn=build_enhancer_cnn(meta["seq_len"])
        m.cnn.load_state_dict(torch.load(path,map_location="cpu"))
        m.cnn.eval()
        return m
