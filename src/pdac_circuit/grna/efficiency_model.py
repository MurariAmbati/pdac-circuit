from __future__ import annotations

import json
from pathlib import Path

import numpy as np

def build_grna_cnn(seq_len: int = 30):
    import torch.nn as nn

    class GuideCNN(nn.Module):
        def __init__(self):
            super().__init__()
            self.conv=nn.Sequential(
                nn.Conv1d(4, 80, 3, padding=1), nn.ReLU(),
                nn.Conv1d(80, 60, 5, padding=2), nn.ReLU(),
                nn.Conv1d(60, 60, 7, padding=3), nn.ReLU(),
            )
            self.pool=nn.AdaptiveMaxPool1d(1)
            self.head=nn.Sequential(nn.Linear(60, 60), nn.ReLU(), nn.Dropout(0.3), nn.Linear(60, 1))

        def forward(self, x, aux=None):
            h=self.pool(self.conv(x)).squeeze(-1)
            return self.head(h).squeeze(-1)

    return GuideCNN()

class GRNAModel:
    def __init__(self, seq_len: int = 30):
        self.seq_len=seq_len
        self.cnn=None
        self.gbm=None
        self.cdf_x=None

    def fit_gbm(self, feats, y):
        from sklearn.ensemble import HistGradientBoostingRegressor

        self.gbm=HistGradientBoostingRegressor(max_iter=400, learning_rate=0.05, max_depth=6, random_state=20260620)
        self.gbm.fit(feats, y)

    def set_cdf(self, scores):
        self.cdf_x=np.sort(np.asarray(scores, dtype=float))

    def to_unit(self, scores) -> np.ndarray:
        if self.cdf_x is None:
            return np.clip(scores, 0, 1)
        return np.searchsorted(self.cdf_x, scores, side="right") / max(len(self.cdf_x), 1)

    def save(self, cnn_path: Path):
        import joblib
        import torch

        torch.save(self.cnn.state_dict(), cnn_path)
        Path(str(cnn_path) + ".meta.json").write_text(
            json.dumps({"seq_len": self.seq_len, "cdf_x": self.cdf_x.tolist() if self.cdf_x is not None else None}),
            encoding="utf-8",
        )
        if self.gbm is not None:
            joblib.dump(self.gbm, str(cnn_path).replace(".pt", ".gbm.joblib"))

    @classmethod
    def load(cls, cnn_path: Path) -> "GRNAModel":
        import joblib
        import torch

        meta=json.loads(Path(str(cnn_path) + ".meta.json").read_text(encoding="utf-8"))
        m=cls(seq_len=meta["seq_len"])
        m.cnn=build_grna_cnn(meta["seq_len"])
        m.cnn.load_state_dict(torch.load(cnn_path, map_location="cpu"))
        m.cnn.eval()
        if meta.get("cdf_x"):
            m.cdf_x=np.array(meta["cdf_x"])
        gp=str(cnn_path).replace(".pt", ".gbm.joblib")
        if Path(gp).exists():
            m.gbm=joblib.load(gp)
        return m
