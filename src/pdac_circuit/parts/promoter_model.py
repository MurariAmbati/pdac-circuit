from __future__ import annotations

import json
from pathlib import Path

import numpy as np

def build_promoter_cnn(seq_len: int = 1000,n_aux: int = 2):
    import torch
    import torch.nn as nn

    class PromoterCNN(nn.Module):
        def __init__(self):
            super().__init__()
            self.conv=nn.Sequential(
                nn.Conv1d(4,128,15,padding=7),nn.BatchNorm1d(128),nn.ReLU(),nn.Dropout(0.2),
                nn.Conv1d(128,128,11,padding=5),nn.BatchNorm1d(128),nn.ReLU(),nn.MaxPool1d(4),
                nn.Conv1d(128,256,7,padding=6,dilation=2),nn.BatchNorm1d(256),nn.ReLU(),nn.MaxPool1d(4),
                nn.Conv1d(256,256,5,padding=8,dilation=4),nn.BatchNorm1d(256),nn.ReLU(),
            )
            self.pool=nn.AdaptiveAvgPool1d(1)
            self.head=nn.Sequential(
                nn.Linear(256 + n_aux,128),nn.ReLU(),nn.Dropout(0.3),nn.Linear(128,1),
            )

        def forward(self,x,aux=None):
            h=self.pool(self.conv(x)).squeeze(-1)
            if aux is not None:
                h=torch.cat([h,aux],dim=1)
            else:
                h=torch.cat([h,h.new_zeros(h.shape[0],self.head[0].in_features - h.shape[1])],dim=1)
            return self.head(h).squeeze(-1)

    return PromoterCNN()

class PromoterModel:

    def __init__(self,seq_len: int = 1000):
        self.seq_len=seq_len
        self.cnn=None
        self.rf=None
        self.cdf_x=None

    def fit_rf(self,kmer_feats,aux_feats,y):
        from sklearn.ensemble import HistGradientBoostingRegressor

        X=np.concatenate([kmer_feats,aux_feats],axis=1)
        self.rf=HistGradientBoostingRegressor(max_iter=300,learning_rate=0.06,max_depth=8,random_state=20260620)
        self.rf.fit(X,y)

    def rf_predict(self,kmer_feats,aux_feats) -> np.ndarray:
        return self.rf.predict(np.concatenate([kmer_feats,aux_feats],axis=1))

    def set_cdf(self,train_scores):
        self.cdf_x=np.sort(np.asarray(train_scores,dtype=float))

    def to_unit(self,scores) -> np.ndarray:
        if self.cdf_x is None:
            return np.clip(scores,0,1)
        return np.searchsorted(self.cdf_x,scores,side="right") / max(len(self.cdf_x),1)

    def save(self,cnn_path: Path):
        import joblib
        import torch

        if self.cnn is not None:
            torch.save(self.cnn.state_dict(),cnn_path)
        meta={"seq_len": self.seq_len,"cdf_x": self.cdf_x.tolist() if self.cdf_x is not None else None}
        Path(str(cnn_path) + ".meta.json").write_text(json.dumps(meta),encoding="utf-8")
        if self.rf is not None:
            joblib.dump(self.rf,str(cnn_path).replace(".pt",".rf.joblib"))

    @classmethod
    def load(cls,cnn_path: Path) -> "PromoterModel":
        import joblib
        import torch

        meta=json.loads(Path(str(cnn_path) + ".meta.json").read_text(encoding="utf-8"))
        m=cls(seq_len=meta["seq_len"])
        m.cnn=build_promoter_cnn(meta["seq_len"])
        m.cnn.load_state_dict(torch.load(cnn_path,map_location="cpu"))
        m.cnn.eval()
        if meta.get("cdf_x"):
            m.cdf_x=np.array(meta["cdf_x"])
        rf_path=str(cnn_path).replace(".pt",".rf.joblib")
        if Path(rf_path).exists():
            m.rf=joblib.load(rf_path)
        return m
