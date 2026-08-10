from __future__ import annotations

from dataclasses import dataclass,field

import pandas as pd

from ..core.provenance import REAL,reduce_data_class

@dataclass
class AnnotatedFeatureMatrix:
    table: pd.DataFrame
    provenance: dict[str,str] = field(default_factory=dict)
    data_classes: list[str] = field(default_factory=lambda: [REAL])
    caveats: list[str] = field(default_factory=list)

    @property
    def data_class(self) -> str:
        return reduce_data_class(self.data_classes)

    def top(self,score_col: str = "composite",k: int = 10) -> pd.DataFrame:
        return self.table.sort_values(score_col,ascending=False).head(k)

    def genes(self) -> list[str]:
        return list(self.table.index)
