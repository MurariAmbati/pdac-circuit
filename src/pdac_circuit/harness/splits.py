from __future__ import annotations

from dataclasses import dataclass

import numpy as np

@dataclass(frozen=True)
class ChromSplit:
    train_idx: np.ndarray
    val_idx: np.ndarray
    test_idx: np.ndarray
    test_chroms: tuple[str, ...]
    val_chroms: tuple[str, ...]

def chromosome_held_out_split(
    chroms, *, test_chroms=("chr8", "chr9"), val_chroms=("chr7",)
) -> ChromSplit:
    chroms=np.asarray([str(c) for c in chroms])
    test_set=set(test_chroms)
    val_set=set(val_chroms)
    test_idx=np.where(np.isin(chroms, list(test_set)))[0]
    val_idx=np.where(np.isin(chroms, list(val_set)))[0]
    train_idx=np.where(~np.isin(chroms, list(test_set | val_set)))[0]
    return ChromSplit(train_idx, val_idx, test_idx, tuple(test_chroms), tuple(val_chroms))
