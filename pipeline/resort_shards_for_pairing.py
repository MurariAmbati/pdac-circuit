from __future__ import annotations

import glob as globmod
import sys
from pathlib import Path

import numpy as np

from pdac_circuit.chromatin.pairing import _coordinate_key

def main(pattern: str, out_dir: str, windows_per_shard: int = 64) -> None:
    paths=sorted(Path(p) for p in globmod.glob(pattern, recursive=True))
    if not paths:
        raise FileNotFoundError(f"no shards matched {pattern!r}")
    rows=[]
    fields: list[str]=[]
    for p in paths:
        with np.load(p, allow_pickle=False) as shard:
            if not fields:
                fields=list(shard.files)
            for i in range(len(shard["start"])):
                key=_coordinate_key(str(shard["chrom"][i]), int(shard["start"][i]), int(shard["end"][i]))
                rows.append((key, {f: shard[f][i].copy() for f in shard.files}))
    rows.sort(key=lambda r: r[0])
    out=Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    n=0
    for s in range(0, len(rows), windows_per_shard):
        chunk=[r[1] for r in rows[s : s + windows_per_shard]]
        payload={f: np.stack([c[f] for c in chunk]) for f in fields}
        np.savez(out / f"shard-{n:06d}.npz", **payload)
        n += 1
    first, last=rows[0][0], rows[-1][0]
    print(f"resorted {len(rows)} windows from {len(paths)} shards -> {n} shards in {out}")
    print(f"  first key {first}  last key {last}")

if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
