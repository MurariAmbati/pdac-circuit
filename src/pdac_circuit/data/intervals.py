from __future__ import annotations

import gzip
from bisect import bisect_left
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator

@dataclass(frozen=True)
class Interval:
    chrom: str
    start: int
    end: int
    name: str = "."
    score: float = 0.0
    strand: str = "."
    signal: float = 0.0

def _open(path: Path):
    return gzip.open(path, "rt") if str(path).endswith(".gz") else open(path)

def read_narrowpeak(path: str | Path) -> list[Interval]:
    out: list[Interval] = []
    with _open(Path(path)) as f:
        for line in f:
            if not line or line.startswith(("#", "track", "browser")):
                continue
            p=line.rstrip("\n").split("\t")
            if len(p) < 3:
                continue
            signal=float(p[6]) if len(p) > 6 and p[6] not in (".", "") else 0.0
            score=float(p[4]) if len(p) > 4 and p[4] not in (".", "") else 0.0
            out.append(
                Interval(p[0], int(p[1]), int(p[2]),
                         name=p[3] if len(p) > 3 else ".",
                         score=score,
                         strand=p[5] if len(p) > 5 else ".",
                         signal=signal)
            )
    return out

def read_bed(path: str | Path) -> list[Interval]:
    out: list[Interval] = []
    with _open(Path(path)) as f:
        for line in f:
            if not line or line.startswith(("#", "track", "browser")):
                continue
            p=line.rstrip("\n").split("\t")
            if len(p) < 3:
                continue
            out.append(
                Interval(p[0], int(p[1]), int(p[2]),
                         name=p[3] if len(p) > 3 else ".",
                         score=float(p[4]) if len(p) > 4 and p[4] not in (".", "") else 0.0,
                         strand=p[5] if len(p) > 5 else ".")
            )
    return out

class IntervalIndex:

    def __init__(self, intervals: list[Interval]):
        self._by_chrom: dict[str, list[Interval]] = {}
        for iv in intervals:
            self._by_chrom.setdefault(iv.chrom, []).append(iv)
        for chrom in self._by_chrom:
            self._by_chrom[chrom].sort(key=lambda x: x.start)
        self._starts={c: [iv.start for iv in ivs] for c, ivs in self._by_chrom.items()}
        self._max_end={}
        for c, ivs in self._by_chrom.items():
            m=0
            arr=[]
            for iv in ivs:
                m=max(m, iv.end)
                arr.append(m)
            self._max_end[c]=arr

    def overlaps(self, chrom: str, start: int, end: int) -> list[Interval]:
        ivs=self._by_chrom.get(chrom)
        if not ivs:
            return []
        starts=self._starts[chrom]
        hi=bisect_left(starts, end)
        res=[]
        for i in range(hi - 1, -1, -1):
            iv=ivs[i]
            if iv.end > start:
                res.append(iv)
            if self._max_end[chrom][i] <= start:
                break
        return res

    def any_overlap(self, chrom: str, start: int, end: int) -> bool:
        return len(self.overlaps(chrom, start, end)) > 0

    def best_signal(self, chrom: str, start: int, end: int) -> float:
        ov=self.overlaps(chrom, start, end)
        return max((iv.signal for iv in ov), default=0.0)

    def __iter__(self) -> Iterator[Interval]:
        for ivs in self._by_chrom.values():
            yield from ivs

    def __len__(self) -> int:
        return sum(len(v) for v in self._by_chrom.values())

def intersect(a: IntervalIndex, intervals: list[Interval]) -> list[Interval]:
    return [iv for iv in intervals if a.any_overlap(iv.chrom, iv.start, iv.end)]
