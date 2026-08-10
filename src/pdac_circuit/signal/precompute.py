from __future__ import annotations

import json
from multiprocessing import Pool

import numpy as np

from ..core.paths import RAW, RESULTS
from .chromatin import chromatin_features
from .metadata import build_metadata

CACHE=RESULTS / "chromatin_states.json"
STD_CHROMS=tuple(f"chr{i}" for i in range(1, 23)) + ("chrX",)

def _bam_path(acc: str) -> str:
    return str(RAW / "encode-bulk" / f"{acc}.bam")

def select_bams(per_mark: int = 1) -> dict[str, list[str]]:
    import json as _json

    from ..core.paths import MANIFESTS

    sizes={a["name"].replace(".bam", ""): a["bytes"]
             for a in _json.loads((MANIFESTS / "encode-bulk.heavy.json").read_text())["artifacts"]
             if a["name"].endswith(".bam")}
    meta=build_metadata()
    by_target: dict[str, list[str]]={}
    for acc, m in meta.items():
        tgt=m.get("target", "?")
        if tgt and tgt != "?":
            by_target.setdefault(tgt, []).append(acc)
    out={}
    for tgt, accs in by_target.items():
        accs.sort(key=lambda a: sizes.get(a, 1e18))
        out[tgt]=accs[:per_mark]
    return out

def _background_regions(n: int, width: int, seed: int) -> list[dict]:
    rng=np.random.default_rng(seed)
    out=[]
    for _ in range(n):
        c=STD_CHROMS[rng.integers(len(STD_CHROMS))]
        pos=int(rng.integers(30_000_000, 60_000_000))
        out.append({"chrom": c, "start": pos, "end": pos + width})
    return out

def _worker(args):
    from .bamio import extract_coverage

    acc, target, regions=args
    try:
        cov=extract_coverage(_bam_path(acc), regions, frag_extend=150)
        means={i: float(cov[i].mean()) for i in cov}
        return target, means, acc
    except Exception as e:
        return target, {"_error": str(e)}, acc

def precompute_chromatin(*, genes: list[str] | None = None, per_mark: int = 1, n_background: int = 40,
                         width: int = 5000, workers: int = 6, seed: int = 20260620) -> dict:
    from ..data.genes import promoter_window
    from ..targeting import build_afm, prioritize_targets

    if genes is None:
        afm=build_afm()
        out, _=prioritize_targets(top_k=40, afm=afm)
        genes=list(out.table.sort_values("composite", ascending=False).index[:40])

    loci, gene_order=[], []
    for g in genes:
        w=promoter_window(g, up=width // 2, down=width // 2)
        if w:
            loci.append(w)
            gene_order.append(g)
    bg=_background_regions(n_background, width, seed)
    regions=loci + bg
    n_loci=len(loci)

    selected=select_bams(per_mark=per_mark)
    jobs=[(acc, tgt, regions) for tgt, accs in selected.items() for acc in accs]
    print(f"[chromatin] {len(gene_order)} loci x {len(jobs)} BAMs ({len(selected)} marks), {workers} workers")

    with Pool(processes=workers) as pool:
        results=pool.map(_worker, jobs)

    per_mark_locus: dict[str, np.ndarray]={}
    for target, means, acc in results:
        if "_error" in means:
            print(f"[chromatin] {acc} ({target}) failed: {means['_error']}")
            continue
        locus_vals=np.array([means.get(i, 0.0) for i in range(n_loci)])
        bg_vals=np.array([means.get(i, 0.0) for i in range(n_loci, len(regions))])
        bg_mean=bg_vals.mean() if bg_vals.size else 0.0
        enrich=locus_vals / (bg_mean + 1e-6)
        if target in per_mark_locus:
            per_mark_locus[target]=np.vstack([per_mark_locus[target], enrich]).mean(axis=0)
        else:
            per_mark_locus[target]=enrich

    states={}
    for li, gene in enumerate(gene_order):
        marks={tgt: float(per_mark_locus[tgt][li]) for tgt in per_mark_locus}
        states[gene]=chromatin_features(marks)

    report={
        "schema": "pdac-circuit.chromatin/1",
        "n_loci": n_loci, "marks": sorted(per_mark_locus.keys()),
        "n_bams_used": len([r for r in results if "_error" not in r[1]]),
        "states": states,
    }
    RESULTS.mkdir(parents=True, exist_ok=True)
    CACHE.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    return report

def load_states() -> dict | None:
    return json.loads(CACHE.read_text(encoding="utf-8")) if CACHE.exists() else None

if __name__ == "__main__":
    import sys

    pm=2 if "--full" in sys.argv else 1
    ng=40
    if "--n-genes" in sys.argv:
        ng=int(sys.argv[sys.argv.index("--n-genes") + 1])
    from ..targeting import build_afm, prioritize_targets

    afm=build_afm()
    out, _=prioritize_targets(top_k=40, afm=afm)
    genes=list(out.table.sort_values("composite", ascending=False).index[:ng])
    rep=precompute_chromatin(genes=genes, per_mark=pm)
    print(f"[chromatin] done: {rep['n_loci']} loci, {len(rep['marks'])} marks, {rep['n_bams_used']} BAMs")
    from collections import Counter
    print("states:", dict(Counter(s["state"] for s in rep["states"].values())))
