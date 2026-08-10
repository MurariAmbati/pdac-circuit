from __future__ import annotations

import json
import time

import numpy as np
from numpy.lib.stride_tricks import sliding_window_view

from pdac_circuit.core.paths import RESULTS

OUT = RESULTS / "genomewide_offtarget_audit.json"
MAX_MM = 4
CHUNK = 20_000_000
GUIDES = {
    "MYBL2": "CGCTGGTGAGACGAGCCGGG",
    "SETDB1": "ACCCCAGACTCACAACTCAG",
    "FOSL1": "TCTGACTCACCCGCGCCGTG",
    "E2F1": "GGAGATGATGACGATCTGCG",
}
_MAP = np.full(256,4,dtype=np.uint8)
for i,c in enumerate("ACGT"):
    _MAP[ord(c)] = i
    _MAP[ord(c.lower())] = i
_C = 1
_G = 2
_COMP = np.array([3,2,1,0,4],dtype=np.uint8)

def encode(s: str) -> np.ndarray:
    return _MAP[np.frombuffer(s.encode("ascii","ignore"),dtype=np.uint8)]

def scan_block(arr,guides_enc,guides_rc,max_mm):
    if arr.size < 23:
        return {g: np.empty((0,3),dtype=np.int64) for g in guides_enc}
    w = sliding_window_view(arr,23)
    plus_idx = np.flatnonzero((w[:,21] == _G) & (w[:,22] == _G))
    minus_idx = np.flatnonzero((w[:,0] == _C) & (w[:,1] == _C))

    out = {}
    for name in guides_enc:
        hits = []
        for idx,gd,lo,strand in ((plus_idx,guides_enc[name],0,0),
                                    (minus_idx,guides_rc[name],3,1)):
            for s in range(0,idx.size,2_000_000):
                sub = idx[s:s + 2_000_000]
                if sub.size == 0:
                    continue
                km = w[sub,lo:lo + 20]
                mm = np.zeros(sub.size,dtype=np.uint8)
                for p in range(20):
                    mm += (km[:,p] != gd[p])
                keep = np.flatnonzero(mm <= max_mm)
                if keep.size:
                    hits.append(np.stack([sub[keep],mm[keep],
                                          np.full(keep.size,strand)],axis=1))
        out[name] = (np.concatenate(hits) if hits else np.empty((0,3),dtype=np.int64))
    return out

def main():
    from pdac_circuit.data.reference import _genome
    from pdac_circuit.grna.offtarget import cfd_style_score

    g = _genome()
    chroms = [f"chr{c}" for c in list(range(1,23)) + ["X","Y"]]
    chroms = [c for c in chroms if c in g or c.replace("chr","") in g]
    guides_enc = {k: encode(v) for k,v in GUIDES.items()}
    guides_rc = {k: _COMP[encode(v)][::-1].copy() for k,v in GUIDES.items()}

    counts = {k: dict.fromkeys(range(MAX_MM + 1),0) for k in GUIDES}
    examples = {k: [] for k in GUIDES}
    scanned_bp = 0
    t0 = time.time()

    for ch in chroms:
        key = ch if ch in g else ch.replace("chr","")
        seq = str(g[key])
        arr = encode(seq)
        del seq
        scanned_bp += arr.size
        for start in range(0,arr.size,CHUNK):
            block = arr[start:start + CHUNK + 22]
            res = scan_block(block,guides_enc,guides_rc,MAX_MM)
            for name,hits in res.items():
                for off,mm,strand in hits:
                    counts[name][int(mm)] += 1
                    if int(mm) >= 1 and len(examples[name]) < 12:
                        km = block[int(off):int(off) + 23]
                        km = _COMP[km][::-1] if int(strand) == 1 else km
                        examples[name].append({
                            "chrom": ch,"pos": int(start + int(off)),
                            "strand": "-" if int(strand) == 1 else "+",
                            "seq23": "".join("ACGTN"[b] for b in km),"mismatches": int(mm)})
        del arr
        done = sum(sum(c.values()) for c in counts.values())
        print(f"  {ch:6} scanned {scanned_bp/1e6:8.1f} Mb | cumulative hits {done} | "
              f"{time.time()-t0:6.0f}s",flush=True)

    rows = []
    for name,proto in GUIDES.items():
        c = counts[name]
        n_off = sum(v for m,v in c.items() if m >= 1)
        extra_perfect = max(0,c[0] - 1)
        off_scores = []
        for e in examples[name]:
            off_scores.append(cfd_style_score(proto,e["seq23"][:20],e["seq23"][21:23]))
        rows.append({
            "gene": name,"protospacer": proto,
            "perfect_match_sites_including_on_target": c[0],
            "additional_perfect_matches_elsewhere": extra_perfect,
            "off_target_sites_by_mismatch": {str(m): c[m] for m in range(1,MAX_MM + 1)},
            "total_off_target_sites_le_4mm": n_off,
            "shipped_report": "0 off-targets, CFD specificity 1.00 (search space: target loci +/-5kb)",
            "worst_example_cfd_style": round(float(max(off_scores)),4) if off_scores else None,
            "example_hits": examples[name][:6],
        })
        print(f"  {name:7} perfect={c[0]}  off<=4mm={n_off}  "
              f"by-mm={ {m: c[m] for m in range(1,MAX_MM+1)} }",flush=True)

    bounded_bp = 4 * 10_000
    rep = {
        "schema": "pdac-circuit.genomewide-offtarget-audit/1","data_class": "REAL",
        "sealed_studies_touched": False,
        "assembly": "hg38 (local FASTA)","max_mismatch": MAX_MM,
        "genome_bp_scanned": int(scanned_bp),"chromosomes": chroms,
        "shipped_search_space_bp_approx": bounded_bp,
        "fraction_of_genome_searched_by_shipped_code": round(bounded_bp / max(scanned_bp,1),9),
        "scorer": "the repository's own mit_single_score / cfd_style_score, applied to real hits",
        "design": ("every NGG site within <=4 mismatches of each proposed guide, both strands, "
                   "across the main hg38 assembly; compared against the shipped bounded search "
                   "which only looks at the target loci +/-5kb"),
        "per_guide": rows,
        "runtime_s": round(time.time() - t0,1),
    }
    OUT.write_text(json.dumps(rep,indent=2),encoding="utf-8")
    print(f"\nscanned {scanned_bp/1e9:.2f} Gb; shipped search covers "
          f"{bounded_bp/scanned_bp*100:.5f}% of it")
    print(f"wrote {OUT}")

if __name__ == "__main__":
    main()
