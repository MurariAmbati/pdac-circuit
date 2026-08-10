from __future__ import annotations

import json
import sys
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from pdac_circuit.core.paths import RESULTS
from pdac_circuit.data.tracks import load_atac_peaks, load_h3k27ac_peaks

STD = {f"chr{i}" for i in range(1, 23)} | {"chrX", "chrY"}


def audit(idx):
    total = distinct = 0
    per_chrom = {}
    for chrom, ivs in idx._by_chrom.items():
        if chrom not in STD:
            continue
        d = len({(i.start, i.end) for i in ivs})
        total += len(ivs)
        distinct += d
        per_chrom[chrom] = {"intervals": len(ivs), "distinct": d}
    return total, distinct, per_chrom


def main():
    out = {
        "schema": "pdac-circuit.peak-duplication-audit/1",
        "data_class": "REAL",
        "generatedAt": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "question": (
            "The enhancer scale-up counted raw peak intervals pooled across every released ENCODE "
            "pancreas experiment. A region called in more than one experiment appears once per "
            "experiment. This audit measures how many of those intervals are distinct genomic "
            "coordinates."
        ),
        "corpora": {},
    }
    for name, load in (("encode-pancreas-atac", load_atac_peaks),
                       ("encode-pancreas-h3k27ac", load_h3k27ac_peaks)):
        total, distinct, per_chrom = audit(load())
        out["corpora"][name] = {
            "intervals": total,
            "distinct_coordinates": distinct,
            "redundant": total - distinct,
            "redundant_fraction": round((total - distinct) / total, 4) if total else None,
            "duplication_factor": round(total / distinct, 4) if distinct else None,
            "per_chromosome": per_chrom,
        }
    a = out["corpora"]["encode-pancreas-atac"]
    out["finding"] = (
        f"Of {a['intervals']:,} pooled ATAC intervals only {a['distinct_coordinates']:,} are distinct "
        f"coordinates, a duplication factor of {a['duplication_factor']}. "
        f"scripts/enhancer_maxdata.py iterates the pooled intervals without deduplication, so a region "
        f"called in k experiments contributes k training rows. The reported growth of the enhancer "
        f"training set therefore overstates the growth in distinct accessible regions."
    )
    out["leakage"] = (
        "This does not create train/test leakage. Splits are held out by chromosome, so every copy of a "
        "duplicated region falls on the same side of the split."
    )
    RESULTS.mkdir(parents=True, exist_ok=True)
    (RESULTS / "peak_duplication_audit.json").write_text(json.dumps(out, indent=2), encoding="utf-8")
    for k, v in out["corpora"].items():
        print(f"  {k:26} {v['intervals']:>10,} intervals  {v['distinct_coordinates']:>10,} distinct  "
              f"{v['duplication_factor']}x  {100*v['redundant_fraction']:.1f}% redundant")
    print("  wrote results/peak_duplication_audit.json")


if __name__ == "__main__":
    main()
