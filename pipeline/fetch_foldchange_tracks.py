from __future__ import annotations

import hashlib
import json
import subprocess
import sys
import time
from pathlib import Path

from pdac_circuit.core.paths import RAW

OUT_DIR=Path(RAW) / "encode-foldchange"
UA="pdac-circuit-fetch/0.1"

TRACKS=[
    {"accession": "ENCFF592MXT", "mark": "H3K27ac", "side": "healthy", "mb": 96.1,
     "biosample": "endocrine pancreas", "experiment": "ENCSR492PXH", "replaces": None},
    {"accession": "ENCFF047WWJ", "mark": "H3K27ac", "side": "pdac", "mb": 412.8,
     "biosample": "Panc1", "experiment": "ENCSR000EXK", "replaces": "ENCFF528UFR"},
    {"accession": "ENCFF174PXJ", "mark": "ATAC-seq", "side": "pdac", "mb": 676.7,
     "biosample": "Panc1", "experiment": "ENCSR591PIX", "replaces": "ENCFF055ZEE"},
    {"accession": "ENCFF140GLW", "mark": "ATAC-seq", "side": "healthy", "mb": 2360.3,
     "biosample": "pancreas", "experiment": "ENCSR530XBF", "replaces": None},
]

def log(m):
    print(f"[{time.strftime('%H:%M:%S')}] {m}", flush=True)

def expected_size(acc: str) -> int | None:
    url=f"https://www.encodeproject.org/files/{acc}/?format=json"
    try:
        r=subprocess.run(["curl", "-sL", "--max-time", "60", "-H", f"User-Agent: {UA}",
                            "-H", "Accept: application/json", url],
                           capture_output=True, text=True, timeout=90)
        return int(json.loads(r.stdout).get("file_size"))
    except Exception:
        return None

def sha256(p: Path) -> str:
    h=hashlib.sha256()
    with open(p, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()

def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    done=[]
    for t in TRACKS:
        acc=t["accession"]
        dest=OUT_DIR / f"{acc}_{t['mark']}_{t['side']}.foldchange.bigWig"
        want=expected_size(acc)
        if dest.exists() and want and dest.stat().st_size == want:
            log(f"{acc} already complete ({dest.stat().st_size/1e6:.1f} MB) - skipping")
            done.append((t, dest, want))
            continue
        url=f"https://www.encodeproject.org/files/{acc}/@@download/{acc}.bigWig"
        log(f"downloading {acc} ({t['mark']}/{t['side']}, ~{t['mb']:.0f} MB)")
        rc=subprocess.call(["curl", "-L", "-C", "-", "--retry", "3", "--retry-delay", "5",
                              "--max-time", "7200", "-H", f"User-Agent: {UA}",
                              "-o", str(dest), url])
        if rc != 0:
            log(f"  curl exit {rc} for {acc} - partial file kept, re-run to resume")
            continue
        got=dest.stat().st_size if dest.exists() else 0
        if want and got != want:
            log(f"  SIZE MISMATCH {acc}: got {got} expected {want} - re-run to resume")
            continue
        log(f"  ok {acc}: {got/1e6:.1f} MB")
        done.append((t, dest, want))

    if not done:
        log("nothing completed")
        return 1

    arts=[]
    for t, dest, want in done:
        log(f"hashing {dest.name}")
        arts.append({
            "bytes": dest.stat().st_size, "dataClass": "REAL", "license": None,
            "localPath": str(dest).replace("/", chr(92)), "name": dest.name,
            "note": (f"ENCODE {t['mark']} FOLD CHANGE OVER CONTROL, {t['biosample']}, GRCh38, "
                     f"experiment {t['experiment']}."
                     + (f" Matched processing run of {t['replaces']} (same derived_from) which was the "
                        f"signal p-value track previously used." if t["replaces"] else
                        " Healthy reference for the PDAC-vs-healthy contrast.")),
            "retrievedAt": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "sha256": sha256(dest), "status": "REAL",
            "url": f"https://www.encodeproject.org/files/{t['accession']}/",
        })
    man={"artifacts": sorted(arts, key=lambda a: a["name"]), "corpus": "encode-foldchange",
           "generatedAt": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
           "note": ("Fold-change-over-control tracks obtained to replace signal p-value tracks, which "
                    "conflate sequencing depth with enrichment. Fold-change files were selected by "
                    "matching the derived_from processing run of the p-value files they replace, so an "
                    "old-vs-new comparison changes normalisation ONLY."),
           "schema": "pdac-circuit.provenance/1",
           "summary": {"gated": 0, "pointer": 0, "real": len(arts)}}
    mp=Path("data/manifests/encode-foldchange.json")
    mp.write_text(json.dumps(man, indent=1, sort_keys=True), encoding="utf-8")
    log(f"wrote {mp} ({len(arts)} artifacts)")
    for a in arts:
        log(f"  {a['bytes']/1e6:9.1f} MB  {a['sha256'][:16]}...  {a['name']}")
    return 0

if __name__ == "__main__":
    sys.exit(main())
