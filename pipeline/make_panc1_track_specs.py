from __future__ import annotations

import hashlib
import json
from pathlib import Path

from pdac_circuit.chromatin.encode import assay_vector
from pdac_circuit.core.paths import RAW

PDAC_DIR = RAW / "encode-panc1-pdac"
SPEC_DIR = RAW.parent / "track_specs" / "encode_panc1_pdac"
FILES_JSON = SPEC_DIR / "_encode_file_index.json"
PERTURBATION_DIM = 22

def panc1_state_vector() -> list[float]:
    state = [0.0] * 18
    state[2] = 1.0
    state[10] = 0.0
    state[11] = 1.0
    state[12] = 0.0
    state[13] = 0.0
    state[14] = 1.0
    state[15] = 1.0
    state[16] = 1.0
    return state

def sha256_file(p: Path) -> str:
    h = hashlib.sha256()
    with open(p,"rb") as f:
        for chunk in iter(lambda: f.read(8 << 20),b""):
            h.update(chunk)
    return h.hexdigest()

def main():
    meta = {r["file"]: r for r in json.loads(FILES_JSON.read_text())}
    SPEC_DIR.mkdir(parents=True,exist_ok=True)
    made,skipped = [],[]
    for bw in sorted(PDAC_DIR.glob("*.signal.bigWig")):
        acc = bw.name.split("_")[0]
        r = meta.get(acc)
        if r is None:
            skipped.append((bw.name,"no ENCODE metadata"))
            continue
        md = {"assay": r["assay"],"target": r["target"] or "","output_type": r["output"],
              "status": "released"}
        try:
            av = list(assay_vector(md))
        except ValueError as e:
            skipped.append((bw.name,f"unmappable assay: {e}"))
            continue
        spec = {
            "accession": acc,
            "assay_features": av,
            "biological_replicate": "",
            "biological_state": "panc1_pdac_cell_line",
            "disease": True,
            "genome": "hg38",
            "metadata_sha256": None,
            "organism": "Homo sapiens",
            "pair_group": "",
            "pair_relation": "unpaired",
            "path": str(bw.resolve()),
            "perturbation_features": [0.0] * PERTURBATION_DIM,
            "perturbation_label": "none",
            "released": "",
            "sample_accession": "",
            "sample_group": r["exp"],
            "source_sha256": sha256_file(bw),
            "split_role": "train_state",
            "state_features": panc1_state_vector(),
            "study": "ENCODE_PANC1_PDAC",
        }
        out = SPEC_DIR / f"{acc}.json"
        out.write_text(json.dumps(spec,indent=2,sort_keys=True))
        made.append((acc,r["target"] or r["assay"],out.name))
        print(f"  spec {acc:14} {r['target'] or r['assay']!s:10} assay_vec={[i for i,x in enumerate(av) if x]}",
              flush=True)
    print(f"\nwrote {len(made)} PDAC TrackSpecs -> {SPEC_DIR}")
    for n,why in skipped:
        print(f"  SKIPPED {n}: {why}")

if __name__ == "__main__":
    main()
