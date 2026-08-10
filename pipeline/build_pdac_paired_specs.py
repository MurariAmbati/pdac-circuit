from __future__ import annotations

import json

from pdac_circuit.core.paths import RAW

HEALTHY_SPECS = RAW.parent / "track_specs" / "encode_healthy_pancreas"
PDAC_SPECS = RAW.parent / "track_specs" / "encode_panc1_pdac"
OUT = RAW.parent / "track_specs" / "pdac_paired"

PAIRS = {
    "H3K27ac": ("ENCFF931BVK","ENCFF528UFR"),
    "ATAC": ("ENCFF074XBT","ENCFF055ZEE"),
    "H3K4me1": ("ENCFF622NVF","ENCFF155VKT"),
}

def main():
    OUT.mkdir(parents=True,exist_ok=True)
    made = []
    for mark,(h_acc,p_acc) in PAIRS.items():
        h_src = HEALTHY_SPECS / f"{h_acc}.json"
        p_src = PDAC_SPECS / f"{p_acc}.json"
        if not h_src.exists() or not p_src.exists():
            print(f"  SKIP {mark}: missing spec ({h_src.exists()=}, {p_src.exists()=})")
            continue
        group = f"PANC1_vs_HEALTHY_{mark}"
        h = json.loads(h_src.read_text())
        p = json.loads(p_src.read_text())
        h["pair_group"] = group
        h["pair_relation"] = "state_reference"
        p["pair_group"] = group
        p["pair_relation"] = "state_treatment"
        for spec in (h,p):
            pf = spec.get("perturbation_features",[])
            if len(pf) < 22:
                spec["perturbation_features"] = list(pf) + [0.0] * (22 - len(pf))
        (OUT / f"{mark}_reference_{h_acc}.json").write_text(json.dumps(h,indent=2,sort_keys=True))
        (OUT / f"{mark}_treatment_{p_acc}.json").write_text(json.dumps(p,indent=2,sort_keys=True))
        made.append((mark,group,h_acc,p_acc))
        print(f"  {mark:8} group={group:28} ref={h_acc} (state_reference)  treat={p_acc} (state_treatment)")
    print(f"\nwrote {len(made)*2} paired specs -> {OUT}")

if __name__ == "__main__":
    main()
