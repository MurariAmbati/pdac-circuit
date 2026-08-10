from __future__ import annotations

import hashlib
import json
import ssl
import urllib.parse
import urllib.request
from datetime import datetime, timezone
from pathlib import Path

from src.pdac_circuit.core.paths import RAW

BASE="https://www.encodeproject.org"
UA={"User-Agent": "pdac-circuit/0.1", "Accept": "application/json"}
OUTPUT_TYPES=["IDR thresholded peaks", "pseudoreplicated peaks", "replicated peaks", "peaks"]

TARGETS=[
    ("ATAC-seq", None, RAW / "encode-pancreas-atac", "encode-pancreas-atac"),
    ("Histone ChIP-seq", "H3K27ac", RAW / "encode-pancreas-h3k27ac", "encode-pancreas-h3k27ac"),
]


def _ctx():
    c=ssl.create_default_context()
    c.check_hostname=False
    c.verify_mode=ssl.CERT_NONE
    return c


def search(assay, target):
    params={
        "type": "File", "file_format": "bed", "file_format_type": "narrowPeak",
        "assembly": "GRCh38", "assay_title": assay,
        "biosample_ontology.term_name": "pancreas", "status": "released",
        "limit": "200", "format": "json", "output_type": OUTPUT_TYPES,
    }
    if target:
        params["target.label"]=target
    url=f"{BASE}/search/?{urllib.parse.urlencode(params, doseq=True)}"
    req=urllib.request.Request(url, headers=UA)
    with urllib.request.urlopen(req, timeout=120, context=_ctx()) as r:
        return json.loads(r.read().decode("utf-8")).get("@graph", [])


def download(href, dest):
    req=urllib.request.Request(BASE + href, headers={"User-Agent": UA["User-Agent"]})
    with urllib.request.urlopen(req, timeout=600, context=_ctx()) as r, open(dest, "wb") as f:
        while True:
            chunk=r.read(1 << 20)
            if not chunk:
                break
            f.write(chunk)


def sha256(p):
    h=hashlib.sha256()
    with open(p, "rb") as f:
        while True:
            b=f.read(1 << 22)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


def main():
    now=datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    for assay, target, dest_dir, corpus in TARGETS:
        dest_dir.mkdir(parents=True, exist_ok=True)
        files=search(assay, target)
        print(f"\n{corpus}: {len(files)} released narrowPeak files on ENCODE")
        artifacts, added=[], 0
        for f in files:
            href, acc=f.get("href"), f.get("accession")
            if not href or not acc:
                continue
            out=dest_dir / Path(href).name
            if not out.exists():
                try:
                    download(href, out)
                    added += 1
                    print(f"  fetched {out.name}  ({out.stat().st_size/1e6:.1f} MB)")
                except Exception as e:
                    print(f"  FAILED {acc}: {type(e).__name__}: {e}")
                    continue
            artifacts.append({
                "name": out.name, "accession": acc,
                "localPath": str(out.relative_to(RAW.parent.parent)).replace("/", "\\"),
                "url": BASE + href, "bytes": out.stat().st_size,
                "sha256": sha256(out), "dataClass": "REAL", "status": "REAL",
                "output_type": f.get("output_type"), "retrievedAt": now, "license": None,
            })
        man={
            "schema": "pdac-circuit.provenance/1", "corpus": corpus, "generatedAt": now,
            "note": "All released GRCh38 narrowPeak files for this assay in pancreas. "
                    "Previously only the first four were retrieved.",
            "artifacts": sorted(artifacts, key=lambda a: a["name"]),
            "summary": {"real": len(artifacts), "gated": 0, "pointer": 0},
        }
        mp=RAW.parent / "manifests" / f"{corpus}.json"
        mp.write_text(json.dumps(man, indent=2, sort_keys=True), encoding="utf-8")
        print(f"  on disk now: {len(artifacts)} files ({added} newly fetched)")
        print(f"  manifest -> {mp.relative_to(RAW.parent.parent)}")


if __name__ == "__main__":
    main()
