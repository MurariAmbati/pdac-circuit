from __future__ import annotations

import hashlib
import json
import threading
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from ..core.paths import MANIFESTS, RAW
from ..core.provenance import REAL, build_doc, make_artifact

ENCODE_BASE="https://www.encodeproject.org"
USER_AGENT="pdac-circuit-bulk/0.1 (research use only)"
CHUNK=1 << 20

PANCREAS_TERMS=[
    "pancreas", "body of pancreas", "endocrine pancreas", "exocrine pancreas",
    "islet of Langerhans", "Islets of Langerhans",
]
ASSAYS=[
    "ATAC-seq", "DNase-seq", "Histone ChIP-seq", "TF ChIP-seq",
    "total RNA-seq", "polyA plus RNA-seq", "WGBS", "RNA-seq", "RAMPAGE",
]

_GNOMAD="https://storage.googleapis.com/gcp-public-data--gnomad/release/3.1.2/vcf/genomes"
GNOMAD_FILES=[
    {"name": f"gnomad.genomes.v3.1.2.sites.chr{c}.vcf.bgz",
     "url": f"{_GNOMAD}/gnomad.genomes.v3.1.2.sites.chr{c}.vcf.bgz", "source": "gnomAD v3.1.2"}
    for c in [str(i) for i in range(1, 23)] + ["X", "Y"]
]

_GTEX="https://storage.googleapis.com/adult-gtex/bulk-gex/v8/rna-seq"
GTEX_FILES=[
    {"name": "GTEx_v8_gene_reads.gct.gz", "url": f"{_GTEX}/GTEx_Analysis_2017-06-05_v8_RNASeQCv1.1.9_gene_reads.gct.gz", "source": "GTEx v8"},
    {"name": "GTEx_v8_transcript_tpm.gct.gz", "url": f"{_GTEX}/GTEx_Analysis_2017-06-05_v8_RNASeQCv1.1.9_transcript_tpm.gct.gz", "source": "GTEx v8"},
    {"name": "GTEx_v8_transcript_reads.gct.gz", "url": f"{_GTEX}/GTEx_Analysis_2017-06-05_v8_RNASeQCv1.1.9_transcript_reads.gct.gz", "source": "GTEx v8"},
    {"name": "GTEx_v8_junction_reads.gct.gz", "url": f"{_GTEX}/GTEx_Analysis_2017-06-05_v8_STARv2.5.3a_junctions.gct.gz", "source": "GTEx v8"},
]

def _ctx():
    import ssl

    try:
        import certifi

        return ssl.create_default_context(cafile=certifi.where())
    except Exception:
        return ssl.create_default_context()

def _encode_search(params: dict) -> list[dict]:
    import urllib.parse

    q=urllib.parse.urlencode(params, doseq=True)
    req=urllib.request.Request(f"{ENCODE_BASE}/search/?{q}", headers={"User-Agent": USER_AGENT, "Accept": "application/json"})
    with urllib.request.urlopen(req, timeout=180, context=_ctx()) as resp:
        return json.loads(resp.read().decode("utf-8")).get("@graph", [])

def resolve_encode(formats=("bam", "bigWig"), max_gb: float = 12.0) -> list[dict]:
    params={
        "type": "File", "status": "released", "assembly": "GRCh38",
        "file_format": list(formats), "assay_title": ASSAYS,
        "biosample_ontology.term_name": PANCREAS_TERMS,
        "limit": "all", "format": "json",
        "field": ["href", "file_size", "md5sum", "file_format", "assay_title", "output_type"],
    }
    cap=int(max_gb * (1 << 30))
    out=[]
    seen=set()
    for f in _encode_search(params):
        href=f.get("href")
        if not href or href in seen or not f.get("file_size"):
            continue
        size=int(f["file_size"])
        if size > cap:
            continue
        seen.add(href)
        out.append({
            "name": Path(href).name, "url": ENCODE_BASE + href,
            "size": size, "md5": f.get("md5sum"),
            "source": f"ENCODE {f.get('assay_title','')}/{f.get('output_type','')}", "corpus": "encode-bulk",
        })
    out.sort(key=lambda d: d["size"])
    return out

def _download_resumable(url: str, dest: Path, expected_md5: str | None = None) -> tuple[str, int]:
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp=dest.with_suffix(dest.suffix + ".part")
    if tmp.exists():
        tmp.unlink()
    h=hashlib.sha256()
    m=hashlib.md5()
    n=0
    req=urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    with urllib.request.urlopen(req, timeout=120, context=_ctx()) as resp, open(tmp, "wb") as fh:
        while True:
            block=resp.read(CHUNK)
            if not block:
                break
            fh.write(block)
            h.update(block)
            m.update(block)
            n += len(block)
    if expected_md5 and m.hexdigest() != expected_md5:
        tmp.unlink(missing_ok=True)
        raise ValueError(f"md5 mismatch for {dest.name}: source {expected_md5} != got {m.hexdigest()}")
    tmp.replace(dest)
    return h.hexdigest(), n

def dir_size_bytes(path: Path) -> int:
    return sum(f.stat().st_size for f in path.rglob("*") if f.is_file())

def bulk_fetch(*, target_gb: float = 110.0, workers: int = 6, include_gnomad: bool = True) -> int:
    target=int(target_gb * (1 << 30))
    RAW.mkdir(parents=True, exist_ok=True)
    start=dir_size_bytes(RAW)
    print(f"[bulk] data/raw currently {start/2**30:.1f} GB; target {target_gb} GB")

    queue: list[dict]=[]
    try:
        enc=resolve_encode()
        print(f"[bulk] resolved {len(enc)} ENCODE pancreas/islet files "
              f"({sum(e['size'] for e in enc)/2**30:.1f} GB available)")
        queue += enc
    except Exception as e:
        print(f"[bulk] ENCODE resolve failed: {type(e).__name__}: {e}")
    if include_gnomad:
        queue += [{**g, "corpus": "gnomad-genomes", "size": None, "md5": None} for g in GNOMAD_FILES]
    queue += [{**g, "corpus": "gtex-full", "size": None, "md5": None} for g in GTEX_FILES]

    lock=threading.Lock()
    state={"bytes": start, "done": 0, "fail": 0, "skip": 0}
    stop=threading.Event()

    manifest: dict[str, list[dict]]={}
    known: set[str]=set()
    for hv in MANIFESTS.glob("*.heavy.json"):
        try:
            doc=json.loads(hv.read_text(encoding="utf-8"))
            arts=doc.get("artifacts", [])
            if arts:
                manifest[hv.stem.replace(".heavy", "")]=arts
                known.update(a["name"] for a in arts if a.get("sha256"))
        except Exception:
            pass

    def worker(spec: dict):
        if stop.is_set():
            return
        corpus=spec["corpus"]
        dest=RAW / corpus / spec["name"]
        try:
            if spec["name"] in known and dest.exists() and (spec.get("size") in (None, dest.stat().st_size)):
                with lock:
                    state["skip"] += 1
                return
            already=dest.exists() and spec.get("size") is not None and dest.stat().st_size == spec["size"]
            if already:
                from ..core.seeds import sha256_file

                sha, nbytes=sha256_file(dest), dest.stat().st_size
            else:
                sha, nbytes=_download_resumable(spec["url"], dest, spec.get("md5"))
            with lock:
                state["done"] += 1
                known.add(spec["name"])
                manifest.setdefault(corpus, []).append(make_artifact(
                    spec["name"], spec["url"], REAL, sha256=sha, n_bytes=nbytes,
                    local_path=str(dest.relative_to(RAW.parent.parent)), note=spec.get("source"),
                ))
                doc=build_doc(corpus, manifest[corpus], note="bulk raw open data")
                (MANIFESTS / f"{corpus}.heavy.json").write_text(json.dumps(doc, indent=2, sort_keys=True), encoding="utf-8")
                cur=dir_size_bytes(RAW)
                print(f"[bulk] +{nbytes/2**30:.2f}GB {spec['name'][:40]:40s} total={cur/2**30:.1f}GB ({state['done']} files)", flush=True)
                if cur >= target:
                    stop.set()
        except Exception as e:
            with lock:
                state["fail"] += 1
                print(f"[bulk] FAIL {spec['name'][:40]}: {type(e).__name__}: {e}", flush=True)

    with ThreadPoolExecutor(max_workers=workers) as ex:
        futs=[]
        for spec in queue:
            if stop.is_set():
                break
            futs.append(ex.submit(worker, spec))
        for _ in as_completed(futs):
            if stop.is_set():
                break

    for corpus, arts in manifest.items():
        doc=build_doc(corpus, arts, note="bulk raw open data")
        (MANIFESTS / f"{corpus}.heavy.json").write_text(json.dumps(doc, indent=2, sort_keys=True), encoding="utf-8")

    final=dir_size_bytes(RAW)
    print(f"[bulk] DONE: data/raw = {final/2**30:.1f} GB ({state['done']} files this run, {state['fail']} failed)")
    return 0 if final >= target else 1
