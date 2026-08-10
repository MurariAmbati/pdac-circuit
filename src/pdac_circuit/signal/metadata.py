from __future__ import annotations

import json
import urllib.request

from ..core.paths import MANIFESTS, RAW

ENCODE="https://www.encodeproject.org"
META_CACHE=RAW / "encode-bulk" / "bam_metadata.json"

def _ctx():
    import ssl

    try:
        import certifi

        return ssl.create_default_context(cafile=certifi.where())
    except Exception:
        return ssl.create_default_context()

def _encode_file(accession: str) -> dict:
    url=f"{ENCODE}/files/{accession}/?format=json"
    req=urllib.request.Request(url, headers={"User-Agent": "pdac-circuit/0.1", "Accept": "application/json"})
    with urllib.request.urlopen(req, timeout=60, context=_ctx()) as resp:
        return json.loads(resp.read().decode("utf-8"))

def _encode_obj(path: str) -> dict:
    req=urllib.request.Request(f"{ENCODE}{path}?format=json", headers={"User-Agent": "pdac-circuit/0.1", "Accept": "application/json"})
    with urllib.request.urlopen(req, timeout=60, context=_ctx()) as resp:
        return json.loads(resp.read().decode("utf-8"))

def bam_accessions() -> list[str]:
    d=json.loads((MANIFESTS / "encode-bulk.heavy.json").read_text(encoding="utf-8"))
    return [a["name"].replace(".bam", "") for a in d["artifacts"] if a["name"].endswith(".bam")]

def build_metadata(refresh: bool = False) -> dict[str, dict]:
    if META_CACHE.exists() and not refresh:
        return json.loads(META_CACHE.read_text(encoding="utf-8"))
    out: dict[str, dict]={}
    for acc in bam_accessions():
        try:
            f=_encode_file(acc)
            exp_path=f.get("dataset") or (f.get("derived_from") or [None])[0]
            assay="?"
            target="?"
            biosample="?"
            if exp_path:
                exp=_encode_obj(exp_path)
                assay=exp.get("assay_title", "?")
                tgt=exp.get("target") or {}
                target=tgt.get("label") if isinstance(tgt, dict) else (tgt.split("/")[-2] if isinstance(tgt, str) else "?")
                bs=exp.get("biosample_ontology") or {}
                biosample=bs.get("term_name", "?") if isinstance(bs, dict) else "?"
            mark_class=_classify(target)
            out[acc]={"assay": assay, "target": target, "biosample": biosample, "mark_class": mark_class,
                        "output_type": f.get("output_type", "?")}
        except Exception as e:
            out[acc]={"assay": "?", "target": "?", "biosample": "?", "mark_class": "unknown", "error": f"{type(e).__name__}"}
    META_CACHE.parent.mkdir(parents=True, exist_ok=True)
    META_CACHE.write_text(json.dumps(out, indent=2, sort_keys=True), encoding="utf-8")
    return out

def _classify(target: str) -> str:
    t=(target or "").upper()
    if t in ("H3K27AC",):
        return "active_enhancer_promoter"
    if t in ("H3K4ME1",):
        return "enhancer"
    if t in ("H3K4ME3",):
        return "promoter"
    if t in ("H3K27ME3",):
        return "repressed"
    if t in ("H3K9ME3",):
        return "heterochromatin"
    if t in ("H3K36ME3",):
        return "transcribed"
    if t and t != "?":
        return "tf"
    return "unknown"

def bams_by_class() -> dict[str, list[str]]:
    meta=build_metadata()
    groups: dict[str, list[str]]={}
    for acc, m in meta.items():
        groups.setdefault(m["mark_class"], []).append(acc)
    return groups
