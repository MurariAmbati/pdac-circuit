from __future__ import annotations

import hashlib
import json
import os
import ssl
import urllib.request
from pathlib import Path

from ..core.paths import CORPORA_JSON, DEPMAP_CRISPR, MANIFESTS, RAW
from ..core.provenance import (
    GATED,
    POINTER,
    REAL,
    build_doc,
    make_artifact,
    write_manifest,
)

USER_AGENT="pdac-circuit-fetch/0.1 (research use only)"
TIMEOUT_S=120
CHUNK=1 << 20

def _ssl_context() -> ssl.SSLContext:
    try:
        import certifi

        return ssl.create_default_context(cafile=certifi.where())
    except Exception:
        return ssl.create_default_context()

def load_corpora() -> list[dict]:
    return json.loads(CORPORA_JSON.read_text(encoding="utf-8"))["corpora"]

def _download(url: str, dest: Path) -> tuple[str, int]:
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp=dest.with_suffix(dest.suffix + ".part")
    h=hashlib.sha256()
    n=0
    req=urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    with urllib.request.urlopen(req, timeout=TIMEOUT_S, context=_ssl_context()) as resp, open(tmp, "wb") as f:
        while True:
            block=resp.read(CHUNK)
            if not block:
                break
            f.write(block)
            h.update(block)
            n += len(block)
    tmp.replace(dest)
    return h.hexdigest(), n

def _fetch_file(corpus_id: str, spec: dict) -> dict:
    name=spec["name"]
    url=spec["url"]
    dest=RAW / corpus_id / name
    if dest.exists():
        from ..core.seeds import sha256_file

        sha=sha256_file(dest)
        return make_artifact(
            name, url, REAL, sha256=sha, n_bytes=dest.stat().st_size,
            local_path=str(dest.relative_to(RAW.parent.parent)), note="already present",
        )
    try:
        sha, n = _download(url, dest)
        return make_artifact(
            name, url, REAL, sha256=sha, n_bytes=n,
            local_path=str(dest.relative_to(RAW.parent.parent)),
        )
    except Exception as e:
        return make_artifact(name, url, POINTER, note=f"download failed: {type(e).__name__}: {e}")

def fetch_corpus(corpus: dict, *, heavy: bool = False, dry_run: bool = False) -> dict:
    cid=corpus["id"]
    tier=corpus["accessTier"]
    artifacts: list[dict] = []

    if tier == "gated":
        env=corpus.get("credentialEnv", "")
        has=bool(os.environ.get(env)) if env else False
        note=corpus.get("credentialHint", "gated — credentials required")
        if not has:
            artifacts.append(make_artifact(cid, corpus.get("landingUrl", ""), GATED, note=note))
            doc=build_doc(cid, artifacts, note=note)
            write_manifest(MANIFESTS, cid, doc)
            return doc
        artifacts.append(make_artifact(cid, corpus.get("landingUrl", ""), POINTER, note="credentials present; gated fetch not auto-pulled"))
        doc=build_doc(cid, artifacts)
        write_manifest(MANIFESTS, cid, doc)
        return doc

    if cid == "depmap-crispr":
        if DEPMAP_CRISPR.exists():
            from ..core.seeds import sha256_file

            sha=sha256_file(DEPMAP_CRISPR)
            artifacts.append(
                make_artifact("CRISPRGeneEffect.csv", corpus["landingUrl"], REAL,
                              sha256=sha, n_bytes=DEPMAP_CRISPR.stat().st_size,
                              local_path=str(DEPMAP_CRISPR), note="reused from sibling glio-ai project")
            )
        else:
            artifacts.append(make_artifact("CRISPRGeneEffect.csv", corpus["landingUrl"], POINTER, note="not materialized on this machine"))
        doc=build_doc(cid, artifacts)
        (MANIFESTS / f"{cid}.heavy.json").write_text(json.dumps(doc, indent=2, sort_keys=True), encoding="utf-8")
        return doc

    if corpus.get("apiResolved") and not heavy:
        from ..core.seeds import sha256_file

        cdir=RAW / cid
        mats=sorted(cdir.glob("*.bed.gz")) + sorted(cdir.glob("*.csv")) if cdir.exists() else []
        if mats:
            for bf in mats:
                artifacts.append(make_artifact(
                    bf.name, corpus.get("landingUrl", ""), REAL, sha256=sha256_file(bf),
                    n_bytes=bf.stat().st_size, local_path=str(bf.relative_to(RAW.parent.parent)),
                    note="resolved via portal/REST API",
                ))
            doc=build_doc(cid, artifacts, note=corpus.get("fetchNote"))
            write_manifest(MANIFESTS, cid, doc)
            return doc

    files=corpus.get("heavyFiles", []) if heavy else corpus.get("files", [])
    if not files:
        note=corpus.get("fetchNote", "open; resolved at ingest, not statically pulled this run")
        artifacts.append(make_artifact(cid, corpus.get("landingUrl", ""), POINTER, note=note))
    elif dry_run:
        for spec in files:
            artifacts.append(make_artifact(spec["name"], spec["url"], POINTER, note="dry-run (not downloaded)"))
    else:
        for spec in files:
            artifacts.append(_fetch_file(cid, spec))

    doc=build_doc(cid, artifacts, note=corpus.get("fetchNote"))
    if heavy:
        (MANIFESTS / f"{cid}.heavy.json").write_text(json.dumps(doc, indent=2, sort_keys=True), encoding="utf-8")
    else:
        write_manifest(MANIFESTS, cid, doc)
    return doc

def run_fetch(*, only=None, all_open=False, all_corpora=False, heavy=None, dry_run=False, list_only=False) -> int:
    corpora=load_corpora()
    if list_only:
        for c in corpora:
            kind="heavy" if c.get("heavyFiles") else ("api" if c.get("apiResolved") else "files")
            print(f"  {c['id']:24s} tier={c['accessTier']:10s} {kind:6s} {c['title']}")
        return 0

    if heavy:
        c=next((x for x in corpora if x["id"] == heavy), None)
        if not c:
            print(f"unknown corpus {heavy!r}")
            return 1
        doc=fetch_corpus(c, heavy=True, dry_run=dry_run)
        print(f"[heavy] {heavy}: {doc['summary']}")
        return 0

    targets: list[dict]
    if all_open:
        targets=[c for c in corpora if c["accessTier"] == "open" and c.get("files")]
    elif all_corpora:
        targets=corpora
    elif only:
        targets=[c for c in corpora if c["id"] == only]
        if not targets:
            print(f"unknown corpus {only!r}")
            return 1
    else:
        print("specify a corpus id, --all-open, --all, --heavy <id>, or --list")
        return 1

    n_real=n_pointer = n_gated = 0
    for c in targets:
        doc=fetch_corpus(c, dry_run=dry_run)
        s=doc["summary"]
        n_real += s["real"]
        n_pointer += s["pointer"]
        n_gated += s["gated"]
        print(f"  {c['id']:24s} REAL={s['real']} POINTER={s['pointer']} GATED={s['gated']}")
    print(f"[fetch] {len(targets)} corpora -> REAL={n_real} POINTER={n_pointer} GATED={n_gated}")
    return 0

def run_verify() -> int:
    from ..core.paths import ROOT
    from ..core.provenance import verify_provenance

    manifests=sorted(MANIFESTS.glob("*.json"))
    manifests=[m for m in manifests if not m.name.endswith(".heavy.json")]
    if not manifests:
        print("[verify] no manifests; run `pdac fetch-data --all-open` first")
        return 1
    ok_all=True
    tot_rehash=0
    for m in manifests:
        doc=json.loads(m.read_text(encoding="utf-8"))
        res=verify_provenance(doc, ROOT)
        tot_rehash += res["real_rehashed"]
        status="OK  " if res["ok"] else "FAIL"
        extra="" if res["ok"] else "  " + "; ".join(f"{f['name']}:{f['reason']}" for f in res["failures"])
        nm="" if res["not_materialized"] == 0 else f" ({res['not_materialized']} not-materialized)"
        print(f"  {status} {m.stem:24s} rehashed={res['real_rehashed']}{nm}{extra}")
        ok_all=ok_all and res["ok"]
    print(f"[verify] {len(manifests)} manifests, {tot_rehash} REAL re-hashed, {'all honest' if ok_all else 'FAILURES'}")
    return 0 if ok_all else 1
