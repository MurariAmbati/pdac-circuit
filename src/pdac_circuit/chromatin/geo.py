from __future__ import annotations

from datetime import datetime
from html.parser import HTMLParser
import hashlib
import json
from pathlib import Path
import re
import time
from urllib.parse import unquote, urljoin, urlparse

from .streaming import sha256_file
from .locking import exclusive_artifact_lock

GEO_FTP="https://ftp.ncbi.nlm.nih.gov/geo/series"
GSE_RE=re.compile(r"^GSE(\d+)$")

def geo_series_bucket(accession: str) -> str:
    match=GSE_RE.fullmatch(accession.upper())
    if not match:
        raise ValueError(f"not a GEO series accession: {accession!r}")
    digits=match.group(1)
    return f"GSE{digits[:-3]}nnn" if len(digits) > 3 else "GSEnnn"

def supplementary_url(accession: str) -> str:
    accession=accession.upper()
    geo_series_bucket(accession)
    return f"{GEO_FTP}/{geo_series_bucket(accession)}/{accession}/suppl/"

class _LinkParser(HTMLParser):
    def __init__(self):
        super().__init__()
        self.hrefs: list[str]=[]

    def handle_starttag(self, tag: str, attrs):
        if tag.lower() != "a":
            return
        values=dict(attrs)
        if values.get("href"):
            self.hrefs.append(str(values["href"]))

def _supplementary_links(html: str, listing_url: str) -> list[dict]:
    parser=_LinkParser()
    parser.feed(html)
    listing=urlparse(listing_url)
    files={}
    for href in parser.hrefs:
        absolute=urljoin(listing_url, href)
        parsed=urlparse(absolute)
        if parsed.scheme != "https" or parsed.netloc != listing.netloc:
            continue
        if not parsed.path.startswith(listing.path):
            continue
        name=unquote(Path(parsed.path).name)
        if not name or href.endswith("/") or name in {"suppl", "Parent Directory"}:
            continue
        if Path(name).name != name or name in {".", ".."}:
            raise ValueError(f"unsafe GEO supplementary filename {name!r}")
        files[name]={"name": name, "url": absolute, "bytes": None}
    return [files[name] for name in sorted(files)]

def _registered_asset(project_root: Path, accession: str) -> dict:
    registry_path=project_root / "pdac_chromatin_assets.json"
    registry=json.loads(registry_path.read_text(encoding="utf-8"))
    matches=[asset for asset in registry["assets"] if asset["id"] == accession]
    if not matches:
        raise ValueError(f"{accession} is not registered in {registry_path.name}")
    asset=matches[0]
    if asset["status"] == "GATED":
        raise PermissionError(f"{accession} is GATED; this fetcher will not bypass authorization")
    if not accession.startswith("GSE"):
        raise ValueError(f"{accession} is registered but is not a GEO series")
    return asset

def build_geo_plan(
    project_root: str | Path,
    accession: str,
    *,
    out: str | Path | None = None,
    probe_sizes: bool = True,
) -> dict:

    import requests
    from requests.adapters import HTTPAdapter
    from urllib3.util.retry import Retry

    root=Path(project_root)
    accession=accession.upper()
    asset=_registered_asset(root, accession)
    listing_url=supplementary_url(accession)
    headers={
        "User-Agent": "pdac-circuit/0.2 (research-use-only)",
        "Accept-Encoding": "identity",
    }
    session=requests.Session()
    session.mount(
        "https://",
        HTTPAdapter(
            max_retries=Retry(
                total=4,
                connect=4,
                read=4,
                backoff_factor=0.5,
                status_forcelist=(429, 500, 502, 503, 504),
                allowed_methods=frozenset({"GET", "HEAD"}),
            )
        ),
    )
    response=session.get(listing_url, headers=headers, timeout=90)
    response.raise_for_status()
    files=_supplementary_links(response.text, listing_url)
    if not files:
        raise RuntimeError(f"no supplementary files exposed by {listing_url}")
    probe_failures=[]
    if probe_sizes:
        for record in files:
            try:
                head=session.head(
                    record["url"], headers=headers, allow_redirects=True, timeout=90
                )
                if head.ok and head.headers.get("Content-Length", "").isdigit():
                    record["bytes"]=int(head.headers["Content-Length"])
                else:
                    probe_failures.append(
                        {"name": record["name"], "status": head.status_code}
                    )
            except requests.RequestException as exc:
                probe_failures.append(
                    {"name": record["name"], "error": f"{type(exc).__name__}: {exc}"}
                )
    split=str(asset.get("split", ""))
    protected="test" in split.lower() or "holdout" in str(asset.get("role", "")).lower()
    plan={
        "schema": "pdac-circuit.geo-download-plan/1",
        "accession": accession,
        "generated_at": datetime.now().astimezone().isoformat(),
        "listing_url": listing_url,
        "listing_sha256": hashlib.sha256(response.content).hexdigest(),
        "asset": asset,
        "protected_from_training": protected,
        "known_total_bytes": sum(record["bytes"] or 0 for record in files),
        "unknown_size_files": sum(record["bytes"] is None for record in files),
        "size_probe_failures": probe_failures,
        "files": files,
    }
    destination=(
        Path(out)
        if out
        else root / "data" / "manifests" / "studies" / f"{accession}.plan.json"
    )
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(plan, indent=2, sort_keys=True), encoding="utf-8")
    plan["plan_path"]=str(destination)
    return plan

def _download_file(
    url: str,
    destination: Path,
    *,
    remaining_bytes: int,
    expected_bytes: int | None = None,
    max_reconnects: int = 20,
) -> int:
    import requests
    from requests.adapters import HTTPAdapter
    from urllib3.util.retry import Retry

    part=destination.with_suffix(destination.suffix + ".part")
    session=requests.Session()
    session.mount(
        "https://",
        HTTPAdapter(
            max_retries=Retry(
                total=4,
                connect=4,
                read=0,
                backoff_factor=0.5,
                status_forcelist=(429, 500, 502, 503, 504),
                allowed_methods=frozenset({"GET"}),
            )
        ),
    )
    attempts=0
    while True:
        existing=part.stat().st_size if part.exists() else 0
        if expected_bytes is not None and existing == expected_bytes:
            break
        if existing > remaining_bytes:
            raise RuntimeError("partial download exceeds --max-total-gb safety limit")
        if expected_bytes is not None and existing > expected_bytes:
            raise RuntimeError("partial download exceeds pinned remote byte size")
        headers={
            "User-Agent": "pdac-circuit/0.2 (research-use-only)",
            "Accept-Encoding": "identity",
        }
        if existing:
            headers["Range"]=f"bytes={existing}-"
        try:
            response=session.get(
                url,
                headers=headers,
                stream=True,
                timeout=(30, 180),
            )
            response.raise_for_status()
            append=existing > 0 and response.status_code == 206
            if existing and not append:
                existing=0
            written=existing
            with part.open("ab" if append else "wb") as handle:
                for chunk in response.iter_content(chunk_size=4 * 1024 * 1024):
                    if not chunk:
                        continue
                    if written + len(chunk) > remaining_bytes:
                        raise RuntimeError("download would exceed --max-total-gb safety limit")
                    if expected_bytes is not None and written + len(chunk) > expected_bytes:
                        raise RuntimeError("download exceeds pinned remote byte size")
                    handle.write(chunk)
                    handle.flush()
                    written += len(chunk)
        except requests.RequestException:
            ftp_url=_official_ncbi_ftp_fallback(url)
            if ftp_url is not None:
                _download_ftp_partial(
                    ftp_url,
                    part,
                    remaining_bytes=remaining_bytes,
                    expected_bytes=expected_bytes,
                    max_reconnects=max_reconnects,
                )
                break
            attempts += 1
            if attempts > max_reconnects:
                raise
            time.sleep(min(30.0, 0.75 * 2 ** min(attempts, 5)))
            continue
        if expected_bytes is None or written == expected_bytes:
            break
        attempts += 1
        if attempts > max_reconnects:
            raise OSError(f"short download after reconnects: {written} != {expected_bytes}")
        time.sleep(min(30.0, 0.75 * 2 ** min(attempts, 5)))
    written=part.stat().st_size
    part.replace(destination)
    return written

def _official_ncbi_ftp_fallback(url: str) -> str | None:

    parsed=urlparse(url)
    if parsed.scheme != "https" or parsed.hostname != "ftp.ncbi.nlm.nih.gov":
        return None
    return parsed._replace(scheme="ftp").geturl()

def _download_ftp_partial(
    url: str,
    part: Path,
    *,
    remaining_bytes: int,
    expected_bytes: int | None,
    max_reconnects: int,
) -> int:

    from ftplib import FTP, all_errors

    parsed=urlparse(url)
    if parsed.scheme != "ftp" or not parsed.hostname or not parsed.path:
        raise ValueError(f"invalid FTP fallback URL: {url}")
    remote_path=unquote(parsed.path)
    attempts=0
    while True:
        existing=part.stat().st_size if part.exists() else 0
        if expected_bytes is not None and existing == expected_bytes:
            return existing
        if existing > remaining_bytes:
            raise RuntimeError("partial download exceeds --max-total-gb safety limit")
        if expected_bytes is not None and existing > expected_bytes:
            raise RuntimeError("partial download exceeds pinned remote byte size")
        written=existing
        try:
            with FTP() as ftp:
                ftp.connect(parsed.hostname, parsed.port or 21, timeout=30)
                ftp.login()
                ftp.set_pasv(True)
                ftp.voidcmd("TYPE I")
                remote_bytes=ftp.size(remote_path)
                if (
                    expected_bytes is not None
                    and remote_bytes is not None
                    and remote_bytes != expected_bytes
                ):
                    raise RuntimeError(
                        f"pinned remote byte size changed: {remote_bytes} != {expected_bytes}"
                    )
                with part.open("ab" if existing else "wb") as handle:

                    def consume(chunk: bytes) -> None:
                        nonlocal written
                        if written + len(chunk) > remaining_bytes:
                            raise RuntimeError(
                                "download would exceed --max-total-gb safety limit"
                            )
                        if (
                            expected_bytes is not None
                            and written + len(chunk) > expected_bytes
                        ):
                            raise RuntimeError("download exceeds pinned remote byte size")
                        handle.write(chunk)
                        handle.flush()
                        written += len(chunk)

                    ftp.retrbinary(
                        f"RETR {remote_path}",
                        consume,
                        blocksize=4 * 1024 * 1024,
                        rest=existing if existing else None,
                    )
        except all_errors:
            attempts += 1
            if attempts > max_reconnects:
                raise
            time.sleep(min(30.0, 0.75 * 2 ** min(attempts, 5)))
            continue
        if expected_bytes is None or written == expected_bytes:
            return written
        attempts += 1
        if attempts > max_reconnects:
            raise OSError(f"short FTP download after reconnects: {written} != {expected_bytes}")
        time.sleep(min(30.0, 0.75 * 2 ** min(attempts, 5)))

def _fetch_geo_plan_unlocked(
    plan_path: str | Path,
    output_root: str | Path,
    *,
    allow_protected_study: bool = False,
    protected_seal_path: str | Path | None = None,
    protected_release_path: str | Path | None = None,
    project_root: str | Path | None = None,
    max_total_gb: float = 25.0,
) -> dict:

    plan_path=Path(plan_path)
    plan=json.loads(plan_path.read_text(encoding="utf-8"))
    if plan.get("schema") != "pdac-circuit.geo-download-plan/1":
        raise ValueError("invalid GEO plan schema")
    if plan.get("protected_from_training") and not allow_protected_study:
        raise PermissionError(
            "study is a registered test/holdout asset; pass --allow-protected-study only to "
            "retrieve it into the isolated protected directory"
        )
    if plan.get("protected_from_training"):
        from .protected import validate_protected_study_seal

        if protected_seal_path is None:
            raise PermissionError("protected study retrieval requires a pre-access seal")
        if project_root is None:
            raise PermissionError("protected study retrieval requires an explicit project root")
        seal=validate_protected_study_seal(
            project_root, protected_seal_path, accession=str(plan["accession"])
        )
        if not seal["ok"]:
            raise PermissionError(f"protected study seal failed: {seal['failures']}")
        from .protected import validate_protected_metadata_release

        if protected_release_path is None:
            raise PermissionError(
                "protected target retrieval requires a post-training release manifest"
            )
        release=validate_protected_metadata_release(
            project_root,
            protected_release_path,
            accession=str(plan["accession"]),
        )
        if not release["ok"]:
            raise PermissionError(f"protected target release failed: {release['failures']}")
    max_bytes=int(max_total_gb * 1024**3)
    if max_bytes <= 0:
        raise ValueError("max_total_gb must be positive")
    known=int(plan.get("known_total_bytes") or 0)
    if known > max_bytes:
        raise RuntimeError(
            f"known study size {known / 1024**3:.2f} GiB exceeds {max_total_gb:.2f} GiB limit"
        )
    accession=str(plan["accession"])
    category="protected" if plan.get("protected_from_training") else "training_candidate"
    destination_root=Path(output_root) / category / accession
    destination_root.mkdir(parents=True, exist_ok=True)
    records=[]
    total_bytes=0
    manifest_path=destination_root / "manifest.json"
    for item in plan["files"]:
        destination=destination_root / item["name"]
        if destination.exists():
            size=destination.stat().st_size
        else:
            size=_download_file(
                item["url"],
                destination,
                remaining_bytes=max_bytes - total_bytes,
                expected_bytes=item.get("bytes"),
            )
        total_bytes += size
        if total_bytes > max_bytes:
            raise RuntimeError("download exceeded max_total_gb safety limit")
        records.append(
            {
                "name": item["name"],
                "url": item["url"],
                "bytes": size,
                "sha256": sha256_file(destination),
            }
        )
        partial_manifest={
            "schema": "pdac-circuit.geo-study-manifest/1",
            "accession": accession,
            "source_plan_sha256": sha256_file(plan_path),
            "protected_from_training": bool(plan.get("protected_from_training")),
            "complete": len(records) == len(plan["files"]),
            "total_bytes": total_bytes,
            "files": records,
        }
        manifest_path.write_text(
            json.dumps(partial_manifest, indent=2, sort_keys=True), encoding="utf-8"
        )
    return partial_manifest

def fetch_geo_plan(
    plan_path: str | Path,
    output_root: str | Path,
    *,
    allow_protected_study: bool = False,
    protected_seal_path: str | Path | None = None,
    protected_release_path: str | Path | None = None,
    project_root: str | Path | None = None,
    max_total_gb: float = 25.0,
) -> dict:
    plan=json.loads(Path(plan_path).read_text(encoding="utf-8"))
    accession=str(plan.get("accession") or "UNKNOWN")
    category="protected" if plan.get("protected_from_training") else "training_candidate"
    lock=Path(output_root) / category / accession / ".fetch.lock"
    with exclusive_artifact_lock(lock):
        return _fetch_geo_plan_unlocked(
            plan_path,
            output_root,
            allow_protected_study=allow_protected_study,
            protected_seal_path=protected_seal_path,
            protected_release_path=protected_release_path,
            project_root=project_root,
            max_total_gb=max_total_gb,
        )
