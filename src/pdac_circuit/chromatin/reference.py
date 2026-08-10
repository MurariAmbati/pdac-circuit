from __future__ import annotations

import gzip
import hashlib
import json
from pathlib import Path
import shutil
import tarfile
import time
import uuid

from .streaming import IndexedFasta, sha256_file
from .locking import exclusive_artifact_lock

REFERENCE_ASSETS={
    "mm9": {
        "url": "https://hgdownload.soe.ucsc.edu/goldenPath/mm9/bigZips/chromFa.tar.gz",
        "bytes": 859_563_384,
        "md5": "1cc32361e254d7b0145b72ce4d0d723e",
        "source": "UCSC mm9 chromFa bigZips",
        "filename": "mm9.chromFa.tar.gz",
        "format": "chrom_fa_tar_gz",
    },
    "mm10": {
        "url": "https://hgdownload.soe.ucsc.edu/goldenPath/mm10/bigZips/latest/mm10.fa.gz",
        "bytes": 898_545_591,
        "md5": "06f4b9503f923153a73314a46ef81863",
        "source": "UCSC mm10 latest bigZips",
        "filename": "mm10.fa.gz",
        "format": "fa_gz",
    },
    "hg19": {
        "url": "https://hgdownload.soe.ucsc.edu/goldenPath/hg19/bigZips/latest/hg19.fa.gz",
        "bytes": 979_164_575,
        "md5": "7707462fc100c7d987c075bc146b16ae",
        "source": "UCSC hg19 latest bigZips",
        "filename": "hg19.fa.gz",
        "format": "fa_gz",
    }
}

def _download(url: str, destination: Path, expected_bytes: int) -> None:
    import requests
    from requests.adapters import HTTPAdapter
    from urllib3.util.retry import Retry

    part=destination.with_suffix(destination.suffix + ".part")
    session=requests.Session()
    session.mount(
        "https://",
        HTTPAdapter(
            max_retries=Retry(
                total=6,
                connect=6,
                read=0,
                backoff_factor=1.0,
                status_forcelist=(429, 500, 502, 503, 504),
                allowed_methods=frozenset({"GET"}),
            )
        ),
    )
    attempts=0
    while True:
        existing=part.stat().st_size if part.exists() else 0
        if existing == expected_bytes:
            break
        if existing > expected_bytes:
            raise RuntimeError(f"partial reference is larger than expected: {part}")
        headers={
            "User-Agent": "pdac-circuit/0.2 (research-use-only)",
            "Accept-Encoding": "identity",
        }
        if existing:
            headers["Range"]=f"bytes={existing}-"
        try:
            response=session.get(url, headers=headers, stream=True, timeout=(30, 180))
            response.raise_for_status()
            append=existing > 0 and response.status_code == 206
            if existing and not append:
                existing=0
            written=existing
            with part.open("ab" if append else "wb") as handle:
                for chunk in response.iter_content(chunk_size=8 * 1024 * 1024):
                    if not chunk:
                        continue
                    written += len(chunk)
                    if written > expected_bytes:
                        raise RuntimeError("reference download exceeds pinned byte size")
                    handle.write(chunk)
                    handle.flush()
        except requests.RequestException:
            attempts += 1
            if attempts > 20:
                raise
            time.sleep(min(30.0, 0.75 * 2 ** min(attempts, 5)))
            continue
        if written == expected_bytes:
            break
        attempts += 1
        if attempts > 20:
            raise OSError(f"short reference download: {written} != {expected_bytes}")
        time.sleep(min(30.0, 0.75 * 2 ** min(attempts, 5)))
    part.replace(destination)

def _md5_file(path: Path, chunk_bytes: int = 8 * 1024 * 1024) -> str:
    digest=hashlib.md5(usedforsecurity=False)
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(chunk_bytes), b""):
            digest.update(chunk)
    return digest.hexdigest()

def build_fasta_index(fasta_path: str | Path) -> Path:

    fasta_path=Path(fasta_path)
    index_path=Path(str(fasta_path) + ".fai")
    records: list[str]=[]
    name=None
    length=0
    sequence_offset=0
    line_bases=0
    line_width=0

    def finish() -> None:
        if name is not None:
            if length <= 0 or line_bases <= 0:
                raise ValueError(f"FASTA record {name!r} has no sequence")
            records.append(
                f"{name}\t{length}\t{sequence_offset}\t{line_bases}\t{line_width}\n"
            )

    with fasta_path.open("rb") as handle:
        while True:
            line_start=handle.tell()
            line=handle.readline()
            if not line:
                break
            if line.startswith(b">"):
                finish()
                header=line[1:].strip().split(maxsplit=1)[0]
                name=header.decode("ascii")
                length=0
                sequence_offset=handle.tell()
                line_bases=0
                line_width=0
                continue
            if name is None:
                raise ValueError(f"sequence before FASTA header at byte {line_start}")
            bases=line.rstrip(b"\r\n")
            if not bases:
                continue
            if line_bases == 0:
                line_bases=len(bases)
                line_width=len(line)
            length += len(bases)
    finish()
    temporary=index_path.with_suffix(index_path.suffix + f".partial-{uuid.uuid4().hex}")
    temporary.write_text("".join(records), encoding="ascii")
    temporary.replace(index_path)
    return index_path

def _materialize_reference_unlocked(
    project_root: str | Path,
    genome: str,
    *,
    keep_compressed: bool = True,
) -> dict:
    root=Path(project_root)
    if genome not in REFERENCE_ASSETS:
        raise ValueError(f"unsupported downloadable reference {genome!r}")
    asset=REFERENCE_ASSETS[genome]
    destination=root / "data" / "raw" / f"{genome}-ref"
    destination.mkdir(parents=True, exist_ok=True)
    compressed=destination / asset["filename"]
    fasta=destination / f"{genome}.fa"
    manifest_path=destination / "manifest.json"

    if not compressed.exists():
        _download(asset["url"], compressed, asset["bytes"])
    if compressed.stat().st_size != asset["bytes"]:
        raise OSError("reference compressed byte size does not match pinned registry")
    observed_md5=_md5_file(compressed)
    if observed_md5 != asset["md5"]:
        raise OSError(f"reference MD5 mismatch: {observed_md5} != {asset['md5']}")

    if not fasta.exists():
        temporary=fasta.with_suffix(fasta.suffix + f".partial-{uuid.uuid4().hex}")
        if asset["format"] == "fa_gz":
            with gzip.open(compressed, "rb") as source, temporary.open("wb") as output:
                shutil.copyfileobj(source, output, length=16 * 1024 * 1024)
        elif asset["format"] == "chrom_fa_tar_gz":
            with tarfile.open(compressed, mode="r:gz") as archive:
                members=[]
                for member in archive:
                    path=Path(member.name)
                    if member.isdir():
                        if path.is_absolute() or ".." in path.parts:
                            raise ValueError(f"unsafe reference directory {member.name!r}")
                        continue
                    if (
                        not member.isfile()
                        or path.is_absolute()
                        or ".." in path.parts
                        or path.suffix.lower() != ".fa"
                    ):
                        raise ValueError(f"unsafe or unexpected reference member {member.name!r}")
                    members.append(member)

                def chromosome_key(member):
                    label=Path(member.name).stem.removeprefix("chr")
                    if label.isdigit():
                        return (0, int(label), "")
                    order={"X": 23, "Y": 24, "M": 25}
                    if label in order:
                        return (0, order[label], "")
                    return (1, 0, label)

                with temporary.open("wb") as output:
                    for member in sorted(members, key=chromosome_key):
                        source=archive.extractfile(member)
                        if source is None:
                            raise OSError(f"cannot read reference member {member.name!r}")
                        with source:
                            shutil.copyfileobj(source, output, length=16 * 1024 * 1024)
        else:
            raise ValueError(f"unsupported reference archive format {asset['format']!r}")
        temporary.replace(fasta)
    index=build_fasta_index(fasta)
    reader=IndexedFasta(fasta, index)
    reader.assert_genome(genome)
    manifest={
        "schema": "pdac-circuit.reference-genome/1",
        "genome": genome,
        "source": asset["source"],
        "url": asset["url"],
        "compressed_bytes": compressed.stat().st_size,
        "compressed_md5": observed_md5,
        "compressed_sha256": sha256_file(compressed),
        "fasta_bytes": fasta.stat().st_size,
        "fasta_sha256": sha256_file(fasta),
        "fai_sha256": sha256_file(index),
        "chromosomes": len(reader.index),
        "signature_valid": True,
        "compressed_retained": keep_compressed,
    }
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")
    if not keep_compressed:
        compressed.unlink()
    manifest["manifest_path"]=str(manifest_path.relative_to(root))
    return manifest

def materialize_reference(
    project_root: str | Path,
    genome: str,
    *,
    keep_compressed: bool = True,
) -> dict:
    root=Path(project_root)
    destination=root / "data" / "raw" / f"{genome}-ref"
    with exclusive_artifact_lock(destination / ".materialize.lock"):
        return _materialize_reference_unlocked(
            root,
            genome,
            keep_compressed=keep_compressed,
        )
