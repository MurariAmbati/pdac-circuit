from __future__ import annotations

from contextlib import contextmanager
from dataclasses import asdict, dataclass
import hashlib
import json
import os
from pathlib import Path
import random
from typing import Iterable, Iterator, Mapping, Sequence
import uuid

import numpy as np

@dataclass(frozen=True)
class FastaIndexRecord:
    length: int
    offset: int
    line_bases: int
    line_width: int

class IndexedFasta:

    def __init__(self, fasta_path: str | Path, index_path: str | Path | None = None):
        self.fasta_path=Path(fasta_path)
        self.index_path=Path(index_path) if index_path else Path(str(self.fasta_path) + ".fai")
        if not self.fasta_path.exists():
            raise FileNotFoundError(self.fasta_path)
        if not self.index_path.exists():
            raise FileNotFoundError(
                f"missing FASTA index {self.index_path}; run `samtools faidx {self.fasta_path}`"
            )
        self.index: dict[str, FastaIndexRecord]={}
        for line in self.index_path.read_text(encoding="utf-8").splitlines():
            name, length, offset, line_bases, line_width, *_=line.split("\t")
            self.index[name]=FastaIndexRecord(
                int(length), int(offset), int(line_bases), int(line_width)
            )
        self._handle=None

    def __getstate__(self):
        state=dict(self.__dict__)
        state["_handle"]=None
        return state

    def _file(self):
        if self._handle is None or self._handle.closed:
            self._handle=self.fasta_path.open("rb")
        return self._handle

    def fetch(self, chrom: str, start: int, end: int) -> str:
        if chrom not in self.index:
            raise KeyError(f"chromosome {chrom!r} not found in FASTA index")
        rec=self.index[chrom]
        if start < 0 or end < start or end > rec.length:
            raise ValueError(f"invalid {chrom}:{start}-{end} for chromosome length {rec.length}")
        if start == end:
            return ""
        handle=self._file()
        chunks=[]
        cursor=start
        while cursor < end:
            line_pos=cursor % rec.line_bases
            take=min(end - cursor, rec.line_bases - line_pos)
            byte_offset=rec.offset + (cursor // rec.line_bases) * rec.line_width + line_pos
            handle.seek(byte_offset)
            chunks.append(handle.read(take))
            cursor += take
        return b"".join(chunks).decode("ascii").upper()

    def chrom_sizes(self) -> dict[str, int]:
        return {name: record.length for name, record in self.index.items()}

    def assert_genome(self, genome: str) -> None:

        signatures={
            "hg38": {"chr1": 248_956_422, "chr22": 50_818_468},
            "hg19": {"chr1": 249_250_621, "chr22": 51_304_566},
            "mm10": {"chr1": 195_471_971, "chr19": 61_431_566},
            "mm9": {"chr1": 197_195_432, "chr19": 61_342_430},
        }
        if genome not in signatures:
            raise ValueError(f"unsupported reference genome {genome!r}")
        observed=self.chrom_sizes()
        mismatches={
            chrom: {"expected": length, "observed": observed.get(chrom)}
            for chrom, length in signatures[genome].items()
            if observed.get(chrom) != length
        }
        if mismatches:
            raise ValueError(
                f"FASTA {self.fasta_path} is not registered {genome}; signature mismatch "
                f"{mismatches}"
            )

_ONE_HOT=np.zeros((256, 4), dtype=np.float32)
for _byte, _index in zip(b"ACGTacgt", (0, 1, 2, 3, 0, 1, 2, 3), strict=True):
    _ONE_HOT[_byte, _index]=1.0

def one_hot_sequence(sequence: str) -> np.ndarray:
    encoded=np.frombuffer(sequence.encode("ascii"), dtype=np.uint8)
    return _ONE_HOT[encoded].T.copy()

def sha256_file(path: str | Path, chunk_bytes: int = 8 * 1024 * 1024) -> str:
    digest=hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(chunk_bytes), b""):
            digest.update(chunk)
    return digest.hexdigest()

def shard_collection_fingerprint(paths: Sequence[str | Path]) -> dict:

    resolved=sorted(Path(path).resolve() for path in paths)
    if not resolved or len(set(resolved)) != len(resolved):
        raise ValueError("shard collection must be nonempty and unique")
    common=Path(os.path.commonpath([str(path.parent) for path in resolved]))
    digest=hashlib.sha256()
    total_bytes=0
    for path in resolved:
        if not path.is_file() or path.suffix.lower() != ".npz":
            raise ValueError(f"training shard is not a materialized NPZ: {path}")
        relative=path.relative_to(common).as_posix()
        size=path.stat().st_size
        file_digest=sha256_file(path)
        digest.update(relative.encode("utf-8"))
        digest.update(b"\0")
        digest.update(str(size).encode("ascii"))
        digest.update(b"\0")
        digest.update(file_digest.encode("ascii"))
        digest.update(b"\n")
        total_bytes += size
    return {
        "schema": "pdac-circuit.shard-collection-fingerprint/1",
        "files": len(resolved),
        "bytes": total_bytes,
        "common_root": str(common),
        "sha256": digest.hexdigest(),
    }

@dataclass(frozen=True)
class TrackSpec:
    accession: str
    path: str
    assay_features: tuple[float, ...]
    state_features: tuple[float, ...]
    perturbation_features: tuple[float, ...]
    sample_group: str
    study: str
    released: str
    disease: bool
    source_sha256: str | None=None
    split_role: str="train_state"
    genome: str="hg38"
    organism: str="Homo sapiens"
    sample_accession: str=""
    biological_state: str="unspecified"
    perturbation_label: str="none"
    metadata_sha256: str | None=None
    biological_replicate: str=""
    pair_group: str=""
    pair_relation: str="unpaired"
    pair_control_family: str=""

    def validate(self) -> None:
        if not self.accession or not self.sample_group or not self.study:
            raise ValueError("accession, sample_group, and study are required")
        if not Path(self.path).exists():
            raise FileNotFoundError(self.path)
        if self.source_sha256 is not None and (
            len(self.source_sha256) != 64
            or any(char not in "0123456789abcdef" for char in self.source_sha256.lower())
        ):
            raise ValueError("source_sha256 must be a 64-character hexadecimal digest")
        for label, vector in (
            ("assay", self.assay_features),
            ("state", self.state_features),
            ("perturbation", self.perturbation_features),
        ):
            if not vector or not np.isfinite(np.asarray(vector, dtype=float)).all():
                raise ValueError(f"{label} feature vector must be nonempty and finite")
        if self.split_role not in {
            "train_state",
            "validation_study",
            "held_out_state",
            "external_study",
            "temporal_test",
        }:
            raise ValueError(f"unsupported split_role {self.split_role!r}")
        organisms={
            "hg38": "Homo sapiens",
            "hg19": "Homo sapiens",
            "mm10": "Mus musculus",
            "mm9": "Mus musculus",
        }
        if self.genome not in organisms:
            raise ValueError(f"unsupported genome {self.genome!r}")
        if self.organism != organisms[self.genome]:
            raise ValueError(
                f"genome {self.genome} requires organism {organisms[self.genome]!r}, "
                f"not {self.organism!r}"
            )
        relations={
            "unpaired",
            "control",
            "intervention",
            "state_reference",
            "state_treatment",
        }
        if self.pair_relation not in relations:
            raise ValueError(f"unsupported pair_relation {self.pair_relation!r}")
        if self.pair_relation != "unpaired" and not self.pair_group:
            raise ValueError("paired TrackSpec requires an explicit pair_group")
        control_families={
            "",
            "unperturbed",
            "mscv_overexpression",
            "mire_shrna",
            "lentiviral_crispr",
        }
        if self.pair_control_family not in control_families:
            raise ValueError(
                f"unsupported pair_control_family {self.pair_control_family!r}"
            )

@dataclass(frozen=True)
class Window:
    chrom: str
    start: int
    end: int

def split_for_window(track: TrackSpec, window: Window) -> str:
    if track.split_role == "validation_study":
        return "validation"
    if track.split_role == "external_study":
        return "external_study_test"
    if track.split_role == "temporal_test":
        return "temporal_test"
    if track.split_role == "held_out_state":
        return "joint_locus_state_test" if window.chrom in {"chr8", "chr9"} else "state_test"
    if window.chrom in {"chr6", "chr7"}:
        return "validation"
    if window.chrom in {"chr8", "chr9"}:
        return "locus_test"
    return "train"

def genome_windows(
    chrom_sizes: dict[str, int],
    *,
    sequence_length: int,
    stride: int | None = None,
    chromosomes: Sequence[str] | None = None,
    max_windows: int | None = None,
    seed: int = 20_260_620,
) -> Iterator[Window]:

    stride=stride or sequence_length
    if sequence_length < 1 or stride < 1:
        raise ValueError("sequence_length and stride must be positive")
    if max_windows is not None and max_windows < 1:
        raise ValueError("max_windows must be positive when provided")
    chromosomes=chromosomes or tuple(f"chr{i}" for i in range(1, 23)) + ("chrX",)

    def all_windows() -> Iterator[Window]:
        for chrom in chromosomes:
            length=chrom_sizes.get(chrom, 0)
            for start in range(0, max(0, length - sequence_length + 1), stride):
                yield Window(chrom, start, start + sequence_length)

    if max_windows is None:
        return all_windows()
    rng=random.Random(seed)
    reservoir: list[Window]=[]
    for index, window in enumerate(all_windows()):
        if index < max_windows:
            reservoir.append(window)
        else:
            replacement=rng.randrange(index + 1)
            if replacement < max_windows:
                reservoir[replacement]=window
    reservoir.sort(key=lambda window: (window.chrom, window.start))
    return iter(reservoir)

@contextmanager
def _open_bigwig(path: str | Path):

    try:
        import pyBigWig

        reader=pyBigWig.open(str(path))
        backend="pyBigWig"
    except ImportError:
        try:
            import pybigtools
        except ImportError as exc:
            raise RuntimeError(
                "chromatin compilation requires pyBigWig or pybigtools"
            ) from exc
        reader=pybigtools.open(str(path))
        backend="pybigtools"
    try:
        yield reader, backend
    finally:
        reader.close()

def _extract_profile(bigwig, backend: str, window: Window, n_bins: int) -> tuple[np.ndarray, np.ndarray]:
    if backend == "pyBigWig":
        values=bigwig.stats(
            window.chrom,
            window.start,
            window.end,
            type="mean",
            nBins=n_bins,
            exact=True,
        )
    else:
        values=bigwig.values(
            window.chrom,
            window.start,
            window.end,
            bins=n_bins,
            summary="mean",
            exact=True,
            fillna=None,
        )
    profile=np.asarray([np.nan if value is None else value for value in values], dtype=np.float32)
    valid=np.isfinite(profile)
    return np.nan_to_num(profile, nan=0.0, posinf=0.0, neginf=0.0), valid

def _assert_bigwig_native_genome(bigwig, track: TrackSpec) -> None:
    try:
        chroms=dict(bigwig.chroms())
    except Exception as exc:
        raise RuntimeError(f"cannot read chromosome dictionary from {track.path}") from exc
    expected_chr1={
        "hg38": 248_956_422,
        "hg19": 249_250_621,
        "mm10": 195_471_971,
        "mm9": 197_195_432,
    }[track.genome]
    if chroms.get("chr1") != expected_chr1:
        raise ValueError(
            f"bigWig {track.accession} does not match registered {track.genome}: "
            f"chr1={chroms.get('chr1')}, expected {expected_chr1}"
        )

def compile_bigwig_track(
    track: TrackSpec,
    windows: Iterable[Window],
    output_dir: str | Path,
    *,
    bin_size: int = 128,
    windows_per_shard: int = 64,
    negative_keep_probability: float = 0.05,
    signal_epsilon: float = 1e-6,
    seed: int = 20_260_620,
    sequence_length: int | None = None,
) -> dict:

    track.validate()
    output_root=Path(output_dir)
    output_dir=output_root / track.accession
    if output_dir.exists():
        raise FileExistsError(
            f"compiled track destination already exists: {output_dir}; use a new output root"
        )
    output_root.mkdir(parents=True, exist_ok=True)
    source_hash=sha256_file(track.path)
    if track.source_sha256 is not None and source_hash != track.source_sha256.lower():
        raise ValueError(
            f"source sha256 mismatch for {track.accession}: "
            f"expected {track.source_sha256.lower()}, observed {source_hash}"
        )
    temporary=output_root / f"{track.accession}.partial-{uuid.uuid4().hex}"
    temporary.mkdir(parents=True)
    if not 0 <= negative_keep_probability <= 1:
        raise ValueError("negative_keep_probability must be in [0, 1]")
    rng=np.random.default_rng(seed)
    batch: list[tuple[Window, np.ndarray, np.ndarray]]=[]
    shard_records=[]
    n_seen=n_kept=0

    def flush() -> None:
        nonlocal batch, n_kept
        if not batch:
            return
        shard_index=len(shard_records)
        path=temporary / f"shard-{shard_index:06d}.npz"
        np.savez_compressed(
            path,
            example_id=np.asarray(
                [
                    f"{track.accession}:{row[0].chrom}:{row[0].start}:{row[0].end}"
                    for row in batch
                ]
            ),
            accession=np.repeat(track.accession, len(batch)),
            sample_group=np.repeat(track.sample_group, len(batch)),
            study=np.repeat(track.study, len(batch)),
            genome=np.repeat(track.genome, len(batch)),
            organism=np.repeat(track.organism, len(batch)),
            pair_group=np.repeat(track.pair_group, len(batch)),
            pair_relation=np.repeat(track.pair_relation, len(batch)),
            split=np.asarray([split_for_window(track, row[0]) for row in batch]),
            chrom=np.asarray([row[0].chrom for row in batch]),
            start=np.asarray([row[0].start for row in batch], dtype=np.int64),
            end=np.asarray([row[0].end for row in batch], dtype=np.int64),
            target=np.stack([row[1] for row in batch]).astype(np.float16),
            valid=np.stack([row[2] for row in batch]).astype(np.uint8),
            assay_features=np.repeat(
                np.asarray(track.assay_features, dtype=np.float16)[None, :], len(batch), axis=0
            ),
            state_features=np.repeat(
                np.asarray(track.state_features, dtype=np.float16)[None, :], len(batch), axis=0
            ),
            perturbation_features=np.repeat(
                np.asarray(track.perturbation_features, dtype=np.float16)[None, :],
                len(batch),
                axis=0,
            ),
            disease_mask=np.full(len(batch), int(track.disease), dtype=np.uint8),
            healthy_mask=np.full(len(batch), int(not track.disease), dtype=np.uint8),
        )
        n_kept += len(batch)
        shard_records.append(
            {
                "path": path.name,
                "examples": len(batch),
                "bytes": path.stat().st_size,
                "sha256": sha256_file(path),
            }
        )
        batch=[]

    with _open_bigwig(track.path) as (bigwig, backend):
        _assert_bigwig_native_genome(bigwig, track)
        for window in windows:
            n_seen += 1
            profile, valid=_extract_profile(
                bigwig, backend, window, (window.end - window.start) // bin_size
            )
            if not valid.any():
                continue
            negative=float(profile[valid].mean()) <= signal_epsilon
            if negative and rng.random() > negative_keep_probability:
                continue
            batch.append((window, profile, valid))
            if len(batch) >= windows_per_shard:
                flush()
        flush()

    manifest={
        "schema": "pdac-circuit.chromatin-shards/3",
        "track": {**asdict(track), "source_sha256": source_hash},
        "bigwig_backend": backend,
        "native_genome_validated": True,
        "bin_size": bin_size,
        "sequence_length": sequence_length,
        "windows_seen": n_seen,
        "windows_kept": n_kept,
        "negative_keep_probability": negative_keep_probability,
        "shards": shard_records,
    }
    manifest_path=temporary / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")
    temporary.replace(output_dir)
    manifest["output_dir"]=str(output_dir)
    return manifest

class ChromatinShardStream:

    def __init__(
        self,
        shard_paths: Sequence[str | Path],
        fasta_path: str | Path | Mapping[str, str | Path],
        *,
        shuffle: bool = True,
        seed: int = 20_260_620,
        epoch: int = 0,
        include_splits: set[str] | frozenset[str] | None = None,
        include_example_ids: set[str] | frozenset[str] | None = None,
        include_studies: set[str] | frozenset[str] | None = None,
        exclude_studies: set[str] | frozenset[str] | None = None,
        validation_only_studies: set[str] | frozenset[str] | None = None,
        include_targets: bool = True,
        conditioning_dimensions: Mapping[str, int] | None = None,
    ):
        self.shard_paths=tuple(Path(path) for path in shard_paths)
        if isinstance(fasta_path, Mapping):
            self.fasta_paths={str(genome): Path(path) for genome, path in fasta_path.items()}
        else:
            self.fasta_paths={"hg38": Path(fasta_path)}
        self.shuffle=shuffle
        self.seed=seed
        self.epoch=epoch
        self.include_splits=frozenset(include_splits) if include_splits is not None else None
        self.include_example_ids=(
            frozenset(include_example_ids) if include_example_ids is not None else None
        )
        if self.include_example_ids is not None and not self.include_example_ids:
            raise ValueError("include_example_ids cannot be empty")
        self.include_studies=(
            frozenset(str(study).upper() for study in include_studies)
            if include_studies is not None
            else None
        )
        self.exclude_studies=frozenset(
            str(study).upper() for study in (exclude_studies or ())
        )
        self.validation_only_studies=frozenset(
            str(study).upper() for study in (validation_only_studies or ())
        )
        if self.include_studies is not None and not self.include_studies:
            raise ValueError("include_studies cannot be empty")
        if self.include_studies is not None and self.include_studies & self.exclude_studies:
            raise ValueError("included and excluded studies must be disjoint")
        self.include_targets=bool(include_targets)
        self.conditioning_dimensions=dict(conditioning_dimensions or {})
        allowed_dimensions={
            "assay_features",
            "state_features",
            "perturbation_features",
        }
        if set(self.conditioning_dimensions) - allowed_dimensions or any(
            not isinstance(value, int) or value < 1
            for value in self.conditioning_dimensions.values()
        ):
            raise ValueError("conditioning dimensions must be positive registered feature sizes")

    def __iter__(self) -> Iterator[dict]:
        fasta_readers: dict[str, IndexedFasta]={}
        order=list(range(len(self.shard_paths)))
        rng=np.random.default_rng(self.seed + self.epoch)
        if self.shuffle:
            rng.shuffle(order)
        try:
            import torch

            worker=torch.utils.data.get_worker_info()
        except ImportError:
            worker=None
        if worker is not None:
            order=order[worker.id :: worker.num_workers]
        for shard_index in order:
            with np.load(self.shard_paths[shard_index], allow_pickle=False) as shard:
                conditioning={}
                for key, expected in self.conditioning_dimensions.items():
                    values=shard[key]
                    if values.ndim != 2:
                        raise ValueError(f"{key} must have shape (examples, features)")
                    if values.shape[1] == expected:
                        conditioning[key]=values
                    elif (
                        key == "perturbation_features"
                        and values.shape[1] < expected
                        and not np.any(values)
                    ):
                        conditioning[key]=np.zeros(
                            (values.shape[0], expected), dtype=values.dtype
                        )
                    else:
                        raise ValueError(
                            f"{self.shard_paths[shard_index]} {key} has "
                            f"{values.shape[1]} features; expected {expected}"
                        )
                examples=list(range(len(shard["start"])))
                if self.shuffle:
                    rng.shuffle(examples)
                for i in examples:
                    genome=str(shard["genome"][i]) if "genome" in shard.files else "hg38"
                    chrom=str(shard["chrom"][i])
                    start, end=int(shard["start"][i]), int(shard["end"][i])
                    accession=(
                        str(shard["accession"][i])
                        if "accession" in shard.files
                        else self.shard_paths[shard_index].parent.name
                    )
                    split=(
                        str(shard["split"][i])
                        if "split" in shard.files
                        else (
                            "validation"
                            if chrom in {"chr6", "chr7"}
                            else "locus_test"
                            if chrom in {"chr8", "chr9"}
                            else "train"
                        )
                    )
                    study=(
                        str(shard["study"][i]).upper()
                        if "study" in shard.files
                        else "LEGACY_SHARD"
                    )
                    if study in self.validation_only_studies and split != "validation":
                        raise ValueError(
                            f"validation-only study {study} contains forbidden split {split!r} "
                            f"in {self.shard_paths[shard_index]}"
                        )
                    if self.include_studies is not None and study not in self.include_studies:
                        continue
                    if study in self.exclude_studies:
                        continue
                    if self.include_splits is not None and split not in self.include_splits:
                        continue
                    example_id=(
                        str(shard["example_id"][i])
                        if "example_id" in shard.files
                        else f"{accession}:{chrom}:{start}:{end}"
                    )
                    if (
                        self.include_example_ids is not None
                        and example_id not in self.include_example_ids
                    ):
                        continue
                    if genome not in self.fasta_paths:
                        raise KeyError(
                            f"shard requires {genome} FASTA but configured references are "
                            f"{sorted(self.fasta_paths)}"
                        )
                    if genome not in fasta_readers:
                        fasta_readers[genome]=IndexedFasta(self.fasta_paths[genome])
                        fasta_readers[genome].assert_genome(genome)
                    fasta=fasta_readers[genome]
                    example={
                        "sequence": one_hot_sequence(fasta.fetch(chrom, start, end)),
                        "assay_features": conditioning.get(
                            "assay_features", shard["assay_features"]
                        )[i].astype(np.float32),
                        "state_features": conditioning.get(
                            "state_features", shard["state_features"]
                        )[i].astype(np.float32),
                        "perturbation_features": conditioning.get(
                            "perturbation_features", shard["perturbation_features"]
                        )[i].astype(np.float32),
                        "disease_mask": np.float32(shard["disease_mask"][i]),
                        "example_id": example_id,
                        "accession": accession,
                        "sample_group": (
                            str(shard["sample_group"][i])
                            if "sample_group" in shard.files
                            else accession
                        ),
                        "study": study,
                        "genome": genome,
                        "organism": (
                            str(shard["organism"][i])
                            if "organism" in shard.files
                            else "Homo sapiens"
                        ),
                        "pair_group": (
                            str(shard["pair_group"][i]) if "pair_group" in shard.files else ""
                        ),
                        "pair_relation": (
                            str(shard["pair_relation"][i])
                            if "pair_relation" in shard.files
                            else "unpaired"
                        ),
                        "split": split,
                        "chrom": chrom,
                        "start": start,
                        "end": end,
                    }
                    if self.include_targets:
                        example["target"]=shard["target"][i].astype(np.float32)
                        example["signal_mask"]=shard["valid"][i].astype(bool)
                    for optional in (
                        "healthy_mask",
                        "paired_delta",
                        "pair_mask",
                        "perturbation_delta",
                        "perturbation_mask",
                    ):
                        if optional in shard.files:
                            value=shard[optional][i]
                            example[optional]=value.astype(np.float32)
                    yield example

class LocalTiledChromatinStream:

    def __init__(self, stream: ChromatinShardStream, *, tile_bp: int, bin_size: int):
        if tile_bp < 1 or bin_size < 1 or tile_bp % bin_size:
            raise ValueError("local tile length must be positive and divisible by bin size")
        self.stream=stream
        self.tile_bp=int(tile_bp)
        self.bin_size=int(bin_size)

    @property
    def epoch(self) -> int:
        return self.stream.epoch

    @epoch.setter
    def epoch(self, value: int) -> None:
        self.stream.epoch=int(value)

    def __iter__(self) -> Iterator[dict]:
        profile_keys=(
            "target",
            "signal_mask",
            "paired_delta",
            "pair_mask",
            "perturbation_delta",
            "perturbation_mask",
        )
        tile_bins=self.tile_bp // self.bin_size
        for example in self.stream:
            sequence_bp=int(example["sequence"].shape[-1])
            if sequence_bp % self.tile_bp:
                raise ValueError(
                    f"source window {sequence_bp} is not divisible by tile {self.tile_bp}"
                )
            parent_id=str(example["example_id"])
            parent_start=int(example["start"])
            for tile_index, bp_start in enumerate(range(0, sequence_bp, self.tile_bp)):
                bp_end=bp_start + self.tile_bp
                bin_start=tile_index * tile_bins
                bin_end=bin_start + tile_bins
                tiled=dict(example)
                tiled["sequence"]=example["sequence"][:, bp_start:bp_end]
                for key in profile_keys:
                    value=example.get(key)
                    if value is not None and np.ndim(value) > 0 and value.shape[-1] > 1:
                        tiled[key]=value[..., bin_start:bin_end]
                tiled["parent_example_id"]=parent_id
                tiled["tile_index"]=tile_index
                tiled["tiles_in_parent"]=sequence_bp // self.tile_bp
                tiled["example_id"]=f"{parent_id}:tile-{tile_index:04d}"
                tiled["start"]=parent_start + bp_start
                tiled["end"]=parent_start + bp_end
                yield tiled

def as_torch_iterable(stream: ChromatinShardStream):

    import torch

    class _Dataset(torch.utils.data.IterableDataset):
        def __iter__(self):
            return iter(stream)

        def set_epoch(self, epoch: int) -> None:
            stream.epoch=int(epoch)

    return _Dataset()

def collate_chromatin(examples: list[dict]) -> dict:
    import torch

    tensor_keys=[
        "sequence",
        "assay_features",
        "state_features",
        "perturbation_features",
        "disease_mask",
    ]
    for key in ("target", "signal_mask"):
        if all(key in example for example in examples):
            tensor_keys.append(key)
    batch={key: torch.from_numpy(np.stack([ex[key] for ex in examples])) for key in tensor_keys}
    for key in (
        "healthy_mask",
        "paired_delta",
        "pair_mask",
        "perturbation_delta",
        "perturbation_mask",
    ):
        if all(key in example for example in examples):
            batch[key]=torch.from_numpy(np.stack([example[key] for example in examples]))
    for key in (
        "example_id",
        "parent_example_id",
        "tile_index",
        "tiles_in_parent",
        "accession",
        "sample_group",
        "study",
        "genome",
        "organism",
        "pair_group",
        "pair_relation",
        "split",
        "chrom",
        "start",
        "end",
    ):
        if all(key in example for example in examples):
            batch[key]=[ex[key] for ex in examples]
    return batch
