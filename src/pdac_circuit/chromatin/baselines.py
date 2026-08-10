from __future__ import annotations

import csv
from datetime import datetime
import io
import json
from pathlib import Path
import re
import uuid

from .streaming import sha256_file

def _atomic_write_bytes(path: Path,content: bytes) -> None:
    path.parent.mkdir(parents=True,exist_ok=True)
    temporary=path.parent / f".{path.name}.partial-{uuid.uuid4().hex}"
    temporary.write_bytes(content)
    temporary.replace(path)

def _atomic_write_json(path: Path,payload: dict) -> None:
    path.parent.mkdir(parents=True,exist_ok=True)
    temporary=path.parent / f".{path.name}.partial-{uuid.uuid4().hex}"
    temporary.write_text(
        json.dumps(payload,indent=2,sort_keys=True) + "\n",encoding="utf-8"
    )
    temporary.replace(path)

def resolve_enformer_target_map(
    target_metadata_path: str | Path,
    policy_path: str | Path,
    out: str | Path,
    *,
    source_url: str | None = None,
) -> dict:
    metadata_path=Path(target_metadata_path)
    policy_path=Path(policy_path)
    policy=json.loads(policy_path.read_text(encoding="utf-8"))
    if policy.get("schema") != "pdac-circuit.enformer-target-policy/1":
        raise ValueError("invalid Enformer target policy schema")
    rows=list(
        csv.DictReader(
            io.StringIO(metadata_path.read_text(encoding="utf-8")),delimiter="\t"
        )
    )
    if not rows or "index" not in rows[0] or "description" not in rows[0]:
        raise ValueError("Enformer target metadata lacks index/description columns")
    head=str(policy.get("head","human"))
    if head not in {"human","mouse"}:
        raise ValueError(f"unsupported Enformer output head {head!r}")
    allowed_genomes=[str(value) for value in policy.get("allowed_genomes",[])]
    if not allowed_genomes:
        allowed_genomes=["hg38","hg19"] if head == "human" else ["mm10","mm9"]
    head_index_offset=int(policy.get("head_index_offset",0))
    if head_index_offset < 0:
        raise ValueError("Enformer head_index_offset must be nonnegative")
    output_target_count=policy.get("output_target_count")
    if output_target_count is not None and int(output_target_count) <= 0:
        raise ValueError("Enformer output_target_count must be positive")
    resolved=[]
    for rule in policy["rules"]:
        expression=re.compile(rule["description_regex"],flags=re.IGNORECASE)
        matches=[row for row in rows if expression.search(row["description"])]
        minimum=int(rule["minimum_matches"])
        maximum=int(rule["maximum_matches"])
        if not minimum <= len(matches) <= maximum:
            raise ValueError(
                f"rule {rule['name']!r} matched {len(matches)} targets; "
                f"require {minimum}..{maximum}"
            )
        source_indices=[int(row["index"]) for row in matches]
        target_indices=[index - head_index_offset for index in source_indices]
        if any(index < 0 for index in target_indices):
            raise ValueError(
                f"rule {rule['name']!r} includes an index below the registered "
                f"{head} head offset {head_index_offset}"
            )
        if output_target_count is not None and any(
            index >= int(output_target_count) for index in target_indices
        ):
            raise ValueError(
                f"rule {rule['name']!r} exceeds the registered {head} output head"
            )
        if len(target_indices) != len(set(target_indices)):
            raise ValueError(f"rule {rule['name']!r} resolves duplicate head indices")
        resolved.append(
            {
                "name": rule["name"],
                "description_regex": rule["description_regex"],
                "aggregation": rule["aggregation"],
                "target_indices": target_indices,
                "source_indices": source_indices,
                "targets": [
                    {
                        "index": int(row["index"]) - head_index_offset,
                        "source_index": int(row["index"]),
                        "identifier": row.get("identifier",""),
                        "description": row["description"],
                    }
                    for row in matches
                ],
            }
        )
    result={
        "schema": "pdac-circuit.enformer-target-map/1",
        "created_at": datetime.now().astimezone().isoformat(),
        "label_blind": True,
        "source_url": source_url or policy.get("source_url"),
        "source_repository": policy.get("source_repository"),
        "source_commit": policy.get("source_commit"),
        "target_metadata_sha256": sha256_file(metadata_path),
        "policy_sha256": sha256_file(policy_path),
        "head": head,
        "head_index_offset": head_index_offset,
        "output_target_count": (
            int(output_target_count) if output_target_count is not None else None
        ),
        "allowed_genomes": allowed_genomes,
        "rules": resolved,
        "unsupported_zero_shot_assays": policy.get("unsupported_zero_shot_assays",[]),
        "unsupported_policy": policy.get("unsupported_policy"),
    }
    destination=Path(out)
    _atomic_write_json(destination,result)
    return result

def build_enformer_target_map(
    project_root: str | Path,
    *,
    refresh: bool = False,
    out: str | Path | None = None,
    policy_path: str | Path | None = None,
    metadata_path: str | Path | None = None,
) -> dict:
    import requests

    root=Path(project_root)
    policy_path=Path(policy_path) if policy_path else root / "configs" / "enformer_target_policy.json"
    if not policy_path.is_absolute():
        policy_path=root / policy_path
    policy=json.loads(policy_path.read_text(encoding="utf-8"))
    head=str(policy.get("head","human"))
    metadata_path=(
        Path(metadata_path)
        if metadata_path
        else root / "data" / "metadata" / f"enformer_targets_{head}.txt"
    )
    if not metadata_path.is_absolute():
        metadata_path=root / metadata_path
    metadata_path.parent.mkdir(parents=True,exist_ok=True)
    if refresh or not metadata_path.exists():
        response=requests.get(
            policy["source_url"],
            headers={"User-Agent": "pdac-circuit/0.2 (research-use-only)"},
            timeout=90,
        )
        response.raise_for_status()
        _atomic_write_bytes(metadata_path,response.content)
    destination=out or root / "data" / "metadata" / (
        "enformer_target_map.json"
        if head == "human"
        else f"enformer_target_map_{head}.json"
    )
    return resolve_enformer_target_map(
        metadata_path,
        policy_path,
        destination,
        source_url=policy["source_url"],
    )

def resolve_borzoi_target_map(
    target_metadata_path: str | Path,
    policy_path: str | Path,
    out: str | Path,
) -> dict:

    metadata_path=Path(target_metadata_path)
    policy_path=Path(policy_path)
    policy=json.loads(policy_path.read_text(encoding="utf-8"))
    if policy.get("schema") != "pdac-circuit.borzoi-target-policy/1":
        raise ValueError("invalid Borzoi target policy schema")
    rows=list(
        csv.DictReader(
            io.StringIO(metadata_path.read_text(encoding="utf-8")),delimiter="\t"
        )
    )
    if not rows or "description" not in rows[0]:
        raise ValueError("Borzoi target metadata lacks descriptions")
    index_key="index" if "index" in rows[0] else ""
    if index_key not in rows[0]:
        raise ValueError("Borzoi target metadata lacks an index column")
    required_transform={
        "identifier",
        "clip",
        "clip_soft",
        "scale",
        "sum_stat",
        "strand_pair",
        "description",
    }
    if not required_transform <= set(rows[0]):
        raise ValueError("Borzoi target metadata lacks inverse-transform columns")
    indices=[int(row[index_key]) for row in rows]
    if indices != list(range(len(rows))):
        raise ValueError("Borzoi target indices must be unique, ordered, and zero-based")
    strand_pair=[int(row["strand_pair"]) for row in rows]
    if any(index < 0 or index >= len(rows) for index in strand_pair):
        raise ValueError("Borzoi strand-pair indices exceed the human output head")
    resolved=[]
    for rule in policy["rules"]:
        expression=re.compile(rule["description_regex"],flags=re.IGNORECASE)
        matches=[row for row in rows if expression.search(row["description"])]
        minimum=int(rule["minimum_matches"])
        maximum=int(rule["maximum_matches"])
        if not minimum <= len(matches) <= maximum:
            raise ValueError(
                f"rule {rule['name']!r} matched {len(matches)} targets; "
                f"require {minimum}..{maximum}"
            )
        targets=[]
        for row in matches:
            targets.append(
                {
                    "index": int(row[index_key]),
                    "identifier": row["identifier"],
                    "clip": float(row["clip"]),
                    "clip_soft": float(row["clip_soft"]),
                    "scale": float(row["scale"]),
                    "sum_stat": row["sum_stat"],
                    "strand_pair": int(row["strand_pair"]),
                    "description": row["description"],
                }
            )
        resolved.append(
            {
                "name": rule["name"],
                "description_regex": rule["description_regex"],
                "aggregation": rule["aggregation"],
                "target_indices": [row["index"] for row in targets],
                "targets": targets,
            }
        )
    result={
        "schema": "pdac-circuit.borzoi-target-map/1",
        "created_at": datetime.now().astimezone().isoformat(),
        "label_blind": True,
        "head": policy["head"],
        "allowed_genomes": policy["allowed_genomes"],
        "source_url": policy["source_url"],
        "source_repository": policy["source_repository"],
        "source_commit": policy["source_commit"],
        "target_metadata_sha256": sha256_file(metadata_path),
        "policy_sha256": sha256_file(policy_path),
        "input_bp": policy["input_bp"],
        "native_bin_bp": policy["native_bin_bp"],
        "native_output_bins": policy["native_output_bins"],
        "comparison_bin_bp": policy["comparison_bin_bp"],
        "comparison_bins": policy["comparison_bins"],
        "target_count": len(rows),
        "strand_pair_index": strand_pair,
        "rules": resolved,
        "unsupported_zero_shot_assays": policy.get("unsupported_zero_shot_assays",[]),
        "unsupported_policy": policy.get("unsupported_policy"),
    }
    destination=Path(out)
    _atomic_write_json(destination,result)
    return result

def build_borzoi_target_map(
    project_root: str | Path,
    *,
    refresh: bool = False,
    out: str | Path | None = None,
) -> dict:
    import requests

    root=Path(project_root)
    policy_path=root / "configs" / "borzoi_target_policy.json"
    policy=json.loads(policy_path.read_text(encoding="utf-8"))
    metadata_path=root / "data" / "metadata" / "borzoi_targets_human.txt"
    metadata_path.parent.mkdir(parents=True,exist_ok=True)
    if refresh or not metadata_path.exists():
        response=requests.get(
            policy["source_url"],
            headers={"User-Agent": "pdac-circuit/0.2 (research-use-only)"},
            timeout=90,
        )
        response.raise_for_status()
        _atomic_write_bytes(metadata_path,response.content)
    destination=out or root / "data" / "metadata" / "borzoi_target_map.json"
    return resolve_borzoi_target_map(metadata_path,policy_path,destination)
