from __future__ import annotations

from collections import Counter
import json
from pathlib import Path

def _bytes_under(path: Path) -> int:
    return sum(item.stat().st_size for item in path.rglob("*") if item.is_file())

def build_inventory(project_root: str | Path) -> dict:
    root = Path(project_root)
    manifests = root / "data" / "manifests"
    raw = root / "data" / "raw"
    corpora = []
    total_artifacts = total_manifest_bytes = 0
    for manifest_path in sorted(manifests.glob("*.json")):
        payload = json.loads(manifest_path.read_text(encoding="utf-8"))
        artifacts = payload.get("artifacts", [])
        size = sum(int(artifact.get("bytes") or 0) for artifact in artifacts)
        total_artifacts += len(artifacts)
        total_manifest_bytes += size
        corpora.append(
            {
                "corpus": payload.get("corpus", manifest_path.stem),
                "manifest": manifest_path.name,
                "artifacts": len(artifacts),
                "bytes": size,
                "gb": round(size / 1024**3, 3),
                "real": payload.get("summary", {}).get("real", 0),
                "pointer": payload.get("summary", {}).get("pointer", 0),
                "gated": payload.get("summary", {}).get("gated", 0),
            }
        )

    bulk_path = manifests / "encode-bulk.heavy.json"
    assay_counts, assay_bytes = Counter(), Counter()
    if bulk_path.exists():
        bulk = json.loads(bulk_path.read_text(encoding="utf-8"))
        for artifact in bulk.get("artifacts", []):
            note = str(artifact.get("note", ""))
            assay = note.removeprefix("ENCODE ").split("/", 1)[0] if note.startswith("ENCODE ") else "other"
            assay_counts[assay] += 1
            assay_bytes[assay] += int(artifact.get("bytes") or 0)

    bam_metadata_path = raw / "encode-bulk" / "bam_metadata.json"
    biosamples, targets = Counter(), Counter()
    if bam_metadata_path.exists():
        metadata = json.loads(bam_metadata_path.read_text(encoding="utf-8"))
        for record in metadata.values():
            biosamples[str(record.get("biosample", "unknown"))] += 1
            targets[str(record.get("target", "unknown"))] += 1

    normal_terms = {"pancreas", "endocrine pancreas"}
    normal_bam_count = sum(count for name, count in biosamples.items() if name in normal_terms)
    return {
        "schema": "pdac-circuit.chromatin-inventory/1",
        "root": str(root),
        "raw_bytes": _bytes_under(raw),
        "raw_gb": round(_bytes_under(raw) / 1024**3, 3),
        "manifest_artifacts": total_artifacts,
        "manifest_bytes": total_manifest_bytes,
        "corpora": sorted(corpora, key=lambda item: item["bytes"], reverse=True),
        "encode_bulk_assays": {
            assay: {"files": assay_counts[assay], "gb": round(assay_bytes[assay] / 1024**3, 3)}
            for assay in sorted(assay_counts)
        },
        "bam_biosamples": dict(biosamples),
        "bam_targets": dict(targets),
        "healthy_pancreas_bams": normal_bam_count,
        "pdac_tumor_bams": 0,
        "critical_gap": (
            "The large ENCODE corpus is healthy/endocrine pancreas. Treat it as a healthy "
            "counterfactual; no PDAC-specific superiority claim is admissible until independent "
            "tumor/organoid chromatin studies are compiled and held out by donor/study."
        ),
    }
