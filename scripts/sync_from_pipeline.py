from __future__ import annotations

import json
import shutil
import subprocess
from datetime import datetime, timezone
from pathlib import Path

SRC = Path("C:/Users/murar/pdac-circuit")
ROOT = Path(__file__).resolve().parents[1]

TREES = [
    ("src", "src"),
    ("scripts", "pipeline"),
    ("results", "results"),
    ("tests", "tests"),
    ("docs", "docs"),
    ("models", "models"),
    ("configs", "configs"),
    ("data/manifests", "data/manifests"),
]
FILES = [
    "pyproject.toml",
    "METHODS.md",
    "RESULTS.md",
    "FINDINGS.md",
    "COMPENDIUM.md",
    "REVIEW_RESPONSE.md",
    "AUDIT_RESPONSE.md",
]


def tracked(rel):
    out = subprocess.run(["git", "-C", str(SRC), "ls-files", rel],
                         capture_output=True, text=True, check=True).stdout
    return [line.strip() for line in out.splitlines() if line.strip()]


def head_sha():
    r = subprocess.run(["git", "-C", str(SRC), "rev-parse", "HEAD"],
                       capture_output=True, text=True, check=True)
    return r.stdout.strip()


def head_subject():
    r = subprocess.run(["git", "-C", str(SRC), "log", "-1", "--format=%s"],
                       capture_output=True, text=True, check=True)
    return r.stdout.strip()


def main():
    sha = head_sha()
    copied = 0
    for rel_src, rel_dst in TREES:
        dst_root = ROOT / rel_dst
        if dst_root.exists():
            shutil.rmtree(dst_root)
        for f in tracked(rel_src):
            s = SRC / f
            if not s.exists():
                continue
            rest = Path(f).relative_to(rel_src)
            d = dst_root / rest
            d.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(s, d)
            copied += 1
        print(f"  {rel_src:18} -> {rel_dst:18} {len(tracked(rel_src)):4} files")
    for rel in FILES:
        s = SRC / rel
        if s.exists():
            shutil.copy2(s, ROOT / rel)
            copied += 1
    stamp = {
        "schema": "pdac-circuit.pipeline-sync/1",
        "source_repo": "https://github.com/MurariAmbati/pdac-chromatin-circuit",
        "source_commit": sha,
        "source_subject": head_subject(),
        "synced_at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "files": copied,
        "note": "Mirrored from the pipeline repository by scripts/sync_from_pipeline.py. "
                "The pipeline repository is authoritative; rerun the script to refresh. "
                "Raw data and trained weights are excluded, as in the source.",
    }
    blob = json.dumps(stamp, indent=2)
    (ROOT / "PIPELINE_SOURCE.json").write_text(blob, encoding="utf-8")
    (ROOT / "_data").mkdir(exist_ok=True)
    (ROOT / "_data" / "pipeline_source.json").write_text(blob, encoding="utf-8")
    print(f"  {copied} files from {sha[:12]} ({stamp['synced_at']})")


if __name__ == "__main__":
    main()
