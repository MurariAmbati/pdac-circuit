from __future__ import annotations

import subprocess

def gpu_free_mib() -> int | None:
    try:
        out=subprocess.run(
            ["nvidia-smi", "--query-gpu=memory.free", "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=15,
        )
        if out.returncode != 0:
            return None
        return int(out.stdout.strip().splitlines()[0])
    except Exception:
        return None

def autosize_batch(default: int = 64, *, target_gib: float = 5.9, max_bs: int = 256) -> int:
    free=gpu_free_mib()
    if free is None:
        return default
    scale=(free / 1024.0) / max(target_gib, 0.1)
    bs=int(max(8, min(max_bs, round(default * scale))))
    return bs
