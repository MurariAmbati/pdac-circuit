from __future__ import annotations

from contextlib import contextmanager
from datetime import datetime
import json
import os
from pathlib import Path

@contextmanager
def exclusive_artifact_lock(path: str | Path):

    path=Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        descriptor=os.open(path, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
    except FileExistsError as exc:
        owner=path.read_text(encoding="utf-8", errors="replace")
        raise RuntimeError(f"artifact fetch is already locked at {path}: {owner}") from exc
    try:
        payload={
            "pid": os.getpid(),
            "created_at": datetime.now().astimezone().isoformat(),
        }
        os.write(descriptor, json.dumps(payload, sort_keys=True).encode("utf-8"))
        os.close(descriptor)
        descriptor=-1
        yield
    finally:
        if descriptor >= 0:
            os.close(descriptor)
        path.unlink(missing_ok=True)
