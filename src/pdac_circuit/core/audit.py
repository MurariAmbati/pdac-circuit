from __future__ import annotations

from dataclasses import dataclass, field

from .provenance import now_iso
from .seeds import sha256_json

@dataclass
class AuditChain:

    steps: list[dict]=field(default_factory=list)

    @property
    def head(self) -> str | None:
        return self.steps[-1]["output_sha"] if self.steps else None

    def add(self, step: str, *, inputs: list[str], payload) -> str:
        record={"step": step, "inputs": list(inputs), "prev": self.head, "payload_sha": sha256_json(payload)}
        out_sha=sha256_json(record)
        self.steps.append(
            {"step": step, "inputs": list(inputs), "output_sha": out_sha, "ts": now_iso()}
        )
        return out_sha

    def to_dict(self) -> dict:
        return {"schema": "pdac-circuit.audit/1", "head": self.head, "steps": list(self.steps)}
