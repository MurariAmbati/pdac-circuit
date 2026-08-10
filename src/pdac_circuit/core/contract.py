from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any

from .provenance import REAL, reduce_data_class

RUO_BANNER="RESEARCH USE ONLY — not for clinical or diagnostic use"

class Verdict(str, Enum):
    OK="OK"
    ABSTAIN="ABSTAIN"
    REFUSE="REFUSE"

@dataclass
class OutputEnvelope:

    verdict: Verdict
    data_class: str=REAL
    cert: str="real"
    payload: dict | None=None
    caveats: list[str]=field(default_factory=list)
    audit: dict | None=None
    ruo_banner: str=RUO_BANNER

    def to_dict(self) -> dict[str, Any]:
        return {
            "verdict": self.verdict.value,
            "data_class": self.data_class,
            "cert": self.cert,
            "payload": self.payload if self.verdict == Verdict.OK else None,
            "caveats": list(self.caveats),
            "audit": self.audit,
            "ruo_banner": self.ruo_banner,
        }

    @classmethod
    def ok(cls, payload: dict, *, data_classes=(), cert: str = "real", caveats=(), audit=None):
        return cls(
            verdict=Verdict.OK,
            data_class=reduce_data_class(list(data_classes)),
            cert=cert,
            payload=payload,
            caveats=list(caveats),
            audit=audit,
        )

    @classmethod
    def abstain(cls, reason: str, *, cert: str = "abstain", data_classes=(), audit=None):
        return cls(
            verdict=Verdict.ABSTAIN,
            data_class=reduce_data_class(list(data_classes)),
            cert=cert,
            payload=None,
            caveats=[reason],
            audit=audit,
        )

    @classmethod
    def refuse(cls, reason: str, *, audit=None):
        return cls(verdict=Verdict.REFUSE, cert="abstain", payload=None, caveats=[reason], audit=audit)
