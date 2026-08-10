from __future__ import annotations

import re

from .contract import OutputEnvelope,Verdict
from .provenance import GATED

_CLINICAL_DIRECTIVE_PATTERNS=[
    r"\badminister\b",
    r"\bprescribe(?:d|s)?\b",
    r"\bdosage\b",
    r"\bdose\s+the\s+patient\b",
    r"\btreat\s+the\s+patient\s+with\b",
    r"\byou\s+should\s+(?:take|inject|receive)\b",
    r"\brecommended\s+treatment\s+for\s+this\s+patient\b",
]
_DIRECTIVE_RE=re.compile("|".join(_CLINICAL_DIRECTIVE_PATTERNS),re.IGNORECASE)

def scan_for_clinical_directive(text: str) -> str | None:
    m=_DIRECTIVE_RE.search(text or "")
    return m.group(0) if m else None

def guard_emission(envelope: OutputEnvelope,*,rendered_text: str = "") -> OutputEnvelope:
    offending=scan_for_clinical_directive(rendered_text)
    if offending:
        return OutputEnvelope.refuse(
            f"hard-stop: output contains a clinical directive ({offending!r}); RUO forbids prescriptive emission"
        )
    if envelope.verdict == Verdict.OK and envelope.data_class == GATED:
        return OutputEnvelope.refuse(
            "hard-stop: result depends on a GATED input without credentials; refusing to emit a confident payload"
        )
    return envelope

def require_inputs_available(required: dict[str,str]) -> list[str]:
    return [feat for feat,dc in required.items() if dc == GATED]
