"""PII redaction applied before any text reaches the model or the logs.

Pattern-based and deterministic on purpose: it is fast, auditable and runs
inline on every request. In production this sits in front of (or is replaced
by) Cloud DLP / Sensitive Data Protection for broader infoType coverage.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

_PATTERNS: list[tuple[str, re.Pattern[str]]] = [
    ("EMAIL", re.compile(r"\b[\w.+-]+@[\w-]+\.[\w.-]+\b")),
    ("CARD", re.compile(r"\b(?:\d[ -]?){13,19}\b")),
    ("PHONE", re.compile(r"(?<!\w)(?:\+?\d{1,3}[ -]?)?(?:\d[ -]?){9,10}\d(?!\w)")),
    ("IPV4", re.compile(r"\b(?:\d{1,3}\.){3}\d{1,3}\b")),
    ("API_KEY", re.compile(r"\b(?:AIza[0-9A-Za-z_-]{35}|sk-[A-Za-z0-9]{20,})\b")),
]


def _luhn_ok(digits: str) -> bool:
    total, parity = 0, len(digits) % 2
    for i, ch in enumerate(digits):
        d = int(ch)
        if i % 2 == parity:
            d *= 2
            if d > 9:
                d -= 9
        total += d
    return total % 10 == 0


@dataclass
class RedactionResult:
    text: str
    findings: dict[str, int] = field(default_factory=dict)

    @property
    def changed(self) -> bool:
        return bool(self.findings)


def redact(text: str) -> RedactionResult:
    findings: dict[str, int] = {}

    for label, pattern in _PATTERNS:
        def _sub(m: re.Match[str], label: str = label) -> str:
            raw = m.group(0)
            if label == "CARD":
                digits = re.sub(r"\D", "", raw)
                if not (13 <= len(digits) <= 19 and _luhn_ok(digits)):
                    return raw  # long number that isn't a card: leave for PHONE / keep
            findings[label] = findings.get(label, 0) + 1
            return f"[{label}]"

        text = pattern.sub(_sub, text)
    return RedactionResult(text, findings)
