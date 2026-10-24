"""Lightweight prompt-injection screening.

A weighted heuristic scorer over known jailbreak / exfiltration phrasings.
It is the cheap first layer of defence-in-depth (alongside tool allowlists,
least-privilege service accounts and Model Armor in production), and it
returns *why* it fired so decisions are explainable in the audit log.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

_RULES: list[tuple[str, float, re.Pattern[str]]] = [
    ("override_instructions", 0.6, re.compile(r"\b(ignore|disregard|forget)\b.{0,30}\b(previous|prior|above|all|system)\b.{0,20}\b(instructions?|rules?|prompts?)\b", re.I)),
    ("reveal_system_prompt", 0.6, re.compile(r"\b(reveal|print|show|repeat|leak)\b.{0,30}\b(system|hidden|initial)\b.{0,10}\b(prompt|instructions?)\b", re.I)),
    ("role_hijack", 0.4, re.compile(r"\b(you are now|act as|pretend to be)\b.{0,40}\b(dan|developer mode|unrestricted|jailbroken|no rules)\b", re.I)),
    ("exfiltrate_secrets", 0.5, re.compile(r"\b(api[_ ]?key|password|secret|credentials?|token)s?\b.{0,40}\b(send|post|email|exfiltrate|upload)\b", re.I)),
    ("tool_coercion", 0.3, re.compile(r"\b(call|invoke|run)\b.{0,20}\b(every|all)\b.{0,10}\btools?\b", re.I)),
    ("delimiter_spoof", 0.3, re.compile(r"(<\s*/?\s*system\s*>|\[\s*system\s*\]|###\s*system)", re.I)),
]


@dataclass(frozen=True)
class InjectionVerdict:
    score: float
    reasons: tuple[str, ...]
    threshold: float = 0.5

    @property
    def blocked(self) -> bool:
        return self.score >= self.threshold


def screen(text: str, threshold: float = 0.5) -> InjectionVerdict:
    reasons = tuple(name for name, _, rx in _RULES if rx.search(text))
    score = min(1.0, sum(w for name, w, _ in _RULES if name in reasons))
    return InjectionVerdict(round(score, 2), reasons, threshold)
