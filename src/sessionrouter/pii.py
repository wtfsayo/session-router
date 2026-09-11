"""PII detection and sanitization for cloud egress control.

Design (from the research synthesis):
- Three egress postures per request: STRICT (never leaves), REDACT (placeholder
  substitution, restore map stays local), OPEN.
- Detection is layered: fast regex/Luhn detectors ship in-tree; a pluggable
  detector protocol lets you bolt on Presidio/GLiNER/a local SLM for recall.
- Sanitization uses typed placeholders ([EMAIL_1], [PERSON_2]...) that are
  stable within a session so cloud context stays coherent across turns.
"""
from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass
from typing import Callable, Optional

from .types import Message, PrivacyTier


@dataclass
class PIIFinding:
    kind: str            # EMAIL | PHONE | SSN | CREDIT_CARD | API_KEY | IP | PERSON | ...
    span: tuple          # (start, end)
    text: str
    confidence: float = 1.0


# --- detectors -------------------------------------------------------------

_EMAIL = re.compile(r"\b[A-Za-z0-9._%+\-]+@[A-Za-z0-9.\-]+\.[A-Za-z]{2,}\b")
_PHONE = re.compile(
    r"(?<!\d)(?:\+?1[\s.\-]?)?(?:\(\d{3}\)|\d{3})[\s.\-]\d{3}[\s.\-]\d{4}(?!\d)"
)
_SSN = re.compile(r"\b\d{3}-\d{2}-\d{4}\b")
_IPV4 = re.compile(r"\b(?:\d{1,3}\.){3}\d{1,3}\b")
_API_KEY = re.compile(
    r"\b(?:sk|pk|key|token|bearer|api[_-]?key|AIza|ghp|gho|xox[baprs]|AKIA|sk-ant|sk-proj)"
    r"[A-Za-z0-9_\-]{16,}\b",
    re.IGNORECASE,
)
_CC = re.compile(r"\b(?:\d[ \-]?){13,19}\b")
_IBAN = re.compile(r"\b[A-Z]{2}\d{2}[A-Z0-9]{11,30}\b")
_PASSPORT = re.compile(r"\b[A-Z]{1,2}\d{6,9}\b")  # weak, high FP — kept at low conf


def _luhn_ok(digits: str) -> bool:
    s, alt = 0, False
    for d in reversed(digits):
        n = int(d)
        if alt:
            n *= 2
            if n > 9:
                n -= 9
        s += n
        alt = not alt
    return s % 10 == 0


def regex_detect(text: str) -> list[PIIFinding]:
    """Fast in-tree detector. Recall-oriented: better a false positive that
    forces local than a leak to cloud."""
    out: list[PIIFinding] = []
    for kind, rx in (
        ("EMAIL", _EMAIL), ("PHONE", _PHONE), ("SSN", _SSN),
        ("API_KEY", _API_KEY), ("IBAN", _IBAN),
    ):
        for m in rx.finditer(text):
            out.append(PIIFinding(kind, m.span(), m.group()))
    for m in _IPV4.finditer(text):
        parts = m.group().split(".")
        if all(0 <= int(p) <= 255 for p in parts):
            out.append(PIIFinding("IP", m.span(), m.group(), 0.8))
    for m in _CC.finditer(text):
        digits = re.sub(r"\D", "", m.group())
        if 13 <= len(digits) <= 19 and _luhn_ok(digits):
            out.append(PIIFinding("CREDIT_CARD", m.span(), m.group()))
    for m in _PASSPORT.finditer(text):
        out.append(PIIFinding("PASSPORT", m.span(), m.group(), 0.5))
    return out


# Pluggable detector: (text) -> [PIIFinding]. E.g. Presidio analyzer, GLiNER,
# or a local SLM. Registered detectors are OR'd with the regex layer.
_extra_detectors: list[Callable[[str], list[PIIFinding]]] = []


def register_detector(fn: Callable[[str], list[PIIFinding]]) -> None:
    _extra_detectors.append(fn)


def detect(text: str) -> list[PIIFinding]:
    findings = regex_detect(text)
    for fn in _extra_detectors:
        findings.extend(fn(text))
    # dedupe overlapping spans: keep longest/highest-confidence per overlap
    findings.sort(key=lambda f: (f.span[0], -(f.span[1] - f.span[0])))
    deduped: list[PIIFinding] = []
    for f in findings:
        if deduped and f.span[0] < deduped[-1].span[1]:
            continue
        deduped.append(f)
    return deduped


# --- sanitizer -------------------------------------------------------------

class Sanitizer:
    """Substitutes PII with session-stable typed placeholders.

    The restore map (placeholder -> original) is session state and MUST NOT
    be included in egress payloads. de_sanitize() restores a cloud response
    before returning it to the user.
    """

    def __init__(self) -> None:
        self._maps: dict[str, dict[str, str]] = {}   # session_id -> restore map

    def _placeholder(self, kind: str, value: str, rmap: dict[str, str]) -> str:
        for ph, orig in rmap.items():
            if orig == value:
                return ph
        n = sum(1 for ph in rmap if ph.startswith(f"[{kind}_")) + 1
        ph = f"[{kind}_{n}]"
        rmap[ph] = value
        return ph

    def sanitize(self, session_id: str, text: str) -> tuple[str, list[PIIFinding]]:
        rmap = self._maps.setdefault(session_id, {})
        findings = detect(text)
        out, last = [], 0
        for f in findings:
            out.append(text[last:f.span[0]])
            out.append(self._placeholder(f.kind, f.text, rmap))
            last = f.span[1]
        out.append(text[last:])
        return "".join(out), findings

    def de_sanitize(self, session_id: str, text: str) -> str:
        rmap = self._maps.get(session_id, {})
        if not rmap:
            return text
        for ph, orig in sorted(rmap.items(), key=lambda kv: -len(kv[0])):
            text = text.replace(ph, orig)
        return text

    def restore_map(self, session_id: str) -> dict[str, str]:
        return self._maps.get(session_id, {})

    def drop_session(self, session_id: str) -> None:
        self._maps.pop(session_id, None)


def classify_tier(
    findings: list[PIIFinding],
    strict_kinds: Optional[set[str]] = None,
) -> PrivacyTier:
    """STRICT if any finding is in strict_kinds (default: credentials/financial
    identifiers), REDACT if anything found, else OPEN."""
    if not findings:
        return PrivacyTier.OPEN
    strict = strict_kinds or {"SSN", "CREDIT_CARD", "API_KEY", "IBAN", "PASSPORT"}
    if any(f.kind in strict for f in findings):
        return PrivacyTier.STRICT
    return PrivacyTier.REDACT


def sanitize_messages(
    sanitizer: Sanitizer, session_id: str, messages: list[Message]
) -> list[Message]:
    out = []
    for m in messages:
        text, _ = sanitizer.sanitize(session_id, m.content)
        out.append(Message(role=m.role, content=text, name=m.name,
                           tool_call_id=m.tool_call_id))
    return out
