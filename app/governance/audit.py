"""Structured audit logging.

Emits one JSON object per line on stdout. On Cloud Run, stdout JSON is parsed
by Cloud Logging automatically (``severity`` and ``message`` are promoted), so
these records can be routed to BigQuery with a log sink for compliance review.
Arguments are PII-redacted before they are written.
"""

from __future__ import annotations

import json
import sys
import time
from collections import deque
from typing import Any

from app.governance.pii import redact

# Ring buffer so the API (and tests) can show recent decisions without a DB.
RECENT: deque[dict[str, Any]] = deque(maxlen=200)


def _scrub(value: Any) -> Any:
    if isinstance(value, str):
        return redact(value).text
    if isinstance(value, dict):
        return {k: _scrub(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_scrub(v) for v in value]
    return value


def audit(event: str, severity: str = "INFO", **fields: Any) -> dict[str, Any]:
    record = {
        "severity": severity,
        "message": event,
        "ts": round(time.time(), 3),
        "component": "gemini-agent-hub",
        **_scrub(fields),
    }
    RECENT.append(record)
    sys.stdout.write(json.dumps(record, default=str) + "\n")
    sys.stdout.flush()
    return record
