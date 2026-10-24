"""Guardrails wired into ADK agent callbacks.

* ``guard_model_input``  (before_model_callback): redacts PII from every user
  turn sent to Gemini and short-circuits the call if the latest user message
  looks like a prompt-injection attempt.
* ``enforce_tool_policy`` (before_tool_callback): role-based tool allowlist,
  read from session state (``user:role``), so a guest can search docs but
  cannot create tickets or calendar events.
* ``audit_tool_result``  (after_tool_callback): structured audit record for
  every tool call, with arguments redacted.
"""

from __future__ import annotations

from typing import Any

from google.genai import types

from app.config import get_settings
from app.governance.audit import audit
from app.governance.injection import screen
from app.governance.pii import redact

READ_TOOLS = {"search_knowledge_base", "list_calendar_events", "find_free_slots", "lookup_employee"}
WRITE_TOOLS = {"create_ticket", "create_calendar_event"}

ROLE_TOOLS: dict[str, set[str]] = {
    "guest": {"search_knowledge_base"},
    "employee": READ_TOOLS | {"create_ticket"},
    "admin": READ_TOOLS | WRITE_TOOLS,
}

# ADK's own agent-transfer tool must always be allowed or routing breaks.
ALWAYS_ALLOWED = {"transfer_to_agent"}

BLOCKED_REPLY = (
    "I can't help with that request. It looks like an attempt to override my "
    "instructions or extract protected information, so it has been logged."
)


def _latest_user_text(contents: list[types.Content]) -> str:
    for content in reversed(contents or []):
        if content.role == "user":
            parts = content.parts or []
            # ADK replays other agents' turns as a user message that opens with
            # "For context:"; only screen what the human actually typed.
            if parts and parts[0].text and parts[0].text.startswith("For context:"):
                continue
            texts = [p.text for p in parts if p.text]
            if texts:
                return "\n".join(texts)
            # A user turn holding only function responses is tool output, not
            # something the human typed; keep looking further back.
    return ""


def guard_model_input(callback_context: Any, llm_request: Any) -> Any:
    settings = get_settings()
    findings: dict[str, int] = {}
    if settings.redact_pii:
        for content in llm_request.contents or []:
            if content.role != "user":
                continue
            for part in content.parts or []:
                if part.text:
                    result = redact(part.text)
                    if result.changed:
                        part.text = result.text
                        for k, v in result.findings.items():
                            findings[k] = findings.get(k, 0) + v
    if findings:
        audit("pii_redacted", agent=callback_context.agent_name, findings=findings)

    if settings.block_prompt_injection:
        verdict = screen(_latest_user_text(llm_request.contents))
        if verdict.blocked:
            audit(
                "prompt_injection_blocked",
                severity="WARNING",
                agent=callback_context.agent_name,
                score=verdict.score,
                reasons=list(verdict.reasons),
            )
            from google.adk.models.llm_response import LlmResponse

            return LlmResponse(
                content=types.Content(role="model", parts=[types.Part(text=BLOCKED_REPLY)])
            )
    return None


def role_of(state: Any) -> str:
    role = state.get("user:role", "employee") if state is not None else "employee"
    return role if role in ROLE_TOOLS else "guest"


def enforce_tool_policy(tool: Any, args: dict[str, Any], tool_context: Any) -> dict | None:
    if tool.name in ALWAYS_ALLOWED:
        return None
    role = role_of(tool_context.state)
    if tool.name not in ROLE_TOOLS[role]:
        audit("tool_denied", severity="WARNING", tool=tool.name, role=role, args=args)
        return {
            "status": "denied",
            "error": f"Role '{role}' is not permitted to use '{tool.name}'. "
            "Tell the user this action needs elevated access.",
        }
    return None


def audit_tool_result(
    tool: Any, args: dict[str, Any], tool_context: Any, tool_response: Any
) -> dict | None:
    status = tool_response.get("status", "ok") if isinstance(tool_response, dict) else "ok"
    audit(
        "tool_call",
        tool=tool.name,
        role=role_of(tool_context.state),
        agent=tool_context.agent_name,
        args=args,
        status=status,
    )
    return None
