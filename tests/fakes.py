"""A deterministic stand-in for Gemini so agent wiring is tested offline."""

from __future__ import annotations

from collections.abc import AsyncGenerator, Callable

from google.adk.models.base_llm import BaseLlm
from google.adk.models.llm_request import LlmRequest
from google.adk.models.llm_response import LlmResponse
from google.genai import types

Policy = Callable[[LlmRequest], types.Part]


class ScriptedLlm(BaseLlm):
    """Calls ``policy(request)`` to decide the next Part (text or function call)."""

    policy: Policy
    calls: int = 0

    async def generate_content_async(
        self, llm_request: LlmRequest, stream: bool = False
    ) -> AsyncGenerator[LlmResponse, None]:
        self.calls += 1
        part = self.policy(llm_request)
        yield LlmResponse(content=types.Content(role="model", parts=[part]))


def _is_replayed(c: types.Content) -> bool:
    """ADK replays other agents' turns as a user message opening with "For context:"."""
    first = (c.parts or [None])[0]
    return bool(first and first.text and first.text.startswith("For context:"))


def last_user_text(req: LlmRequest) -> str:
    for c in reversed(req.contents):
        if c.role == "user" and not _is_replayed(c):
            for p in c.parts or []:
                if p.text:
                    return p.text
    return ""


def last_function_response(req: LlmRequest):
    last = req.contents[-1] if req.contents else None
    for p in (last.parts or []) if last else []:
        if p.function_response:
            return p.function_response
    return None
