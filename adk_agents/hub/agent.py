"""Entry point for `adk web adk_agents` / `adk run adk_agents/hub` (ADK dev UI)."""

from app.agents.root import build_root_agent
from app.bootstrap import bootstrap

bootstrap()
root_agent = build_root_agent()
