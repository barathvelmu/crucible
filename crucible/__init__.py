"""Crucible, a multi-agent evaluation forge on Google ADK."""
from __future__ import annotations

from . import agent  # noqa: F401  (exposes root_agent for adk web/run discovery)

__version__ = "0.1.0"
__all__ = ["agent"]
