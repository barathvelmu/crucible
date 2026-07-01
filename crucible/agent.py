"""ADK entry point.

`adk web` / `adk run` discover the `root_agent` exported here. Set
CRUCIBLE_OFFLINE=1 to run the scripted offline model (no API key); otherwise
the agents call Gemini and a GOOGLE_API_KEY (or Vertex config) is required.
"""
from __future__ import annotations

from .agents.pipeline import build_pipeline

root_agent = build_pipeline()
