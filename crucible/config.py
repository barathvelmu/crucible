"""Central configuration for Crucible.

Model names, illustrative pricing, and runtime toggles live here so the rest
of the package never hard-codes them.
"""
from __future__ import annotations

import os
from pathlib import Path

PACKAGE_ROOT = Path(__file__).resolve().parent
CORPUS_DIR = PACKAGE_ROOT / "corpus"
PAPERS_PATH = CORPUS_DIR / "papers.json"
LAB_NOTES_PATH = CORPUS_DIR / "lab_notes.md"

# Gemini models per role. Override via env without touching code.
# Defaults are all gemini-2.5-flash so the pipeline runs on a free AI Studio
# tier out of the box (gemini-2.5-pro is not served on the free tier). On a paid
# tier, set CRUCIBLE_JUDGE_MODEL=gemini-2.5-pro for a stronger, separate judge.
RESEARCHER_MODEL = os.getenv("CRUCIBLE_RESEARCHER_MODEL", "gemini-2.5-flash")
REVISER_MODEL = os.getenv("CRUCIBLE_REVISER_MODEL", "gemini-2.5-flash")
JUDGE_MODEL = os.getenv("CRUCIBLE_JUDGE_MODEL", "gemini-2.5-flash")

# Illustrative list prices in USD per 1M tokens (input, output). These are for
# cost-per-request accounting in the eval report, edit to match current rates.
PRICING_USD_PER_1M = {
    "gemini-2.5-flash": {"input": 0.30, "output": 2.50},
    "gemini-2.5-pro": {"input": 1.25, "output": 10.00},
    # Fallback used by the offline scripted model so cost math still runs.
    "scripted-offline": {"input": 0.0, "output": 0.0},
}

# Run the whole pipeline with a deterministic local model and no API key.
OFFLINE = os.getenv("CRUCIBLE_OFFLINE", "0").lower() in ("1", "true", "yes")
