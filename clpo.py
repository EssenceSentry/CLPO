from __future__ import annotations

# ─────────────────────────── Imports ───────────────────────────────
import argparse
# import functools # Moved to utils
# import hashlib # Moved to utils
# import json # No longer directly needed in clpo.py
# import os # Moved to utils (for CACHE) and constants (for SEED)
import pathlib
import random
# import re # No longer directly needed in clpo.py
import sys
# import time # Moved to utils
# import uuid # No longer directly needed in clpo.py after PromptData move
# from abc import ABC, abstractmethod # Moved to adapters
from typing import ( # List and Dict are used by main()
    # Any, # No longer directly needed
    # Callable, # No longer directly needed
    Dict, # Retained for main
    List, # Retained for main
    # Optional, # No longer directly needed
    # ParamSpec, # Moved to utils
    # Pattern, # Moved to utils (for parse_markdown_scores)
    # Tuple, # Moved to config.py (used by CLPOConfig)
    # Type, # Moved to adapters (no longer directly needed in clpo.py)
    # TypeVar, # Moved to utils
    # Union, # Moved to config.py (used by CLPOConfig)
)

# import autogen # Moved to orchestrator.py
import numpy as np
# import openai # No longer directly needed in clpo.py
import trueskill
# import yaml # Moved to config.py
# from autogen import AssistantAgent, UserProxyAgent # Moved to agents.py and engines.py
# from diskcache import Cache # Moved to utils
from loguru import logger
# from openai.error import OpenAIError # Moved to engines
# from pydantic import BaseModel, Field, validator # Moved to config.py
from rich.logging import RichHandler
# from tqdm import tqdm # Moved to engines

# sklearn imports (KMeans, HAS_SKLEARN) moved to engines.py

# ─────────────────────────── Constants ───────────────────────────────
from clpo.constants import ( # Changed to absolute import
    CACHE_DIR_ENV,
    CACHE_EXPIRE_ENV,
    DEFAULT_CACHE_DIR,
    DEFAULT_CACHE_EXPIRE,
    # DEFAULT_RETRY_ATTEMPTS, # Moved to constants, used by utils
    # DEFAULT_RETRY_DELAY,    # Moved to constants, used by utils
    # RETRY_BACKOFF_FACTOR,   # Moved to constants, used by utils
    SEED,
)

# Import moved utilities
from clpo.utils import ( # Changed to absolute import
    _cached,
    _embs_hash,
    # _freeze is not directly called from clpo.py, only by _cached in utils
    parse_markdown_scores,
    retry_on_exception,
)

# ─────────────────────────── Global Setup ──────────────────────────────
random.seed(SEED)
np.random.seed(SEED)
trueskill.setup(draw_probability=0.01) # Removed random_state=SEED

# CACHE definition moved to utils.py
# Helper functions _embs_hash, _freeze, _cached moved to utils.py

# Adapter layer moved to clpo/adapters.py
from clpo.adapters import ( # Changed to absolute import
    ADAPTER_REGISTRY,
    BaseAdapter,
    GeminiAdapter,
    OpenAIAdapter,
    adapter_class_for,
)

# AgentFactory moved to clpo/agents.py
from clpo.agents import AgentFactory # Changed to absolute import

# Import for PromptData
from clpo.data_models import PromptData # Changed to absolute import

# Import for Engines
from clpo.engines import ( # Changed to absolute import
    EvaluationEngine,
    InitialPopulationGenerator,
    RubricEngine,
    TaskEngine,
    TuningEngine,
)

# Import for Orchestrator
from clpo.orchestrator import CLPO_Orchestrator # Changed to absolute import

# Imports for Config (moved from here to clpo/config.py)
# Relevant items will be imported back in main() or where needed.
from clpo.config import CLPOConfig, check_api_keys # Changed to absolute import

# ─────────────────────────── Main ───────────────────────────────────────
# Price loading, CLPOConfig definition, and check_api_keys and its helpers
# have been moved to clpo/config.py
def main() -> None:
    """
    Main entry point for the CLPO-B Prompt Evolution tool.

    Parses command-line arguments to configure the prompt evolution process, sets up logging,
    validates API keys, and initializes the orchestrator. Supports resuming from a previous
    state and allows overriding the halving schedule.

    Command-line arguments:
        --sh           : Override halving schedule (format: "1,1;2,2;3,3").
        --log-level    : Set logging level (default: "INFO").
        --resume       : Resume from the specified state file if it exists.
        --state-file   : Path to the JSONL state file (default: "clpo_state.jsonl").
    """
    parser = argparse.ArgumentParser(description="CLPO-B Prompt Evolution")
    parser.add_argument(
        "--sh", type=str, help="Override halving schedule, e.g. 1,1;2,2;3,3"
    )
    parser.add_argument("--log-level", type=str, default="INFO", help="Logging level")
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume from clpo_state.jsonl if it exists",
    )
    parser.add_argument(
        "--state-file",
        type=str,
        default="clpo_state.jsonl",
        help="Path to JSONL state file",
    )
    args = parser.parse_args()

    logger.remove()
    logger.add(RichHandler(), level=args.log_level.upper())

    cfg_kwargs: dict[str, object] = {"human_in_the_loop": False}
    if args.sh:
        try:
            cfg_kwargs["sh_levels"] = [
                tuple(map(int, x.split(","))) for x in args.sh.split(";")
            ]
        except ValueError as e:
            logger.error("Invalid --sh format: %s", e)
            sys.exit(1)

    BASE_PROMPT = (
        "You are a meticulous planner. Given any high-level objective, produce a numbered "
        "step-by-step plan (10-25 steps). Each step ≤ 25 words and must state resources, "
        "timeline, and potential risks."
    )

    cfg = CLPOConfig(**cfg_kwargs)
    if not check_api_keys(cfg):
        sys.exit(1)

    orch = CLPO_Orchestrator(
        cfg, BASE_PROMPT, state_file=pathlib.Path(args.state_file), resume=args.resume
    )
    with orch:
        if not (args.resume and orch._load_state()):
            # only prepare when we could not resume
            orch.prepare()
        orch.run()


if __name__ == "__main__":
    main()
