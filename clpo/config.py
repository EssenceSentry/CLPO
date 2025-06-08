from __future__ import annotations

import pathlib
from typing import Dict, List, Tuple, Union

import yaml
from loguru import logger
from pydantic import BaseModel, Field, validator

# ─────────────────────────── Prices Loader ───────────────────────────────
# Moved here as it's tightly coupled with CLPOConfig pricing defaults

_DEFAULT_PRICES: Dict[str, Dict[str, float]] = {
    "gpt-4o-mini": {"input": 0.150, "output": 0.600},
    "gemini-1.5-flash-latest": {"input": 0.00035, "output": 0.00105},
    "default": {"input": 0.003, "output": 0.003},
}

def load_prices(
    default_prices: Dict[str, Dict[str, float]],
    prices_path: pathlib.Path = pathlib.Path("prices.yaml"),
) -> Dict[str, Dict[str, float]]:
    """
    Load prices from a YAML file if it exists, falling back to defaults on error.

    Args:
        default_prices: The default price dictionary.
        prices_path: Path to the YAML file.

    Returns:
        A dictionary of prices.
    """
    try:
        loaded = yaml.safe_load(prices_path.read_text()) if prices_path.exists() else {}
        if not isinstance(loaded, dict):
            raise ValueError("prices.yaml did not contain a dict")
        tmp = default_prices.copy()
        for m, d_loaded in loaded.items(): # renamed d to d_loaded to avoid confusion
            if isinstance(d_loaded, dict) and {"input", "output"} <= set(d_loaded):
                tmp[m] = {"input": float(d_loaded["input"]), "output": float(d_loaded["output"])}
            else:
                logger.warning("Skipping invalid price entry for %s", m)
        return tmp
    except Exception as exc:  # catch YAML + IO errors
        logger.warning("Using default price table due to error: %s", exc)
        return default_prices

_PRICES = load_prices(_DEFAULT_PRICES)


# ─────────────────────────── Configuration Model ────────────────────────────────
EvaluatorConfigType = Union[str, Tuple[str, float]]


class CLPOConfig(BaseModel):
    """
    CLPOConfig is a configuration model for controlling the behavior of a prompt optimization system.
    ... (docstring from clpo.py)
    """

    # Credentials
    openai_api_key: str = Field("", env="OPENAI_API_KEY")
    google_api_key: str = Field("", env="GOOGLE_API_KEY")

    # Model selection
    task_gen_model: str = "gpt-4o-mini"
    rubric_gen_model: str = "gpt-4o-mini"
    initial_pop_gen_model: str = "gpt-4o-mini"
    prompt_tuner_model: str = "gpt-4o-mini"
    applicator_model: str = "gpt-4o-mini"
    embedding_model: str = "text-embedding-3-small"
    evaluators_config: List[EvaluatorConfigType] = Field(
        default_factory=lambda: [("gpt-4o-mini", 3), ("gemini-1.5-flash-latest", 1)]
    )

    # Pricing / temps
    prices: Dict[str, Dict[str, float]] = Field(default_factory=lambda: _PRICES)
    model_temperatures: Dict[str, float] = Field(default_factory=dict)

    # Prompt and system templates (defaults keep config valid)
    task_gen_template: str = "## Task Generator\nGenerate diverse tasks."
    rubric_gen_template: str = (
        "## Rubric Generator\nReturn a markdown table ending with **TOTAL** row."
    )
    init_pop_system_msg: str = "## Initial‑Pop System\nYou create alternative prompts."
    initial_pop_user_template: str = "Produce {n} prompt variants based on: {task}"
    prompt_tuner_template: str = (
        "## Prompt Tuner\nCombine the following prompts into a stronger one."
    )
    applicator_system_msg: str = "## Applicator\nGiven TASK and PROMPT, produce a plan."
    eval_system_msg: str = "## Evaluator\nScore the plan using the rubric."

    # Evolution hyper‑parameters
    mu: int = 3
    lam: int = 4
    sh_levels: List[Tuple[int, int]] = Field(
        default_factory=lambda: [(1, 1), (2, 2), (3, 3)]
    )
    budget_usd: float = 30.0
    generations: int = 10
    holdout_interval: int = 3
    holdout_seeds: int = 2
    spot_rate: float = 0.05
    use_embeddings_for_tasks: bool = False
    num_tasks_to_generate_raw: int = 40
    num_tasks_final: int = 8
    num_holdout_tasks: int = 2
    human_in_the_loop: bool = True

    # Validators
    @validator("evaluators_config")  # type: ignore[misc]
    def _v_eval(cls, v: List[EvaluatorConfigType]):
        if not v:
            raise ValueError("evaluators_config cannot be empty")
        return v

    @validator("prices")  # type: ignore[misc]
    def _v_price(cls, v_prices: Dict[str, Dict[str, float]]): # renamed v to v_prices
        for m, d_price in v_prices.items(): # renamed d to d_price
            if not {"input", "output"} <= set(d_price):
                raise ValueError(f"Price dict for {m} missing keys")
        return v_prices

    # helper
    def get_autogen_cost_map(self) -> Dict[str, Dict[str, float]]:
        return {
            m: {
                "input_cost_per_1k_tokens": d_cost["input"], #renamed d to d_cost
                "output_cost_per_1k_tokens": d_cost["output"],#renamed d to d_cost
            }
            for m, d_cost in self.prices.items() #renamed d to d_cost
        }


# ─────────────────────────── Credential Check Utilities ─────────────────────────────
ALL_MODEL_FIELDS = [
    "task_gen_model",
    "rubric_gen_model",
    "initial_pop_gen_model",
    "prompt_tuner_model",
    "applicator_model",
]


def _models_from_config(cfg: CLPOConfig) -> List[str]:
    """
    Extracts and returns a list of all model identifier strings referenced in the given configuration.
    ... (docstring from clpo.py)
    """
    models: List[str] = [getattr(cfg, field) for field in ALL_MODEL_FIELDS]
    for item in cfg.evaluators_config:
        models.append(item if isinstance(item, str) else item[0])
    return models


def check_api_keys(cfg: CLPOConfig) -> bool:
    """
    Checks if the required API keys are present in the given configuration.
    ... (docstring from clpo.py)
    """
    models = _models_from_config(cfg)
    need_openai = any(
        m.startswith("gpt") for m in models
    ) or cfg.embedding_model.startswith("text-")
    need_gemini = any(m.startswith("gemini") for m in models)

    missing: List[str] = []
    if need_openai and not cfg.openai_api_key:
        missing.append("OPENAI_API_KEY")
    if need_gemini and not cfg.google_api_key:
        missing.append("GOOGLE_API_KEY")

    if missing:
        logger.error("Missing credentials: %r", missing)
        return False
    return True
