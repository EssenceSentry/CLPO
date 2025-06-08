from __future__ import annotations
from typing import Dict, List, Tuple, Union
import pathlib
import re
import yaml
from pydantic import BaseModel, Field, validator
from loguru import logger

# ─────────────────────────── Prices Loader ───────────────────────────────
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
        for m, d in loaded.items():
            if isinstance(d, dict) and {"input", "output"} <= set(d):
                tmp[m] = {"input": float(d["input"]), "output": float(d["output"])}
            else:
                logger.warning("Skipping invalid price entry for %s", m)
        return tmp
    except Exception as exc:  # catch YAML + IO errors
        logger.warning("Using default price table due to error: %s", exc)
        return default_prices


_DEFAULT_PRICES: Dict[str, Dict[str, float]] = {
    "gpt-4o-mini": {"input": 0.150, "output": 0.600},
    "gemini-1.5-flash-latest": {"input": 0.00035, "output": 0.00105},
    "default": {"input": 0.003, "output": 0.003},
}

_PRICES = load_prices(_DEFAULT_PRICES)

# ─────────────────────────── Configuration ────────────────────────────────
EvaluatorConfigType = Union[str, Tuple[str, float]]


class CLPOConfig(BaseModel):
    """
    CLPOConfig is a configuration model for controlling the behavior of a prompt optimization system.

    Attributes:
        openai_api_key (str): API key for OpenAI services, loaded from the "OPENAI_API_KEY" environment variable.
        google_api_key (str): API key for Google services, loaded from the "GOOGLE_API_KEY" environment variable.

        task_gen_model (str): Model name used for task generation.
        rubric_gen_model (str): Model name used for rubric generation.
        initial_pop_gen_model (str): Model name used for initial population generation.
        prompt_tuner_model (str): Model name used for prompt tuning.
        applicator_model (str): Model name used for applying prompts to tasks.
        embedding_model (str): Model name used for generating embeddings.
        evaluators_config (List[EvaluatorConfigType]): List of tuples specifying evaluator models and their counts.

        prices (Dict[str, Dict[str, float]]): Nested dictionary mapping model names to their input/output token prices.
        model_temperatures (Dict[str, float]): Optional mapping of model names to temperature values.

        task_gen_template (str): Template for task generation prompts.
        rubric_gen_template (str): Template for rubric generation prompts.
        init_pop_system_msg (str): System message for initial population generation.
        initial_pop_user_template (str): User prompt template for initial population.
        prompt_tuner_template (str): Template for prompt tuning.
        applicator_system_msg (str): System message for the applicator.
        eval_system_msg (str): System message for the evaluator.

        mu (int): Number of parents in evolutionary algorithm.
        lam (int): Number of offspring in evolutionary algorithm.
        sh_levels (List[Tuple[int, int]]): Successive halving levels for evaluation.
        budget_usd (float): Total budget in USD for the optimization process.
        generations (int): Number of evolutionary generations.
        holdout_interval (int): Interval (in generations) for holdout evaluation.
        holdout_seeds (int): Number of seeds for holdout evaluation.
        spot_rate (float): Spot rate for cost estimation.
        use_embeddings_for_tasks (bool): Whether to use embeddings for task selection.
        num_tasks_to_generate_raw (int): Number of raw tasks to generate.
        num_tasks_final (int): Number of final tasks to select.
        num_holdout_tasks (int): Number of holdout tasks.
        human_in_the_loop (bool): Whether to include human feedback in the loop.

    Methods:
        get_autogen_cost_map() -> Dict[str, Dict[str, float]]:
            Returns a mapping of model names to their input/output costs per 1k tokens.

    Validators:
        _v_eval: Ensures that evaluators_config is not empty.
        _v_price: Ensures that each model's price dictionary contains both "input" and "output" keys.
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
    def _v_price(cls, v: Dict[str, Dict[str, float]]):
        for m, d in v.items():
            if not {"input", "output"} <= set(d):
                raise ValueError(f"Price dict for {m} missing keys")
        return v

    # helper
    def get_autogen_cost_map(self) -> Dict[str, Dict[str, float]]:
        return {
            m: {
                "input_cost_per_1k_tokens": d["input"],
                "output_cost_per_1k_tokens": d["output"],
            }
            for m, d in self.prices.items()
        }


# ─────────────────────────── Credential Check ─────────────────────────────
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

    Args:
        cfg (CLPOConfig): The configuration object containing model references.

    Returns:
        List[str]: A list of all model strings found in the configuration, including those specified
            in the fields listed in ALL_MODEL_FIELDS and those in the 'evaluators_config' attribute.
    """
    models: List[str] = [getattr(cfg, field) for field in ALL_MODEL_FIELDS]
    for item in cfg.evaluators_config:
        models.append(item if isinstance(item, str) else item[0])
    return models


def check_api_keys(cfg: CLPOConfig) -> bool:
    """
    Checks if the required API keys are present in the given configuration.

    This function determines which API keys are needed based on the models specified
    in the configuration. It checks for the presence of the OpenAI API key if any
    model starts with "gpt" or if the embedding model starts with "text-". It checks
    for the Google API key if any model starts with "gemini". If any required API
    keys are missing, it logs an error and returns False.

    Args:
        cfg (CLPOConfig): The configuration object containing model and API key information.

    Returns:
        bool: True if all required API keys are present, False otherwise.
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


# ─────────────────────────── Markdown-table parser ────────────────────────
_SCORE_TOTAL = re.compile(r"TOTAL\\s*[:\\-]?\\s*([0-9]+(?:\\.[0-9]+)?)", re.I)
_ROW = re.compile(r"\\|\\s*([^|]+?)\\s*\\|\\s*([0-9\\.]+)\\s*\\|")


def parse_markdown_scores(md: str) -> tuple[float, dict[str, float]]:
    """
    Parses a markdown table containing scores and returns the total score and a dictionary of per-row scores.

    This function attempts to robustly extract scores from a markdown-formatted table. It first tries to use pandas to parse the table, supporting both HTML and some markdown flavors. If pandas is unavailable or parsing fails, it falls back to using regular expressions.

    Args:
        md (str): The markdown string containing the table with scores.

    Returns:
        tuple[float, dict[str, float]]: A tuple containing:
            - The total score as a float.
            - A dictionary mapping each row's key (converted to lowercase and stripped) to its corresponding score as a float.

    Notes:
        - The function heuristically determines the total score from the last row's maximum numeric value if using pandas.
        - Rows with keys such as "max pts", "total", or "score" are excluded from the per-row dictionary.
        - If neither pandas nor regex finds a total, the total score defaults to 0.0.
    """
    # 1) try pandas (handles html tables, some md flavours via markdown-it)
    try:
        import pandas as pd

        tbls = pd.read_html(md, flavor="bs4")
        if tbls:
            df = tbls[0]
            # find numeric columns & a TOTAL row
            numeric = df.select_dtypes("number")
            if not numeric.empty:
                total = float(numeric.iloc[-1].max())  # heuristic
                subs = {
                    str(df.iloc[i, 0]).strip().lower(): float(numeric.iloc[i].max())
                    for i in range(len(df) - 1)  # skip TOTAL line
                }
                return total, subs
    except Exception:  # pandas missing or parse failure → regex fallback
        pass

    # 2) regex fallback
    total_match = _SCORE_TOTAL.search(md)
    total = float(total_match.group(1)) if total_match else 0.0
    subs: dict[str, float] = {}
    for k, v in _ROW.findall(md):
        key = k.strip().lower()
        if key in {"max pts", "total", "score"}:
            continue
        subs[key] = float(v)
    return total, subs


