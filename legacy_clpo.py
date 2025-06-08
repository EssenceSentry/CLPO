from __future__ import annotations

# ─────────────────────────── Imports ───────────────────────────────
import argparse
import functools
import hashlib
import json
import os
import pathlib
import random
import re
import sys
import time
import uuid
from abc import ABC, abstractmethod
from typing import (
    Any,
    Callable,
    Dict,
    List,
    Optional,
    ParamSpec,
    Pattern,
    Tuple,
    Type,
    TypeVar,
    Union,
)

import autogen
import numpy as np
import openai
import trueskill
import yaml
from autogen import AssistantAgent, UserProxyAgent
from diskcache import Cache
from loguru import logger
from openai.error import OpenAIError
from pydantic import BaseModel, Field, validator
from rich.logging import RichHandler
from tqdm import tqdm

try:
    from sklearn.cluster import KMeans

    HAS_SKLEARN = True
except ImportError:  # pragma: no cover
    KMeans = None  # type: ignore
    HAS_SKLEARN = False

# ─────────────────────────── Constants ───────────────────────────────
DEFAULT_RETRY_ATTEMPTS: int = 3
DEFAULT_RETRY_DELAY: float = 1.0
RETRY_BACKOFF_FACTOR: float = 2.0
CACHE_DIR_ENV: str = "CLPO_CACHE_DIR"
CACHE_EXPIRE_ENV: str = "CLPO_CACHE_EXPIRE"
DEFAULT_CACHE_DIR: str = ".clpo_cache"
DEFAULT_CACHE_EXPIRE: int = 86_400

# ─────────────────────────── Retry Decorator ──────────────────────────
P = ParamSpec("P")
R = TypeVar("R")


def retry_on_exception(
    max_attempts: int = DEFAULT_RETRY_ATTEMPTS,
    initial_delay: float = DEFAULT_RETRY_DELAY,
    backoff_factor: float = RETRY_BACKOFF_FACTOR,
    exceptions: Tuple[Type[Exception], ...] = (Exception,),
) -> Callable[[Callable[P, R]], Callable[P, R]]:
    """
    A decorator that retries a function call with exponential back-off upon specified exceptions.

    Args:
        max_attempts (int): Maximum number of retry attempts before giving up. Defaults to DEFAULT_RETRY_ATTEMPTS.
        initial_delay (float): Initial delay (in seconds) before the first retry. Defaults to DEFAULT_RETRY_DELAY.
        backoff_factor (float): Factor by which the delay increases after each failed attempt. Defaults to RETRY_BACKOFF_FACTOR.
        exceptions (Tuple[Type[Exception], ...]): Tuple of exception types that trigger a retry. Defaults to (Exception,).

    Returns:
        Callable[[Callable[P, R]], Callable[P, R]]: A decorator that applies retry logic to the target function.

    Raises:
        The last exception encountered if all retry attempts fail.
        RuntimeError: If the function does not return a value after all retries.

    Example:
        @retry_on_exception(max_attempts=3, initial_delay=1.0, backoff_factor=2.0)
        def unreliable_function():
            # function implementation
    """

    def decorator(func: Callable[P, R]) -> Callable[P, R]:
        @functools.wraps(func)
        def wrapper(*args: P.args, **kwargs: P.kwargs) -> R:  # type: ignore[misc]
            delay = initial_delay
            for attempt in range(1, max_attempts + 1):
                try:
                    return func(*args, **kwargs)
                except exceptions as e:
                    if attempt >= max_attempts:
                        logger.bind(func=func.__name__, attempt=attempt).error(
                            "failed: {}", e
                        )
                        raise
                    logger.bind(func=func.__name__, attempt=attempt).warning(
                        "failed: {}; retrying in {:.1f}s", e, delay
                    )
                    time.sleep(delay)
                    delay *= backoff_factor
            raise RuntimeError("Function did not return a value after retries.")

        return wrapper

    return decorator


# ─────────────────────────── Global Setup ──────────────────────────────
SEED: int = int(os.getenv("CLPO_SEED", "42"))
random.seed(SEED)
np.random.seed(SEED)
trueskill.setup(draw_probability=0.01, random_state=SEED)

CACHE = Cache(
    os.getenv(CACHE_DIR_ENV, DEFAULT_CACHE_DIR),
    expire=int(os.getenv(CACHE_EXPIRE_ENV, str(DEFAULT_CACHE_EXPIRE))),
)


# ─────────────────────────── Helpers ────────────────────────────────


def _embs_hash(embs: List[List[float]]) -> str:
    """
    Generates a SHA-1 hash for a list of embedding vectors.

    Each float in the embedding vectors is rounded to 6 decimal places before serialization.
    The list of vectors is serialized to a JSON string with sorted keys and compact separators,
    then hashed using SHA-1.

    Args:
        embs (List[List[float]]): A list of embedding vectors, where each vector is a list of floats.

    Returns:
        str: A SHA-1 hash representing the rounded and serialized embeddings.
    """
    rounded = [[round(x, 6) for x in vec] for vec in embs]
    serial = json.dumps(rounded, sort_keys=True, separators=(",", ":"))
    return hashlib.sha1(serial.encode("utf‑8")).hexdigest()


T = TypeVar("T")


def _freeze(obj: Any) -> Any:
    """
    Recursively convert a given object into a canonical, hashable representation.

    This function is useful for creating hashable versions of complex, nested data structures
    such as dictionaries, lists, and tuples. It processes the input as follows:
    - Dictionaries are converted to tuples of sorted (key, value) pairs, with values recursively frozen.
    - Lists and tuples are converted to tuples of recursively frozen elements.
    - Immutable primitive types (str, int, float, bool, None) are returned as-is.
    - For unsupported types, a tuple containing the type name and the object's repr is returned to reduce collision risk.

    Args:
        obj (Any): The object to be frozen.

    Returns:
        Any: A hashable, canonical representation of the input object.
    """
    if isinstance(obj, dict):
        return tuple((k, _freeze(obj[k])) for k in sorted(obj))
    if isinstance(obj, (list, tuple)):
        return tuple(_freeze(v) for v in obj)
    if isinstance(obj, (str, int, float, bool, type(None))):
        return obj
    # Fallback: include type name to reduce collision risk
    return (type(obj).__name__, repr(obj))


def _cached(
    key: Dict[str, Any], fn: Callable[[], T], cache_backend: Cache = CACHE
) -> T:
    """
    Caches the result of a function call using a structural hash of the provided key.

    Args:
        key (Dict[str, Any]): A dictionary representing the cache key. The key is structurally hashed to generate a unique cache identifier.
        fn (Callable[[], T]): A zero-argument function whose result should be cached.
        cache_backend (Cache, optional): The cache backend to use for storing results. Defaults to CACHE.

    Returns:
        T: The result of the function call, either retrieved from the cache or freshly computed.

    Notes:
        - The cache key is frozen and hashed using SHA-1 to ensure uniqueness based on structure.
        - If the result is not in the cache, the function is called and its result is cached.
    """
    frozen_key = _freeze(key)
    digest = hashlib.sha1(repr(frozen_key).encode("utf-8")).hexdigest()
    if digest in cache_backend:
        return cache_backend[digest]  # type: ignore[return-value]
    val = fn()
    cache_backend[digest] = val
    return val


# ─────────────────────────── Adapter Layer ──────────────────────────────
class BaseAdapter(ABC):
    """
    BaseAdapter is an abstract base class for language model adapters.

    Args:
        model (str): The name or identifier of the model to be used.
        cfg (Dict[str, Union[str, float]]): Configuration dictionary containing model-specific parameters.

    Attributes:
        model (str): Stores the model name or identifier.
        cfg (Dict[str, Union[str, float]]): Stores the configuration parameters for the model.

    Methods:
        llm_config() -> Dict[str, Union[str, float]]:
            Abstract method that should return the configuration dictionary for the language model.
    """

    def __init__(self, model: str, cfg: Dict[str, Union[str, float]]):
        self.model = model
        self.cfg = cfg

    @abstractmethod
    def llm_config(self) -> Dict[str, Union[str, float]]: ...


class OpenAIAdapter(BaseAdapter):
    """
    Adapter class for interfacing with OpenAI language models.

    Methods:
        llm_config() -> Dict[str, Union[str, float]]:
            Returns a dictionary containing configuration parameters for the OpenAI LLM,
            including the model name, temperature, and API key.
    """

    def llm_config(self) -> Dict[str, Union[str, float]]:
        return {
            "model": self.model,
            "temperature": float(self.cfg.get("temperature", 0.7)),
            "api_key": self.cfg["openai_api_key"],
        }


class GeminiAdapter(BaseAdapter):
    """
    Adapter class for integrating with the Gemini LLM API.

    Methods
    -------
    llm_config() -> Dict[str, Union[str, float]]:
        Returns a dictionary containing the configuration parameters for the Gemini LLM,
        including the model name, temperature, and Google API key.

    Attributes
    ----------
    model : str
        The name or identifier of the Gemini model to use.
    cfg : dict
        Configuration dictionary containing parameters such as 'temperature' and 'google_api_key'.
    """

    def llm_config(self) -> Dict[str, Union[str, float]]:
        return {
            "model": self.model,
            "temperature": float(self.cfg.get("temperature", 0.7)),
            "google_api_key": self.cfg.get("google_api_key", ""),
        }


ADAPTER_REGISTRY: Dict[str, Type[BaseAdapter]] = {
    "gpt": OpenAIAdapter,
    "gemini": GeminiAdapter,
}


def adapter_class_for(model: str) -> Type[BaseAdapter]:
    """
    Returns the adapter class corresponding to the given model name prefix.

    Args:
        model (str): The name of the model for which to find the adapter class.

    Returns:
        Type[BaseAdapter]: The adapter class associated with the model's prefix.

    Raises:
        ValueError: If no adapter class is found for the given model prefix.
    """
    for prefix, cls in ADAPTER_REGISTRY.items():
        if model.startswith(prefix):
            return cls
    raise ValueError(f"No adapter found for model prefix: {model}")


# ─────────────────────────── AgentFactory ────────────────────────────────
class AgentFactory:
    """
    AgentFactory is responsible for creating and managing various agent instances used in the CLPO system.

    Args:
        cfg (CLPOConfig): Configuration object containing model names, API keys, templates, and other settings.

    Attributes:
        cfg (CLPOConfig): The configuration object.
        _proxy_cache (Dict[str, UserProxyAgent]): Cache for user proxy agents to avoid redundant instantiation.

    Methods:
        adapter_config(model, cfg):
            Static method to generate the LLM adapter configuration for a given model using the provided config.

        _make(model, role, system):
            Internal helper to instantiate an AssistantAgent with the specified model, role, and system message.

        get_proxy(name):
            Retrieves a cached UserProxyAgent by name, or creates one if it does not exist.

        create_task_generator():
            Creates an AssistantAgent for task generation.

        create_rubric_generator():
            Creates an AssistantAgent for rubric generation.

        create_initial_pop_generator():
            Creates an AssistantAgent for initial population generation.

        create_prompt_tuner():
            Creates an AssistantAgent for prompt tuning.

        create_applicator():
            Creates an AssistantAgent for application tasks.

        create_evaluators(names):
            Creates a list of AssistantAgents for evaluation, one for each model name provided.

        create_human_auditor():
            Creates a UserProxyAgent configured for human auditing.
    """

    def __init__(self, cfg: "CLPOConfig"):
        self.cfg = cfg
        self._proxy_cache: Dict[str, UserProxyAgent] = {}

    @staticmethod
    def adapter_config(
        model: str,
        cfg: "CLPOConfig",
    ) -> Dict[str, Union[str, float]]:
        adapter_cls = adapter_class_for(model)
        return adapter_cls(
            model,
            {
                "openai_api_key": cfg.openai_api_key,
                "google_api_key": cfg.google_api_key,
                "temperature": cfg.model_temperatures.get(model, 0.7),
            },
        ).llm_config()

    def _make(
        self,
        model: str,
        role: str,
        system: str,
    ) -> AssistantAgent:
        llm_cfg = self.adapter_config(model, self.cfg)
        return AssistantAgent(
            name=f"{role}_{uuid.uuid4().hex[:5]}",
            system_message=system,
            llm_config=llm_cfg,
        )

    def get_proxy(self, name: str) -> UserProxyAgent:
        if name not in self._proxy_cache:
            self._proxy_cache[name] = UserProxyAgent(
                name=name,
                human_input_mode="NEVER",
                code_execution_config=False,
                default_auto_reply="",
            )
        return self._proxy_cache[name]

    # Factory helpers (unchanged apart from _make signature) ...

    def create_task_generator(self) -> AssistantAgent:
        return self._make(
            self.cfg.task_gen_model, "TaskGen", self.cfg.task_gen_template
        )

    def create_rubric_generator(self) -> AssistantAgent:
        return self._make(
            self.cfg.rubric_gen_model, "RubricGen", self.cfg.rubric_gen_template
        )

    def create_initial_pop_generator(self) -> AssistantAgent:
        return self._make(
            self.cfg.initial_pop_gen_model, "InitPopGen", self.cfg.init_pop_system_msg
        )

    def create_prompt_tuner(self) -> AssistantAgent:
        return self._make(
            self.cfg.prompt_tuner_model, "PromptTuner", self.cfg.prompt_tuner_template
        )

    def create_applicator(self) -> AssistantAgent:
        return self._make(
            self.cfg.applicator_model, "Applicator", self.cfg.applicator_system_msg
        )

    def create_evaluators(self, names: List[str]) -> List[AssistantAgent]:
        return [
            self._make(m, f"Eval{i}", self.cfg.eval_system_msg)
            for i, m in enumerate(names)
        ]

    def create_human_auditor(self) -> UserProxyAgent:
        return UserProxyAgent(
            name="human_auditor", human_input_mode="ALWAYS", code_execution_config=False
        )


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


# ─────────────────────────── TaskEngine ──────────────────────────────────
class TaskEngine:
    """
    TaskEngine is responsible for generating and splitting tasks into training and holdout sets.

    This class interacts with an agent factory to create task generators and human auditors, and supports both simple and embedding-based task generation. It can optionally use human-in-the-loop QA for task selection and supports robust error handling and fallback mechanisms.

    Attributes:
        cfg (CLPOConfig): Configuration object containing task generation and API settings.
        factory (AgentFactory): Factory for creating agents and proxies.
        task_gen: Task generator agent.
        human: Human auditor agent.
        train_tasks (List[str]): List of training task descriptions.
        holdout_tasks (List[str]): List of holdout task descriptions.

    Methods:
        _gen_simple():
            Generates a list of task descriptions using a simple prompt, with fallback to markdown parsing if JSON parsing fails.

        _embed_tasks(tasks: List[str]) -> List[List[float]]:
            Obtains embeddings for a list of task descriptions using the OpenAI API.

        _gen_with_embeddings():
            Generates a larger set of tasks, embeds them, clusters them using KMeans, and selects representative tasks from each cluster. Falls back to simple generation on failure.

        _qa(tasks: List[str]) -> List[str]:
            Optionally performs human-in-the-loop QA to drop tasks based on user input.

        generate_and_split():
            Generates tasks (using embeddings if configured), optionally performs QA, and splits the tasks into training and holdout sets.
    """

    def __init__(self, cfg: CLPOConfig, factory: AgentFactory) -> None:
        self.cfg = cfg
        self.factory = factory
        self.task_gen = factory.create_task_generator()
        self.human = factory.create_human_auditor()
        self.train_tasks: List[str] = []
        self.holdout_tasks: List[str] = []

    @retry_on_exception(exceptions=(OpenAIError,))
    def _gen_simple(self) -> List[str]:
        """
        Generates a list of task descriptions by interacting with a proxy chat initiator.

        The method sends a prompt to the proxy to generate a specified number of task descriptions,
        expecting a JSON array of strings as the response. If the response cannot be parsed as valid
        JSON, it falls back to extracting numbered lines from the raw response as task descriptions.

        Returns:
            List[str]: A list of task description strings, limited to the configured number.

        Raises:
            ValueError: If the JSON response is not a list of strings.
        """
        proxy = self.factory.get_proxy("task_gen_initiator")
        prompt = (
            f"Please output exactly {self.cfg.num_tasks_final} task descriptions "
            "as a JSON array of strings, with no additional text. "
            "Example:\n"
            '["Task one", "Task two", ...]'
        )
        raw = (
            proxy.initiate_chat(
                self.task_gen,
                message=prompt,
                max_turns=1,
                summary_method="last_msg",
            ).summary
            or ""
        )
        try:
            tasks = json.loads(raw)
            if not (isinstance(tasks, list) and all(isinstance(t, str) for t in tasks)):
                raise ValueError("Invalid JSON format for tasks")
            return tasks[: self.cfg.num_tasks_final]
        except Exception as e:
            logger.bind(error=str(e)).warning(
                "JSON parse failed; falling back to markdown parsing"
            )
            lines = [
                l.strip("•*-. ")
                for l in raw.splitlines()
                if l.strip() and l[0].isdigit()
            ]
            return lines[: self.cfg.num_tasks_final]

    @retry_on_exception(exceptions=(OpenAIError,))
    def _embed_tasks(self, tasks: List[str]) -> List[List[float]]:
        client = openai.OpenAI(api_key=self.cfg.openai_api_key)
        resp = client.embeddings.create(input=tasks, model=self.cfg.embedding_model)
        return [d.embedding for d in resp.data]

    @retry_on_exception(exceptions=(OpenAIError,))
    def _gen_with_embeddings(self) -> List[str]:
        """
        Generates a list of brief task descriptions using embeddings and clustering.

        This method attempts to generate a specified number of unique and diverse task descriptions
        by leveraging a language model and clustering their embeddings. The process is as follows:
        1. Requests a language model to generate a raw list of task descriptions in JSON format.
        2. Validates and parses the generated tasks.
        3. If the number of valid tasks is insufficient, falls back to a simpler generation method.
        4. Computes embeddings for the tasks and clusters them using KMeans to ensure diversity.
        5. Selects the most central task from each cluster to form the final set.
        6. If embedding or clustering fails, falls back to the simple generation method.

        Returns:
            List[str]: A list of brief, diverse task descriptions.

        Falls back to a simpler generation method if:
            - scikit-learn is not available,
            - the language model output is invalid,
            - there are not enough tasks,
            - or embedding/clustering fails.
        """
        if not HAS_SKLEARN:
            return self._gen_simple()
        proxy = self.factory.get_proxy("task_gen_initiator_emb")
        prompt = (
            f"Please output exactly {self.cfg.num_tasks_to_generate_raw} brief task descriptions "
            "(≤25 words each) as a JSON array of strings, with no additional text. Example:\n"
            '["Task one", "Task two", ...]'
        )
        raw = (
            proxy.initiate_chat(
                self.task_gen,
                message=prompt,
                max_turns=1,
                summary_method="last_msg",
            ).summary
            or ""
        )
        try:
            all_tasks = json.loads(raw)
            if not (
                isinstance(all_tasks, list)
                and all(isinstance(t, str) for t in all_tasks)
            ):
                raise ValueError("Invalid JSON format for tasks")
            all_tasks = all_tasks[: self.cfg.num_tasks_to_generate_raw]
        except Exception as e:
            logger.bind(error=str(e)).warning(
                "JSON parse failed; falling back to simple generation"
            )
            return self._gen_simple()

        if len(all_tasks) < self.cfg.num_tasks_final:
            return self._gen_simple()

        try:
            embs = _cached(
                {"tasks": all_tasks, "model": self.cfg.embedding_model},
                lambda: self._embed_tasks(all_tasks),
            )

            def fit_kmeans_on_embs() -> "KMeans":  # type: ignore
                if not HAS_SKLEARN or KMeans is None:
                    raise RuntimeError(
                        "scikit-learn is not installed; cannot cluster tasks."
                    )
                return KMeans(
                    n_clusters=self.cfg.num_tasks_final, random_state=SEED, n_init=10
                ).fit(embs)

            km = _cached(
                {
                    "emb_sha": _embs_hash(embs),
                    "k": self.cfg.num_tasks_final,
                },
                fit_kmeans_on_embs,
            )
        except (OpenAIError, ValueError) as e:
            logger.bind(error=str(e)).warning(
                "Embedding or clustering failed; falling back"
            )
            return self._gen_simple()

        # Choose the most central task in each cluster
        clusters: Dict[int, List[int]] = {
            i: [] for i in range(self.cfg.num_tasks_final)
        }
        for idx, lbl in enumerate(km.labels_):
            clusters[lbl].append(idx)

        chosen: List[int] = []
        for lbl, indices in clusters.items():
            if not indices:
                continue
            centroid = km.cluster_centers_[lbl]
            best_idx = min(
                indices, key=lambda i: np.linalg.norm(np.array(embs[i]) - centroid)
            )
            chosen.append(best_idx)

        remaining = [i for i in range(len(all_tasks)) if i not in chosen]
        random.shuffle(remaining)
        while len(chosen) < self.cfg.num_tasks_final and remaining:
            chosen.append(remaining.pop())

        return [all_tasks[i] for i in chosen[: self.cfg.num_tasks_final]]

    def _qa(self, tasks: List[str]) -> List[str]:
        if not (self.cfg.human_in_the_loop and sys.stdin.isatty()):
            return tasks
        disp = "\n".join(f"{i}: {t}" for i, t in enumerate(tasks))
        ans = self.human.get_human_input(
            f"Reply indices to drop (0-indexed) or 'ok':\n{disp}"
        )
        if ans.strip().lower() == "ok":
            return tasks
        drops = {int(x) for x in re.findall(r"\d+", ans)}
        return [t for i, t in enumerate(tasks) if i not in drops]

    def generate_and_split(self) -> None:
        """
        Generates tasks, applies question-answering processing, and splits them into training and holdout sets.

        The method first generates tasks using either embeddings or a simple method based on configuration.
        It then processes the tasks with a QA step. If the total number of tasks is less than or equal to the
        configured number of holdout tasks, all tasks are assigned to training and none to holdout. Otherwise,
        the tasks are shuffled and split into holdout and training sets according to the configuration.

        Logs the number of training and holdout tasks after splitting.
        """
        tasks = (
            self._gen_with_embeddings()
            if self.cfg.use_embeddings_for_tasks
            else self._gen_simple()
        )
        tasks = self._qa(tasks)
        if len(tasks) < self.cfg.num_holdout_tasks + 1:
            self.train_tasks = tasks
            self.holdout_tasks = []
        else:
            random.shuffle(tasks)
            self.holdout_tasks = tasks[: self.cfg.num_holdout_tasks]
            self.train_tasks = tasks[self.cfg.num_holdout_tasks :]
        logger.bind(train=len(self.train_tasks), holdout=len(self.holdout_tasks)).info(
            "Task split"
        )


# ─────────────────────────── EvaluationEngine ────────────────────────────
class EvaluationEngine:
    """
    EvaluationEngine applies a set of evaluators (agents) to assess prompts against tasks using a rubric, aggregates their scores, and manages evaluation workflows.

    Attributes:
        cfg (CLPOConfig): Configuration object containing evaluator settings.
        applicator: An agent or tool used to apply the rubric.
        proxy: Proxy agent for managing chat interactions.
        evaluators (List[Tuple[AssistantAgent, float]]): List of evaluator agents and their associated weights.

    Methods:
        __init__(cfg, factory):
            Initializes the EvaluationEngine with configuration and agent factory.

        _normalize(lst):
            Normalizes evaluator configuration into names and weights.

        _generate_plan(task, prompt):
            Generates a plan for a given task and prompt using the proxy agent.

        _get_score(agent, msg):
            Gets a score from an evaluator agent for a given message.

        _score_raw(plan, rubric):
            Queries all evaluators for scores on a plan and rubric, parsing their markdown table responses.

        _weighted(scores):
            Computes the weighted sum of scores based on evaluator weights.

        _merge_subs(subs):
            Merges subscores from all evaluators using their weights.

        evaluate_prompt(prompt, tasks, rubric, s, k):
            Evaluates a single prompt over sampled tasks and updates its scores and subscores.

        evaluate_population(pop, tasks, rubric, s, k):
            Evaluates a population of prompts, applying the evaluation process to each.
    """

    def __init__(self, cfg: CLPOConfig, factory: AgentFactory) -> None:
        self.cfg = cfg
        self.applicator = factory.create_applicator()
        self.proxy = factory.get_proxy("eval_proxy")
        names, weights = self._normalize(cfg.evaluators_config)
        self.evaluators: List[Tuple[AssistantAgent, float]] = list(
            zip(factory.create_evaluators(names), weights)
        )

    @staticmethod
    def _normalize(lst: List[EvaluatorConfigType]) -> Tuple[List[str], List[float]]:
        if all(isinstance(x, str) for x in lst):
            return list([str(x) for x in lst]), [1.0] * len(lst)
        names, w = zip(*[(str(a), float(b)) for a, b in lst])
        total = sum(w)
        return list(names), [wi / total for wi in w]

    @retry_on_exception(exceptions=(OpenAIError,))
    def _generate_plan(self, task: str, prompt: str) -> str:
        return (
            self.proxy.initiate_chat(
                self.applicator,
                message=f"TASK:\n{task}\n---\nPROMPT:\n{prompt}",
                max_turns=1,
                summary_method="last_msg",
            ).summary
            or "Error"
        )

    @retry_on_exception(exceptions=(OpenAIError,))
    def _get_score(self, agent: AssistantAgent, msg: str) -> str:
        return (
            self.proxy.initiate_chat(
                agent,
                message=msg,
                max_turns=1,
                summary_method="last_msg",
            ).summary
            or ""
        )

    def _score_raw(
        self, plan: str, rubric: str
    ) -> Tuple[List[float], List[Dict[str, float]]]:
        """Query every evaluator and parse their returned markdown table."""
        prompt_msg = (
            f"Rubric:\n{rubric}\n\n"
            f"Plan to evaluate:\n{plan}\n\n"
            "Return the full markdown table including TOTAL score:"
        )
        totals: list[float] = []
        subs: list[dict[str, float]] = []
        for agent, _ in self.evaluators:
            txt = self._get_score(agent, prompt_msg)
            t, s = parse_markdown_scores(txt)
            totals.append(t)
            subs.append(s)
        return totals, subs

    def _weighted(self, scores: List[float]) -> float:
        """
        Apply per-evaluator z-score normalization plus logistic stretch,
        then weight and sum.
        """
        arr = np.array(scores, dtype=float)
        mu = arr.mean()
        sigma = arr.std()
        # avoid divide-by-zero
        if sigma > 1e-6:
            z = (arr - mu) / sigma
        else:
            z = np.zeros_like(arr)
        stretched = 1 / (1 + np.exp(-z))
        weights = [w for _, w in self.evaluators]
        return float((stretched * weights).sum())

    def _merge_subs(self, subs: List[Dict[str, float]]) -> Dict[str, float]:
        weights = [w for _, w in self.evaluators]
        merged: Dict[str, float] = {}
        for key in set().union(*subs):
            vals = [d.get(key, 0.0) for d in subs]
            merged[key] = float(np.average(vals, weights=weights))
        return merged

    def evaluate_prompt(
        self, prompt: "PromptData", tasks: List[str], rubric: str, s: int, k: int
    ) -> None:
        prompt.reset_eval()
        sampled = random.sample(tasks, min(k, len(tasks)))
        for task in sampled:
            for _ in range(s):
                plan = self._generate_plan(task, prompt.text)
                if plan == "Error":
                    prompt.scores.append(0.0)
                    prompt.subscores.append({})
                else:
                    totals, subs = self._score_raw(plan, rubric)
                    prompt.scores.append(self._weighted(totals))
                    prompt.subscores.append(self._merge_subs(subs))
        prompt.update_ema()

    def evaluate_population(
        self, pop: List["PromptData"], tasks: List[str], rubric: str, s: int, k: int
    ) -> None:
        for p in tqdm(pop, desc=f"Eval s={s}, k={k}"):
            self.evaluate_prompt(p, tasks, rubric, s, k)


# ─────────────────────────── RubricEngine ────────────────────────────────
class RubricEngine:
    """
    RubricEngine is responsible for generating an evaluation rubric using a language model agent.

    Attributes:
        cfg (CLPOConfig): Configuration object containing settings for rubric generation.
        proxy: Proxy object used to initiate chat with the rubric generator agent.
        agent: The rubric generator agent instance.

    Methods:
        generate() -> str:
            Generates an evaluation rubric by initiating a chat with the agent.
            Returns the rubric as a string.
            Raises ValueError if the generated rubric does not contain the required "TOTAL:" section.
    """

    def __init__(self, cfg: CLPOConfig, factory: AgentFactory) -> None:
        self.cfg = cfg
        self.proxy = factory.get_proxy("rubric_gen_initiator")
        self.agent = factory.create_rubric_generator()

    @retry_on_exception()
    def generate(self) -> str:
        resp = self.proxy.initiate_chat(
            self.agent,
            message="Create rubric now as specified.",
            max_turns=1,
            summary_method="last_msg",
        )
        raw = resp.summary or ""
        if "TOTAL:" not in raw:
            raise ValueError("Rubric format invalid: missing TOTAL")
        return raw


# ─────────────────────────── PromptData ─────────────────────────────────
class PromptData:
    """
    PromptData holds information about a single prompt, including its text, evaluation scores, TrueSkill rating, and an exponentially moving average (EMA) of its scores.

    Attributes:
        text (str): The text of the prompt.
        scores (List[float]): List of evaluation scores for the prompt.
        subscores (List[Dict[str, float]]): List of dictionaries containing subscores for the prompt.
        ema (float): Exponentially moving average of the scores.
        id (str): Unique identifier for the prompt.
        rating (trueskill.Rating): TrueSkill rating object for the prompt.

    Methods:
        fitness() -> float:
            Calculates the fitness score as the TrueSkill mean minus twice the standard deviation.

        update_ema() -> None:
            Updates the EMA based on the current scores, penalizing by 0.3 times the standard deviation.

        reset_eval() -> None:
            Clears all scores and subscores, and resets the EMA to zero.

        __repr__() -> str:
            Returns a string representation of the prompt, including its ID, fitness, and EMA.
    """

    __slots__ = ("text", "scores", "subscores", "ema", "id", "rating")

    def __init__(self, text: str) -> None:
        self.text: str = text
        self.scores: List[float] = []
        self.subscores: List[Dict[str, float]] = []
        self.ema: float = 0.0
        self.id: str = uuid.uuid4().hex[:8]
        self.rating = trueskill.Rating()

    def fitness(self) -> float:
        """
        Calculates a fitness score based on the TrueSkill rating and the length of the prompt text.

        The fitness score is computed as the TrueSkill mean (`mu`) minus twice the standard deviation (`sigma`),
        with an additional penalty of 0.1 for every 50 tokens in the prompt text.

        Returns:
            float: The computed fitness score.
        """
        base = self.rating.mu - 2 * self.rating.sigma
        tokens = len(self.text.split())
        penalty = 0.1 * (tokens / 50)
        return base - penalty

    def update_ema(self) -> None:
        """
        Updates the exponential moving average (EMA) of the scores.

        If the `scores` list is not empty, calculates the mean and standard deviation of the scores,
        then updates `self.ema` to be the mean minus 0.3 times the standard deviation.
        If the `scores` list is empty, sets `self.ema` to 0.0.
        """
        if self.scores:
            m, sd = float(np.mean(self.scores)), float(np.std(self.scores))
            self.ema = m - 0.3 * sd
        else:
            self.ema = 0.0

    def reset_eval(self) -> None:
        self.scores.clear()
        self.subscores.clear()
        self.ema = 0.0

    def __repr__(self) -> str:
        return f"Prompt(id={self.id}, fitness={self.fitness():.2f}, ema={self.ema:.2f})"


# ─────────────────────────── InitialPopulationGenerator ──────────────────
class InitialPopulationGenerator:
    """
    InitialPopulationGenerator is responsible for generating an initial population of prompts for a given task using a language model agent.

    Args:
        cfg (CLPOConfig): Configuration object containing templates and settings for population generation.
        factory (AgentFactory): Factory object used to create the agent and proxy required for prompt generation.

    Methods:
        generate(base: str, n: int) -> List["PromptData"]:
            Generates a list of n PromptData objects for the specified base task.
            Utilizes a language model agent to create prompts, parsing the output for valid prompt entries.
            Retries on OpenAIError exceptions.
    """

    def __init__(self, cfg: CLPOConfig, factory: AgentFactory):
        self.cfg = cfg
        self.agent = factory.create_initial_pop_generator()
        self.proxy = factory.get_proxy("init_pop_proxy")

    @retry_on_exception(exceptions=(OpenAIError,))
    def generate(self, base: str, n: int) -> List["PromptData"]:  # noqa: F821
        msg = self.cfg.initial_pop_user_template.format(n=n, task=base)
        raw = (
            self.proxy.initiate_chat(
                self.agent, message=msg, max_turns=1, summary_method="last_msg"
            ).summary
            or ""
        )
        items = re.findall(r"```text\s*(.*?)\s*```", raw, re.DOTALL)
        if not items:
            items = [
                re.sub(r"^\d+[\.\)]\s*", "", l).strip()
                for l in raw.splitlines()
                if l.strip() and l[0].isdigit()
            ]
        return [PromptData(t) for t in items[:n]]


# ─────────────────────────── TuningEngine ───────────────────────────────
class TuningEngine:
    """
    TuningEngine is responsible for generating a new child prompt by combining the text of two parent prompts.
    It utilizes an agent and a proxy, both provided by an AgentFactory, to perform the prompt tuning operation.

    Attributes:
        agent: An instance of a prompt tuner agent created by the factory.
        proxy: A proxy object used to facilitate communication with the agent.

    Methods:
        tune(parents: List[PromptData]) -> PromptData:
            Combines the text of the given parent prompts and uses the agent and proxy to generate a new prompt.
            If the agent fails to generate a new prompt, the text of the first parent is returned.
    """

    def __init__(self, factory: AgentFactory) -> None:
        self.agent = factory.create_prompt_tuner()
        self.proxy = factory.get_proxy("tuner_proxy")

    @retry_on_exception(exceptions=(OpenAIError,))
    def tune(self, parents: List[PromptData]) -> PromptData:
        block = "\n---\n".join(p.text for p in parents)
        text = (
            self.proxy.initiate_chat(
                self.agent,
                message=block,
                max_turns=1,
                summary_method="last_msg",
            ).summary
            or parents[0].text
        )
        return PromptData(text.strip())


# ─────────────────────────── CLPO_Orchestrator ──────────────────────────
class CLPO_Orchestrator:
    """
    CLPO_Orchestrator orchestrates the end-to-end process of prompt optimization using a configurable evolutionary strategy.

    This class manages the initialization, evaluation, and evolution of a population of prompts, leveraging various engines for task generation, rubric creation, evaluation, and tuning. It supports checkpointing and resuming via state files, tracks budget usage, and maintains a history of the best prompts found.

    Args:
        cfg (CLPOConfig): Configuration object specifying evolutionary and evaluation parameters.
        base_prompt (str): The initial prompt to seed the population.
        state_file (pathlib.Path | None, optional): Path to the state file for checkpointing. Defaults to "clpo_state.jsonl".
        resume (bool, optional): Whether to resume from a previous state file. Defaults to False.

    Attributes:
        cfg (CLPOConfig): Configuration for the orchestrator.
        base_prompt (str): The base prompt used for initialization.
        state_file (pathlib.Path): Path to the state file for checkpointing.
        resume_flag (bool): Indicates if resume mode is enabled.
        factory (AgentFactory): Factory for creating agents.
        task_engine (TaskEngine): Engine for generating and splitting tasks.
        rubric_engine (RubricEngine): Engine for generating evaluation rubrics.
        tuner (TuningEngine): Engine for generating offspring prompts.
        init_gen (InitialPopulationGenerator): Generator for the initial prompt population.
        eval_engine (Optional[EvaluationEngine]): Engine for evaluating prompts.
        population (List[PromptData]): Current population of prompts.
        best_history (List[PromptData]): History of best prompts per generation.
        rubric (str): Evaluation rubric.
        budget_used (float): Total budget used in USD.

    Methods:
        __enter__(): Context manager entry; starts logging usage and costs.
        __exit__(exc_type, exc, tb): Context manager exit; stops logging and prints usage summary.
        _update_budget(): Updates the budget used based on evaluation costs.
        _update_ratings(): Updates prompt ratings using TrueSkill based on evaluation scores.
        _halving(candidates): Performs successive halving selection on candidate prompts.
        prepare(): Prepares tasks and rubric if not already present (resume-safe).
        _prompt_to_dict(p): Serializes a PromptData object to a dictionary.
        _dict_to_prompt(d): Deserializes a dictionary to a PromptData object.
        _save_state(generation): Saves the current state to the state file.
        _load_state(): Loads the last saved state from the state file.
        run(): Runs the evolutionary optimization process, managing generations, evaluation, and checkpointing.
    """

    def __init__(
        self,
        cfg: CLPOConfig,
        base_prompt: str,
        *,
        state_file: pathlib.Path | None = None,
        resume: bool = False,
    ) -> None:
        self.cfg = cfg
        self.base_prompt = base_prompt
        self.state_file = state_file or pathlib.Path("clpo_state.jsonl")
        self.resume_flag = resume

        # Factories / engines
        self.factory = AgentFactory(cfg)
        self.task_engine = TaskEngine(cfg, self.factory)
        self.rubric_engine = RubricEngine(cfg, self.factory)
        self.tuner = TuningEngine(self.factory)
        self.init_gen = InitialPopulationGenerator(cfg, self.factory)  # ← restored

        # Runtime state
        self.eval_engine: Optional[EvaluationEngine] = None
        self.population: List[PromptData] = []
        self.best_history: List[PromptData] = []
        self.rubric: str = ""
        self.budget_used: float = 0.0

    def __enter__(self) -> "CLPO_Orchestrator":
        autogen.ChatCompletion.start_logging(cost_map=self.cfg.get_autogen_cost_map())
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        usage = autogen.ChatCompletion.get_usage_summary(clear_history=False)
        autogen.ChatCompletion.stop_logging()
        if usage:
            autogen.ChatCompletion.print_usage_summary(usage_summary=usage)

    def _update_budget(self) -> None:
        if autogen.ChatCompletion.is_logging():
            s = autogen.ChatCompletion.get_usage_summary()
            if s and "total_cost" in s:
                self.budget_used = s["total_cost"]

    def _update_ratings(self) -> None:
        for i, pi in enumerate(self.population):
            for pj in self.population[i + 1 :]:
                if not pi.scores or not pj.scores:
                    # Cannot compare yet – postpone until both have data
                    continue
                si, sj = float(np.mean(pi.scores)), float(np.mean(pj.scores))
                if abs(si - sj) < 1e-6:
                    pi.rating, pj.rating = trueskill.rate_1vs1(
                        pi.rating, pj.rating, drawn=True
                    )
                elif si > sj:
                    pi.rating, pj.rating = trueskill.rate_1vs1(pi.rating, pj.rating)
                else:
                    pj.rating, pi.rating = trueskill.rate_1vs1(pj.rating, pi.rating)

    def _halving(
        self, candidates: List["PromptData"]
    ) -> List["PromptData"]:  # noqa: F821
        """
        Performs successive halving on a list of candidate PromptData objects.

        This method iteratively evaluates and prunes the candidate list according to the configured
        successive halving levels (`sh_levels`). At each level, candidates are evaluated using the
        evaluation engine, and only the top-performing half (or at least one) are retained for the
        next round. The process continues until the budget is exhausted or all levels are completed.

        Args:
            candidates (List["PromptData"]): The initial list of candidate PromptData objects to be evaluated.

        Returns:
            List["PromptData"]: The top-performing candidates after successive halving, limited to `self.cfg.mu`.

        Raises:
            RuntimeError: If the evaluation engine is not ready.
        """
        alive = candidates
        for level, (s, k) in enumerate(self.cfg.sh_levels):
            if self.budget_used >= self.cfg.budget_usd:
                logger.info("Budget exhausted before level %d", level)
                break
            if self.eval_engine is None:
                raise RuntimeError("EvaluationEngine not ready")
            self.eval_engine.evaluate_population(
                alive, self.task_engine.train_tasks, self.rubric, s, k
            )
            self._update_budget()
            self._update_ratings()
            alive.sort(key=lambda p: p.fitness(), reverse=True)
            if level < len(self.cfg.sh_levels) - 1:
                alive = alive[: max(1, len(alive) // 2)]
        alive.sort(key=lambda p: p.fitness(), reverse=True)
        return alive[: self.cfg.mu]

    def prepare(self) -> None:
        """
        Prepares the necessary tasks and rubric for evaluation if they do not already exist.

        This method ensures that task generation and rubric creation are performed only once,
        making the process resume-safe. If training tasks are missing, it generates and splits them,
        then updates the budget. Similarly, if the rubric is missing, it generates the rubric and
        updates the budget. Finally, it initializes the evaluation engine with the current configuration
        and factory.

        Raises:
            Any exceptions raised by the task or rubric generation engines.
        """
        if not self.task_engine.train_tasks:
            self.task_engine.generate_and_split()
            self._update_budget()

        if not self.rubric:
            self.rubric = self.rubric_engine.generate()
            self._update_budget()

        self.eval_engine = EvaluationEngine(self.cfg, self.factory)

    def _prompt_to_dict(self, p: PromptData) -> dict[str, Any]:
        return {
            "id": p.id,
            "text": p.text,
            "scores": p.scores,
            "ema": p.ema,
            "mu": p.rating.mu,
            "sigma": p.rating.sigma,
        }

    def _dict_to_prompt(self, d: dict[str, Any]) -> PromptData:
        p = PromptData(d["text"])
        p.id = d["id"]
        p.scores = d["scores"]
        p.ema = d["ema"]  # keep stored EMA
        p.rating = trueskill.Rating(mu=d["mu"], sigma=d["sigma"])
        p.update_ema()  # recompute in case scores changed
        return p

    def _save_state(self, *, generation: int) -> None:
        """
        Saves the current state of the object to a file.

        This method serializes the current generation, budget used, population, best history,
        training and holdout tasks, and rubric into a dictionary and writes it as a JSON string
        to the state file. The file is overwritten each time to prevent unbounded growth.
        This allows for resuming runs with identical data.

        Args:
            generation (int): The current generation number to be saved.

        Returns:
            None
        """
        rec = {
            "generation": generation,
            "budget": self.budget_used,
            "population": [self._prompt_to_dict(p) for p in self.population],
            "best_history": [self._prompt_to_dict(p) for p in self.best_history],
            # Persist benchmark so resume runs on *identical* data
            "train_tasks": self.task_engine.train_tasks,
            "holdout_tasks": self.task_engine.holdout_tasks,
            "rubric": self.rubric,
        }
        # overwrite (not append) to avoid unbounded growth
        with self.state_file.open("w", encoding="utf-8") as fh:
            fh.write(json.dumps(rec) + "\n")

    def _load_state(self) -> bool:
        """
        Loads the state of the object from a file.

        Attempts to read the state from `self.state_file`, parse the last line as JSON,
        and restore the object's attributes (`population`, `best_history`, `budget_used`,
        `train_tasks`, `holdout_tasks`, and `rubric`) from the parsed data.

        Returns:
            bool: True if the state was successfully loaded and restored, False otherwise.
        """
        try:
            text = self.state_file.read_text(encoding="utf-8").strip()
            if not text:
                return False
            data = json.loads(text.splitlines()[-1])
        except (OSError, json.JSONDecodeError):
            logger.warning(
                "Could not load state file %s; starting fresh", self.state_file
            )
            return False

        self.population = [self._dict_to_prompt(d) for d in data["population"]]
        self.best_history = [self._dict_to_prompt(d) for d in data["best_history"]]
        self.budget_used = data["budget"]
        self.task_engine.train_tasks = data.get("train_tasks", [])
        self.task_engine.holdout_tasks = data.get("holdout_tasks", [])
        self.rubric = data.get("rubric", "")
        return True

    def run(self) -> None:
        """
        Executes the main evolutionary optimization loop for prompt generation and evaluation.

        This method manages the population of prompts, including initialization, evaluation,
        and successive generations of evolution. It performs the following steps:

        1. Ensures the population is initialized, either by generating new prompts or resuming from a previous state.
        2. Evaluates the initial population if none of the prompts have been scored yet.
        3. Iteratively generates offspring, evaluates them, and selects the best prompts using a halving strategy for a specified number of generations.
        4. Tracks and logs the best prompt in each generation, updates the evaluation budget, and saves the state after each generation.
        5. Stops the process if the evaluation budget is exceeded.
        6. Logs the final best prompt after completion.

        Raises:
            RuntimeError: If the evaluation engine is not initialized when required.
        """
        # Ensure population exists (resume logic lives in main)
        if not self.population:
            self.population = self.init_gen.generate(self.base_prompt, self.cfg.mu) or [
                PromptData(self.base_prompt) for _ in range(self.cfg.mu)
            ]

        # Initial evaluation if none of the prompts have scores yet
        if not any(p.scores for p in self.population):
            s0, k0 = self.cfg.sh_levels[0]
            if self.eval_engine is None:
                raise RuntimeError("EvaluationEngine not initialized")
            self.eval_engine.evaluate_population(
                self.population,
                self.task_engine.train_tasks,
                self.rubric,
                s0,
                k0,
            )
            self._update_budget()
            self._update_ratings()
            self._save_state(generation=0)

        # Successive generations
        for gen in range(1, self.cfg.generations + 1):
            # — Progressive reveal of 1 unseen task every 4 gens —
            if gen % 4 == 0 and self.task_engine.holdout_tasks:
                retired = self.task_engine.train_tasks.pop(0)
                new_task = self.task_engine.holdout_tasks.pop(0)
                self.task_engine.train_tasks.append(new_task)
                self.task_engine.holdout_tasks.append(retired)
                logger.bind(gen=gen).info("Swapped task into train set: %s", new_task)

            # generate offspring
            offspring = [
                self.tuner.tune(random.sample(self.population, 2))
                for _ in range(self.cfg.lam)
            ]
            # halving + evaluation
            self.population = self._halving(self.population + offspring)
            self.best_history.append(self.population[0])

            logger.bind(
                gen=gen,
                best_fitness=self.population[0].fitness(),
                budget=self.budget_used,
            ).info("Generation complete")

            self._save_state(generation=gen)

            if self.budget_used > self.cfg.budget_usd:
                logger.bind(gen=gen).info("Stopping: budget limit reached.")
                break

        # Log the best prompt
        best = self.population[0]
        logger.bind(final_fitness=best.fitness()).info(
            "=== BEST PROMPT ===\n{}", best.text
        )


# ─────────────────────────── Main ───────────────────────────────────────
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
