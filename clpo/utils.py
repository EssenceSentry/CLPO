import functools
import time
from typing import Callable, ParamSpec, Tuple, Type, TypeVar

from loguru import logger

from .constants import (
    DEFAULT_RETRY_ATTEMPTS,
    DEFAULT_RETRY_DELAY,
    RETRY_BACKOFF_FACTOR,
)

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


# ─────────────────────────── Hashing and Caching Helpers ────────────────────────────────
import hashlib
import json
from typing import Any, Callable, Dict, List

import os
from diskcache import Cache

from .constants import (
    CACHE_DIR_ENV,
    DEFAULT_CACHE_DIR,
    CACHE_EXPIRE_ENV,
    DEFAULT_CACHE_EXPIRE,
)

# Initialize CACHE directly in utils.py
CACHE = Cache(
    os.getenv(CACHE_DIR_ENV, DEFAULT_CACHE_DIR),
    expire=int(os.getenv(CACHE_EXPIRE_ENV, str(DEFAULT_CACHE_EXPIRE))),
)


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


T_cached = TypeVar("T_cached")


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
    key: Dict[str, Any], fn: Callable[[], T_cached], cache_backend: Cache = CACHE # Use the CACHE defined in this module
) -> T_cached:
    """
    Caches the result of a function call using a structural hash of the provided key.

    Args:
        key (Dict[str, Any]): A dictionary representing the cache key. The key is structurally hashed to generate a unique cache identifier.
        fn (Callable[[], T_cached]): A zero-argument function whose result should be cached.
        cache_backend (Cache, optional): The cache backend to use for storing results. Defaults to CACHE.

    Returns:
        T_cached: The result of the function call, either retrieved from the cache or freshly computed.

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


# ─────────────────────────── Markdown Parsing ───────────────────────────────────
import re
from typing import Dict, Tuple # Pattern, Union, Optional - Add if pandas is removed or for stricter typing

# Keep pandas import for now, as per instructions
try:
    import pandas as pd
    HAS_PANDAS = True
except ImportError:
    HAS_PANDAS = False # pragma: no cover

_SCORE_TOTAL = re.compile(r"TOTAL\s*[:\-]?\s*([0-9]+(?:\.[0-9]+)?)", re.I)
_ROW = re.compile(r"\|\s*([^|]+?)\s*\|\s*([0-9\.]+)\s*\|")


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
    if HAS_PANDAS:
        try:
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
