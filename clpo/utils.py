import functools
import hashlib
import json
try:
    from sklearn.cluster import KMeans
    HAS_SKLEARN = True
except ImportError:  # pragma: no cover
    KMeans = None  # type: ignore
    HAS_SKLEARN = False

import os
import random
import time
from typing import Any, Callable, Dict, Tuple, Type, TypeVar

from diskcache import Cache
from loguru import logger

DEFAULT_RETRY_ATTEMPTS: int = 3
DEFAULT_RETRY_DELAY: float = 1.0
RETRY_BACKOFF_FACTOR: float = 2.0
CACHE_DIR_ENV: str = "CLPO_CACHE_DIR"
CACHE_EXPIRE_ENV: str = "CLPO_CACHE_EXPIRE"
DEFAULT_CACHE_DIR: str = ".clpo_cache"
DEFAULT_CACHE_EXPIRE: int = 86_400

P = TypeVar("P")
R = TypeVar("R")


def retry_on_exception(
    max_attempts: int = DEFAULT_RETRY_ATTEMPTS,
    initial_delay: float = DEFAULT_RETRY_DELAY,
    backoff_factor: float = RETRY_BACKOFF_FACTOR,
    exceptions: Tuple[Type[Exception], ...] = (Exception,),
) -> Callable[[Callable[..., R]], Callable[..., R]]:
    """Retry a function call with exponential back-off on specified exceptions."""

    def decorator(func: Callable[..., R]) -> Callable[..., R]:
        @functools.wraps(func)
        def wrapper(*args: Any, **kwargs: Any) -> R:
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


SEED: int = int(os.getenv("CLPO_SEED", "42"))
random.seed(SEED)

CACHE = Cache(
    os.getenv(CACHE_DIR_ENV, DEFAULT_CACHE_DIR),
    expire=int(os.getenv(CACHE_EXPIRE_ENV, str(DEFAULT_CACHE_EXPIRE))),
)

T = TypeVar("T")


def _freeze(obj: Any) -> Any:
    if isinstance(obj, dict):
        return tuple((k, _freeze(obj[k])) for k in sorted(obj))
    if isinstance(obj, (list, tuple)):
        return tuple(_freeze(v) for v in obj)
    if isinstance(obj, (str, int, float, bool, type(None))):
        return obj
    return (type(obj).__name__, repr(obj))


def _cached(key: Dict[str, Any], fn: Callable[[], T], cache_backend: Cache = CACHE) -> T:
    frozen_key = _freeze(key)
    digest = hashlib.sha1(repr(frozen_key).encode("utf-8")).hexdigest()
    if digest in cache_backend:
        return cache_backend[digest]  # type: ignore[return-value]
    val = fn()
    cache_backend[digest] = val
    return val


def _embs_hash(embs: list[list[float]]) -> str:
    rounded = [[round(x, 6) for x in vec] for vec in embs]
    serial = json.dumps(rounded, sort_keys=True, separators=(",", ":"))
    return hashlib.sha1(serial.encode("utf-8")).hexdigest()
