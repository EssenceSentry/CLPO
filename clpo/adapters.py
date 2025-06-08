from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Dict, Type, Union


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
