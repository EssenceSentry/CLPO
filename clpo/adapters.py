from abc import ABC, abstractmethod

from typing import Dict, Type, Union

class BaseAdapter(ABC):
    def __init__(self, model: str, cfg: Dict[str, Union[str, float]]):
        self.model = model
        self.cfg = cfg

    @abstractmethod
    def llm_config(self) -> Dict[str, Union[str, float]]:
        ...


class OpenAIAdapter(BaseAdapter):
    def llm_config(self) -> Dict[str, Union[str, float]]:
        return {
            "model": self.model,
            "temperature": float(self.cfg.get("temperature", 0.7)),
            "api_key": self.cfg["openai_api_key"],
        }


class GeminiAdapter(BaseAdapter):
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
    for prefix, cls in ADAPTER_REGISTRY.items():
        if model.startswith(prefix):
            return cls
    raise ValueError(f"No adapter found for model prefix: {model}")
