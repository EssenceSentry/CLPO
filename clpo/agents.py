from __future__ import annotations

import uuid
from typing import Dict, List, Union # TYPE_CHECKING removed as it's no longer needed for CLPOConfig

from autogen import AssistantAgent, UserProxyAgent

from .adapters import adapter_class_for
from .config import CLPOConfig # Updated import for CLPOConfig


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
