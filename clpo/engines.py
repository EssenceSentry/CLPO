from __future__ import annotations

import json
import random
import re
import sys
from typing import Dict, List, Tuple, Union # TYPE_CHECKING removed

import numpy as np
import openai
from autogen import AssistantAgent, UserProxyAgent # For EvaluationEngine type hints
from loguru import logger
from openai import OpenAIError # Corrected import
from tqdm import tqdm

from .agents import AgentFactory
from .constants import SEED
from .data_models import PromptData
from .utils import (
    _cached,
    _embs_hash,
    parse_markdown_scores,
    retry_on_exception,
)

# Updated import for CLPOConfig and related types
from .config import CLPOConfig, EvaluatorConfigType

# Sklearn imports - moved here as TaskEngine is the only user
try:
    from sklearn.cluster import KMeans
    HAS_SKLEARN = True
except ImportError:  # pragma: no cover
    KMeans = None  # type: ignore
    HAS_SKLEARN = False


# ─────────────────────────── TaskEngine ──────────────────────────────────
class TaskEngine:
    """
    TaskEngine is responsible for generating and splitting tasks into training and holdout sets.
    ... (docstring from clpo.py)
    """

    def __init__(self, cfg: "CLPOConfig", factory: AgentFactory) -> None:
        self.cfg = cfg
        self.factory = factory
        self.task_gen = factory.create_task_generator()
        self.human = factory.create_human_auditor()
        self.train_tasks: List[str] = []
        self.holdout_tasks: List[str] = []

    @retry_on_exception(exceptions=(OpenAIError,))
    def _gen_simple(self) -> List[str]:
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
    ... (docstring from clpo.py)
    """
    # EvaluatorConfigType is used by CLPOConfig, but EvaluationEngine._normalize uses it too.
    # So, it should be defined or imported here if CLPOConfig is not directly imported at runtime.
    # For now, assuming CLPOConfig's EvaluatorConfigType will be available via type checking.
    # If not, `List[Union[str, Tuple[str, float]]]` can be used directly.
    # Let's use direct type for now to avoid issues if CLPOConfig is not available for this.
    # EvaluatorConfigType = List[Union[str, Tuple[str, float]]] # type: ignore # This local definition is removed


    def __init__(self, cfg: "CLPOConfig", factory: AgentFactory) -> None:
        self.cfg = cfg
        self.applicator = factory.create_applicator()
        self.proxy = factory.get_proxy("eval_proxy")
        names, weights = self._normalize(cfg.evaluators_config)
        self.evaluators: List[Tuple[AssistantAgent, float]] = list(
            zip(factory.create_evaluators(names), weights)
        )

    @staticmethod
    def _normalize(lst: List[EvaluatorConfigType]) -> Tuple[List[str], List[float]]: # Use imported EvaluatorConfigType
        if all(isinstance(x, str) for x in lst):
            return list([str(x) for x in lst]), [1.0] * len(lst)
        names, w = zip(*[(str(a), float(b)) for a, b in lst if isinstance(a, str) and isinstance(b, (int, float))]) # Added type check for safety
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
        arr = np.array(scores, dtype=float)
        mu = arr.mean()
        sigma = arr.std()
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
        self, prompt: PromptData, tasks: List[str], rubric: str, s: int, k: int
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
        self, pop: List[PromptData], tasks: List[str], rubric: str, s: int, k: int
    ) -> None:
        for p in tqdm(pop, desc=f"Eval s={s}, k={k}"):
            self.evaluate_prompt(p, tasks, rubric, s, k)


# ─────────────────────────── RubricEngine ────────────────────────────────
class RubricEngine:
    """
    RubricEngine is responsible for generating an evaluation rubric using a language model agent.
    ... (docstring from clpo.py)
    """
    def __init__(self, cfg: "CLPOConfig", factory: AgentFactory) -> None:
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


# ─────────────────────────── InitialPopulationGenerator ──────────────────
class InitialPopulationGenerator:
    """
    InitialPopulationGenerator is responsible for generating an initial population of prompts for a given task using a language model agent.
    ... (docstring from clpo.py)
    """
    def __init__(self, cfg: "CLPOConfig", factory: AgentFactory):
        self.cfg = cfg
        self.agent = factory.create_initial_pop_generator()
        self.proxy = factory.get_proxy("init_pop_proxy")

    @retry_on_exception(exceptions=(OpenAIError,))
    def generate(self, base: str, n: int) -> List[PromptData]:
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
    ... (docstring from clpo.py)
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

# Ensure Union import is at the top with other typing imports
