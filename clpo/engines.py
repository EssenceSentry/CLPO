from __future__ import annotations
import json
import random
import re
import sys
from typing import List, Dict, Tuple

import numpy as np
import openai
from autogen import AssistantAgent, UserProxyAgent
from loguru import logger
from openai.error import OpenAIError
from tqdm import tqdm

from .agent_factory import AgentFactory
from .config import CLPOConfig, EvaluatorConfigType, parse_markdown_scores
from .prompt_data import PromptData
from .utils import retry_on_exception, _cached, _embs_hash, SEED, HAS_SKLEARN, KMeans

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


