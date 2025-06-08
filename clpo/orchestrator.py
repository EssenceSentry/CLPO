from __future__ import annotations
import json
import pathlib
import random
from typing import Any, List, Optional

import autogen
import numpy as np
import trueskill
from loguru import logger

from .agent_factory import AgentFactory
from .config import CLPOConfig
from .prompt_data import PromptData
from .engines import TaskEngine, RubricEngine, TuningEngine, InitialPopulationGenerator, EvaluationEngine

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


