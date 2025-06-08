from __future__ import annotations

import json
import pathlib
import random
from typing import Any, List, Optional # TYPE_CHECKING removed

import autogen # For autogen.ChatCompletion
import numpy as np
import trueskill
from loguru import logger

from .agents import AgentFactory
from .data_models import PromptData
from .engines import (
    EvaluationEngine,
    InitialPopulationGenerator,
    RubricEngine,
    TaskEngine,
    TuningEngine,
)

from .config import CLPOConfig # Updated import for CLPOConfig


class CLPO_Orchestrator:
    """
    CLPO_Orchestrator orchestrates the end-to-end process of prompt optimization using a configurable evolutionary strategy.
    ... (docstring from clpo.py)
    """

    def __init__(
        self,
        cfg: "CLPOConfig",
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
        self.init_gen = InitialPopulationGenerator(cfg, self.factory)

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
        self, candidates: List[PromptData]
    ) -> List[PromptData]:
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
        p.ema = d["ema"]
        p.rating = trueskill.Rating(mu=d["mu"], sigma=d["sigma"])
        p.update_ema()
        return p

    def _save_state(self, *, generation: int) -> None:
        rec = {
            "generation": generation,
            "budget": self.budget_used,
            "population": [self._prompt_to_dict(p) for p in self.population],
            "best_history": [self._prompt_to_dict(p) for p in self.best_history],
            "train_tasks": self.task_engine.train_tasks,
            "holdout_tasks": self.task_engine.holdout_tasks,
            "rubric": self.rubric,
        }
        with self.state_file.open("w", encoding="utf-8") as fh:
            fh.write(json.dumps(rec) + "\n")

    def _load_state(self) -> bool:
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
        if not self.population:
            self.population = self.init_gen.generate(self.base_prompt, self.cfg.mu) or [
                PromptData(self.base_prompt) for _ in range(self.cfg.mu)
            ]

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

        for gen in range(1, self.cfg.generations + 1):
            if gen % 4 == 0 and self.task_engine.holdout_tasks:
                retired = self.task_engine.train_tasks.pop(0)
                new_task = self.task_engine.holdout_tasks.pop(0)
                self.task_engine.train_tasks.append(new_task)
                self.task_engine.holdout_tasks.append(retired)
                logger.bind(gen=gen).info("Swapped task into train set: %s", new_task)

            offspring = [
                self.tuner.tune(random.sample(self.population, 2))
                for _ in range(self.cfg.lam)
            ]
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

        best = self.population[0]
        logger.bind(final_fitness=best.fitness()).info(
            "=== BEST PROMPT ===\n{}", best.text
        )
