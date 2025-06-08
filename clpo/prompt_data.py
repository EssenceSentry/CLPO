from typing import List, Dict
import uuid
import numpy as np
import trueskill

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


