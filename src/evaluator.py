# evaluator/evaluator.py

from .reward import RewardCalculator

class Evaluator:
    def __init__(self, mode: str = "filter", threshold: float = 0.9, verbose: bool = False):
        """
        Args:
            mode: 'filter' or 'grpo'
            threshold: filter cutoff (used only in filter mode)
        """
        assert mode in ("filter", "grpo")
        self.mode = mode
        self.threshold = threshold
        self.calculator = RewardCalculator(verbose=verbose, normalize=True if mode == "filter" else False)

    def evaluate(self, predicted: dict, ground_truth: dict) -> dict:
        validity = self.calculator.self_check(predicted)
        if validity["status"] != "VALID":
            result = {"score": 0.0, "diagnostics": {"status": validity["status"], "reason": validity["reason"]}}
            if self.mode == "filter":
                return {"keep": False, **result}
            elif self.mode == "grpo":
                return {"reward": self.calculator.get_min_possible_reward(ground_truth), **result}
            
        score, matches, diagnostics = self.calculator.weighted_accuracy(predicted, ground_truth)

        if self.mode == "filter":
            return {
                "keep": score >= self.threshold,
                "score": score,
                "diagnostics": {
                    "matches": matches,
                    "contributions": diagnostics,
                    "total": score
                }
            }
        elif self.mode == "grpo":
            return {
                "reward": score,
                "diagnostics": {
                    "matches": matches,
                    "contributions": diagnostics,
                    "total": score
                }
            }