"""Evaluation metrics for spatial LLM."""

from collections import defaultdict


class SpatialLLMEvaluator:
    """Computes accuracy metrics for spatial reasoning tasks."""

    def __init__(self):
        self.metrics = defaultdict(list)

    def evaluate_accuracy(self, predictions, targets):
        if not predictions:
            return 0.0
        correct = sum(
            p.strip().lower() == t.strip().lower()
            for p, t in zip(predictions, targets)
        )
        return correct / len(predictions)

    def evaluate_by_task_type(self, predictions, targets, task_types):
        results = defaultdict(lambda: {"correct": 0, "total": 0})
        for pred, target, task_type in zip(predictions, targets, task_types):
            results[task_type]["total"] += 1
            if pred.strip().lower() == target.strip().lower():
                results[task_type]["correct"] += 1
        return {
            task_type: m["correct"] / m["total"]
            for task_type, m in results.items()
        }

    def compute_all_metrics(self, predictions, targets, task_types, difficulties):
        return {
            "overall_accuracy": self.evaluate_accuracy(predictions, targets),
            "by_task_type": self.evaluate_by_task_type(predictions, targets, task_types),
        }
