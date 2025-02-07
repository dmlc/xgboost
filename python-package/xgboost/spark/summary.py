"""Xgboost training summary integration submodule."""

from dataclasses import dataclass, field
from typing import Dict, List


@dataclass
class XGBoostTrainingSummary:
    """
    A class that holds the training and validation objective history
    of an XGBoost model during its training process.
    """

    train_objective_history: Dict[str, List[float]] = field(default_factory=dict)
    validation_objective_history: Dict[str, List[float]] = field(default_factory=dict)

    @staticmethod
    def from_metrics(
        metrics: Dict[str, Dict[str, List[float]]],
    ) -> "XGBoostTrainingSummary":
        """
        Create an XGBoostTrainingSummary instance from a nested dictionary of metrics.

        Parameters
        ----------
        metrics : dict of str to dict of str to list of float
            A dictionary containing training and validation metrics.
            Example format:
                {
                    "training": {"logloss": [0.1, 0.08]},
                    "validation": {"logloss": [0.12, 0.1]}
                }

        Returns
        -------
        A new instance of XGBoostTrainingSummary.

        """
        train_objective_history = metrics.get("training", {})
        validation_objective_history = metrics.get("validation", {})
        return XGBoostTrainingSummary(
            train_objective_history, validation_objective_history
        )
