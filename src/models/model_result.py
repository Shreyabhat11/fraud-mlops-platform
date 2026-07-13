"""
Model Result Object

Stores everything produced by a trained model.

Author:
Enterprise Fraud Detection Platform
"""

from dataclasses import dataclass

import numpy as np


@dataclass
class ModelResult:

    name: str

    model: object

    probabilities: np.ndarray

    predictions: np.ndarray

    metrics: dict

    threshold: float

    training_time: float

    feature_importance: dict | None = None