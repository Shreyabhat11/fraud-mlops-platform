"""
Enterprise Model Trainer

- Model agnostic
- Callback ready
- MLflow ready
- Optuna ready
- Timing support
- Early stopping support

Author:
Enterprise Fraud Detection Platform
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any

import pandas as pd

from src.models.model_factory import ModelFactory
from src.models.model_result import ModelResult
from src.utils.logger import get_logger

logger = get_logger(__name__)


@dataclass(slots=True)
class TrainingContext:

    X_train: pd.DataFrame
    y_train: pd.Series

    X_valid: pd.DataFrame
    y_valid: pd.Series


class ModelTrainer:

    def __init__(self):

        self.factory = ModelFactory()

    # -----------------------------------------------------

    def train_all(

        self,

        context: TrainingContext,

    ):

        results = []

        for spec in self.factory.create():

            logger.info("=" * 60)

            logger.info(f"Training {spec.name}")

            start = time.perf_counter()

            estimator = spec.estimator

            fit_kwargs = self._fit_kwargs(

                spec.name,

                context,

            )

            estimator.fit(

                context.X_train,

                context.y_train,

                **fit_kwargs,

            )

            elapsed = round(

                time.perf_counter() - start,

                2,

            )

            probability = estimator.predict_proba(

                context.X_valid

            )[:, 1]

            prediction = (

                probability >= 0.5

            ).astype(int)

            importance = None

            if hasattr(

                estimator,

                "feature_importances_",

            ):

                importance = dict(

                    zip(

                        context.X_train.columns,

                        estimator.feature_importances_,

                    )

                )

            results.append(

                ModelResult(

                    name=spec.name,

                    model=estimator,

                    probabilities=probability,

                    predictions=prediction,

                    metrics={},

                    threshold=0.5,

                    training_time=elapsed,

                    feature_importance=importance,

                )

            )

            logger.info(

                f"{spec.name} finished in {elapsed}s"

            )

        return results

    # -----------------------------------------------------

    @staticmethod
    def _fit_kwargs(

        name: str,

        context: TrainingContext,

    ) -> dict[str, Any]:

        kwargs = {}

        if name == "XGBoost":

            kwargs = {

                "eval_set": [

                    (

                        context.X_valid,

                        context.y_valid,

                    )

                ],

                "verbose": False,

            }

        elif name == "LightGBM":

            kwargs = {

                "eval_set": [

                    (

                        context.X_valid,

                        context.y_valid,

                    )

                ]

            }

        elif name == "CatBoost":

            kwargs = {

                "eval_set": (

                    context.X_valid,

                    context.y_valid,

                ),

                "use_best_model": True,

            }

        return kwargs