"""
Enterprise Model Evaluator

Evaluates trained models, ranks them, and selects the best model.

Author:
Enterprise Fraud Detection Platform
"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from src.models.metrics import MetricsCalculator
from src.utils.logger import get_logger

logger = get_logger(__name__)


class ModelEvaluator:

    def __init__(self):

        self.results = []

    # -----------------------------------------------------

    def evaluate(

        self,

        model_results,

        y_valid,

    ):

        logger.info("Evaluating models...")

        leaderboard = []

        for result in model_results:

            metrics = MetricsCalculator.evaluate(

                y_valid,

                result.probabilities,

            )

            result.metrics = metrics

            result.threshold = metrics["Threshold"]

            leaderboard.append({

                "Model": result.name,

                **metrics,

                "Training_Time": round(

                    result.training_time,

                    2,

                ),

            })

        leaderboard = pd.DataFrame(

            leaderboard

        )

        leaderboard = leaderboard.sort_values(

            "ROC_AUC",

            ascending=False,

        ).reset_index(drop=True)

        self.results = leaderboard

        logger.info("\n%s", leaderboard)

        return leaderboard

    # -----------------------------------------------------

    def best_model(

        self,

        model_results,

    ):

        best = self.results.iloc[0]["Model"]

        for result in model_results:

            if result.name == best:

                return result

    # -----------------------------------------------------

    def save(

        self,

        output_directory,

    ):

        output_directory = Path(output_directory)

        output_directory.mkdir(

            parents=True,

            exist_ok=True,

        )

        csv_path = (

            output_directory /

            "leaderboard.csv"

        )

        json_path = (

            output_directory /

            "leaderboard.json"

        )

        self.results.to_csv(

            csv_path,

            index=False,

        )

        with open(

            json_path,

            "w",

            encoding="utf-8",

        ) as file:

            json.dump(

                self.results.to_dict(

                    orient="records"

                ),

                file,

                indent=4,

            )

        logger.info(

            "Leaderboard saved."

        )