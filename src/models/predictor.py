"""
Enterprise Predictor

Handles inference using saved bundle.

Author:
Enterprise Fraud Detection Platform
"""

from __future__ import annotations

import pandas as pd

from src.models.bundle import ModelBundle

from src.utils.logger import get_logger

logger = get_logger(__name__)


class Predictor:

    def __init__(

        self,

        bundle_directory,

    ):

        bundle = ModelBundle(

            bundle_directory

        ).load()

        self.model = bundle["model"]

        self.pipeline = bundle["pipeline"]

        self.threshold = bundle["threshold"]

    # ----------------------------------------------------

    def predict(

        self,

        dataframe: pd.DataFrame,

    ):

        logger.info(

            "Running inference..."

        )

        X = self.pipeline.transform(

            dataframe

        )

        probabilities = self.model.predict_proba(

            X

        )[:, 1]

        predictions = (

            probabilities >= self.threshold

        ).astype(int)

        result = dataframe.copy()

        result["FraudProbability"] = probabilities

        result["Prediction"] = predictions

        return result

    # ----------------------------------------------------

    def predict_single(

        self,

        record: dict,

    ):

        df = pd.DataFrame(

            [record]

        )

        return self.predict(df)