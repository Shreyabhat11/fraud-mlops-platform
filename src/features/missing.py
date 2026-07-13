"""
Missing Value Transformer

Handles missing values in a production-ready manner.

Strategy

Numerical
    Median

Categorical
    Missing

High Missing Columns
    Drop

Author
Enterprise Fraud Detection Platform
"""

from __future__ import annotations

import pandas as pd

from src.features.base import BaseTransformer
from src.utils.logger import get_logger

logger = get_logger(__name__)


class MissingValueTransformer(BaseTransformer):

    def __init__(

        self,

        missing_threshold: float = 0.95,

    ):

        super().__init__()

        self.threshold = missing_threshold

        self.numeric_fill = {}

        self.categorical_fill = {}

        self.drop_columns = []

    # ----------------------------------------------------

    def fit(

        self,

        X,

        y=None

    ):

        logger.info("Fitting MissingValueTransformer")

        missing_percent = X.isna().mean()

        self.drop_columns = missing_percent[
            missing_percent > self.threshold
        ].index.tolist()

        X = X.drop(columns=self.drop_columns)

        numerical = X.select_dtypes(
            include="number"
        )

        categorical = X.select_dtypes(
            exclude="number"
        )

        self.numeric_fill = numerical.median().to_dict()

        self.categorical_fill = {

            col: "Missing"

            for col in categorical.columns

        }

        self.metadata = {

            "columns_dropped":

                self.drop_columns,

            "dropped_count":

                len(self.drop_columns),

            "numerical_imputed":

                len(self.numeric_fill),

            "categorical_imputed":

                len(self.categorical_fill),

            "strategy_numeric":

                "median",

            "strategy_categorical":

                "Missing",

        }

        self.is_fitted = True

        logger.info(

            f"Dropped {len(self.drop_columns)} columns."

        )

        return self

    # ----------------------------------------------------

    def transform(

        self,

        X

    ):

        if not self.is_fitted:

            raise RuntimeError(

                "Transformer not fitted."

            )

        X = X.copy()

        X = X.drop(

            columns=self.drop_columns,

            errors="ignore"

        )

        for col, value in self.numeric_fill.items():

            if col in X.columns:

                X[col] = X[col].fillna(value)

        for col, value in self.categorical_fill.items():

            if col in X.columns:

                X[col] = X[col].fillna(value)

        logger.info(

            "Missing value transformation completed."

        )

        return X