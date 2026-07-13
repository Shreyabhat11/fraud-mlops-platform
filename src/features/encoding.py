"""
Categorical Encoding Transformer

Hybrid encoding strategy:

- Low cardinality      -> Label Encoding
- High cardinality     -> Frequency Encoding
- Rare categories      -> "__RARE__"

Author:
Enterprise Fraud Detection Platform
"""

from __future__ import annotations

import pandas as pd

from src.features.base import BaseTransformer
from src.utils.logger import get_logger

logger = get_logger(__name__)


class CategoricalEncoder(BaseTransformer):

    def __init__(

        self,

        low_cardinality=10,

        rare_threshold=0.005,

    ):

        super().__init__()

        self.low_cardinality = low_cardinality

        self.rare_threshold = rare_threshold

        self.frequency_maps = {}

        self.label_maps = {}

    def fit(self, X, y=None):

        logger.info("Fitting categorical encoder")

        categorical = X.select_dtypes(
            include="object"
        )

        for col in categorical.columns:

            value_counts = categorical[col].value_counts(
                normalize=True
            )

            rare = value_counts[
                value_counts < self.rare_threshold
            ].index

            series = categorical[col].replace(
                rare,
                "__RARE__"
            )

            if series.nunique() <= self.low_cardinality:

                mapping = {

                    value: idx

                    for idx, value

                    in enumerate(
                        sorted(series.dropna().unique())
                    )

                }

                self.label_maps[col] = mapping

            else:

                freq = series.value_counts(
                    normalize=True
                )

                self.frequency_maps[col] = freq.to_dict()

        self.metadata = {

            "label_encoded":

                list(self.label_maps.keys()),

            "frequency_encoded":

                list(self.frequency_maps.keys()),

        }

        self.is_fitted = True

        return self

    def transform(self, X):

        if not self.is_fitted:

            raise RuntimeError(
                "Transformer not fitted."
            )

        X = X.copy()

        for col, mapping in self.label_maps.items():

            if col not in X.columns:
                continue

            X[col] = (

                X[col]

                .map(mapping)

                .fillna(-1)

                .astype("int32")

            )

        for col, mapping in self.frequency_maps.items():

            if col not in X.columns:
                continue

            X[col] = (

                X[col]

                .map(mapping)

                .fillna(0)

                .astype("float32")

            )

        logger.info("Categorical encoding completed.")

        return X