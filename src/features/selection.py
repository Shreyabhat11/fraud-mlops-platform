"""
Feature Selection Transformer

Removes:
- Constant Features
- Highly Correlated Features
- Low Variance Features

Author:
Enterprise Fraud Detection Platform
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.feature_selection import VarianceThreshold

from src.features.base import BaseTransformer
from src.utils.logger import get_logger

logger = get_logger(__name__)


class FeatureSelector(BaseTransformer):

    def __init__(

        self,

        variance_threshold=0.0,

        correlation_threshold=0.98,

    ):

        super().__init__()

        self.variance_threshold = variance_threshold
        self.correlation_threshold = correlation_threshold

        self.selector = VarianceThreshold(
            threshold=self.variance_threshold
        )

        self.selected_columns = None
        self.correlated_columns = []

    # -------------------------------------------------

    def fit(self, X, y=None):

        logger.info("Running Feature Selection")

        X = X.copy()

        numeric = X.select_dtypes(include=np.number)

        self.selector.fit(numeric)

        keep = numeric.columns[
            self.selector.get_support()
        ]

        numeric = numeric[keep]

        corr = numeric.corr().abs()

        upper = corr.where(

            np.triu(

                np.ones(corr.shape),

                k=1

            ).astype(bool)

        )

        self.correlated_columns = [

            column

            for column in upper.columns

            if any(

                upper[column]

                > self.correlation_threshold

            )

        ]

        self.selected_columns = [

            c

            for c in X.columns

            if c not in self.correlated_columns

        ]

        self.metadata = {

            "selected_features":

                len(self.selected_columns),

            "removed_correlated":

                len(self.correlated_columns),

            "correlation_threshold":

                self.correlation_threshold,

        }

        self.is_fitted = True

        logger.info(

            f"Selected {len(self.selected_columns)} features."

        )

        return self

    # -------------------------------------------------

    def transform(self, X):

        if not self.is_fitted:

            raise RuntimeError("Transformer not fitted.")

        return X[self.selected_columns].copy()