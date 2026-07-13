"""
Enterprise Base Transformer

Compatible with:
- sklearn Pipeline
- joblib
- MLflow
- GridSearchCV
- Optuna

Author:
Enterprise Fraud Detection Platform
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

import pandas as pd
from sklearn.base import BaseEstimator
from sklearn.base import TransformerMixin


class BaseTransformer(

    BaseEstimator,

    TransformerMixin,

    ABC,

):

    """
    Base class for every transformer.
    """

    def __init__(self):

        self.metadata_: dict[str, Any] = {}

    @abstractmethod
    def fit(

        self,

        X: pd.DataFrame,

        y=None,

    ):

        ...

    @abstractmethod
    def transform(

        self,

        X: pd.DataFrame,

    ) -> pd.DataFrame:

        ...

    # -----------------------------------------------------

    def get_metadata(self):

        return self.metadata_

    # -----------------------------------------------------

    def get_feature_names_out(

        self,

        input_features=None,

    ):

        if hasattr(self, "feature_names_out_"):

            return self.feature_names_out_

        return input_features

    # -----------------------------------------------------

    def __repr__(self):

        return self.__class__.__name__