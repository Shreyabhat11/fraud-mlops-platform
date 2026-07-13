"""
Enterprise Feature Pipeline

- sklearn compatible
- Serializable
- Metadata collection
- Train/Inference consistency

Author:
Enterprise Fraud Detection Platform
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import joblib
import pandas as pd
from sklearn.pipeline import Pipeline

from src.features.encoding import CategoricalEncoder
from src.features.engineering import FraudFeatureGenerator
from src.features.missing import MissingValueTransformer
from src.features.selection import FeatureSelector
from src.utils.logger import get_logger

logger = get_logger(__name__)


class FeaturePipeline:

    def __init__(self):

        self.pipeline = Pipeline(
            steps=[
                ("missing", MissingValueTransformer()),
                ("engineering", FraudFeatureGenerator()),
                ("encoding", CategoricalEncoder()),
                ("selection", FeatureSelector()),
            ]
        )

        self.metadata_: dict[str, Any] = {}

    # --------------------------------------------------

    def fit(self, X: pd.DataFrame, y=None):

        logger.info("Fitting feature pipeline...")

        self.pipeline.fit(X, y)

        self.feature_names_ = self.pipeline.transform(X).columns.tolist()

        self.metadata_ = {}

        for name, transformer in self.pipeline.named_steps.items():

            if hasattr(transformer, "get_metadata"):

                self.metadata_[name] = transformer.get_metadata()

        logger.info("Feature pipeline fitted successfully.")

        return self

    # --------------------------------------------------

    def transform(self, X: pd.DataFrame):

        return self.pipeline.transform(X)

    # --------------------------------------------------

    def fit_transform(self, X, y=None):

        return self.pipeline.fit_transform(X, y)

    # --------------------------------------------------

    def save(self, path):

        path = Path(path)

        path.mkdir(parents=True, exist_ok=True)

        joblib.dump(
            self.pipeline,
            path / "feature_pipeline.pkl",
        )

        with open(
            path / "feature_metadata.json",
            "w",
            encoding="utf-8",
        ) as f:

            json.dump(
                self.metadata_,
                f,
                indent=4,
            )

        with open(
            path / "feature_names.json",
            "w",
            encoding="utf-8",
        ) as f:

            json.dump(
                self.feature_names_,
                f,
                indent=4,
            )

    # --------------------------------------------------

    @classmethod
    def load(cls, path):

        obj = cls()

        obj.pipeline = joblib.load(
            Path(path) / "feature_pipeline.pkl"
        )

        return obj

    # --------------------------------------------------

    @property
    def feature_names(self):

        return self.feature_names_