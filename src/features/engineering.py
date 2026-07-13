"""
Fraud Specific Feature Engineering

Creates domain-specific features for IEEE-CIS.

Author:
Enterprise Fraud Detection Platform
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.features.base import BaseTransformer
from src.utils.logger import get_logger

logger = get_logger(__name__)


class FraudFeatureGenerator(BaseTransformer):

    def fit(self, X, y=None):

        self.metadata = {

            "feature_generator":

                "IEEE-CIS v2"

        }

        self.is_fitted = True

        return self

    def transform(self, X):

        if not self.is_fitted:

            raise RuntimeError(
                "Transformer not fitted."
            )

        X = X.copy()

        logger.info(
            "Generating fraud features..."
        )

        # -----------------------------------------
        # Transaction Amount
        # -----------------------------------------

        if "TransactionAmt" in X.columns:

            X["TransactionAmt_Log"] = np.log1p(

                X["TransactionAmt"]

            )

            X["TransactionAmt_Squared"] = (

                X["TransactionAmt"] ** 2

            )

        # -----------------------------------------
        # Missing Values
        # -----------------------------------------

        X["Missing_Count"] = X.isna().sum(axis=1)

        # -----------------------------------------
        # Email Match
        # -----------------------------------------

        if {

            "P_emaildomain",

            "R_emaildomain"

        }.issubset(X.columns):

            X["Email_Match"] = (

                X["P_emaildomain"]

                ==

                X["R_emaildomain"]

            ).astype(int)

        # -----------------------------------------
        # Card Features
        # -----------------------------------------

        cards = [

            "card1",

            "card2",

            "card3",

            "card5"

        ]

        available = [

            c for c in cards

            if c in X.columns

        ]

        if available:

            X["Card_Missing"] = (

                X[available]

                .isna()

                .sum(axis=1)

            )

        # -----------------------------------------
        # Device
        # -----------------------------------------

        if "DeviceType" in X.columns:

            X["Is_Mobile"] = (

                X["DeviceType"]

                .fillna("unknown")

                .str.lower()

                .eq("mobile")

                .astype(int)

            )

        # -----------------------------------------
        # Transaction Time
        # -----------------------------------------

        if "TransactionDT" in X.columns:

            seconds = X["TransactionDT"]

            X["Hour"] = (

                (seconds // 3600)

                % 24

            )

            X["Day"] = (

                (seconds // 86400)

                % 7

            )

            X["Is_Night"] = (

                X["Hour"]

                .between(0, 5)

            ).astype(int)

        logger.info(

            "Generated %d engineered features",

            X.shape[1]

        )

        return X