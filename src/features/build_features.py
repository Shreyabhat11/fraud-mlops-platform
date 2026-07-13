from typing import Dict, List

from sklearn.base import BaseEstimator, TransformerMixin

import pandas as pd
import numpy as np

class FeatureBuilder(BaseEstimator,TransformerMixin):
    """
    Custom feature engineering transformer.

    Responsibilities:
    -----------------
    1. Drop columns with excessive missing values.
    2. Learn numerical and categorical columns.
    3. Learn frequency encoding for categorical features.
    4. Apply identical preprocessing during inference.
    """

    def __init__(self,missing_threshold=95):
        self.missing_threshold = missing_threshold

        self.columns_to_drop: List[str] = []
        self.numeric_columns: List[str] = []
        self.categorical_columns: List[str] = []

        self.frequency_maps: Dict[str, Dict] = {}
        self.numeric_medians: Dict[str, float] = {}
    
    def fit(self, X, y=None):

        X = X.copy()
        missing_percent = (
            X.isnull()
            .mean()
            *100
        )

        self.columns_to_drop = (missing_percent[missing_percent > self.missing_threshold].index.tolist())
        
        if "isFraud" in X.columns:
            X = X.drop(columns=["isFraud"])
        
        X = X.drop(
            columns=self.columns_to_drop,
            errors="ignore"
        )

        self.numeric_columns = X.select_dtypes(include=np.number).columns.tolist()
        self.categorical_columns = X.select_dtypes(include='object').columns.tolist()

        self.numeric_columns = [col for col in self.numeric_columns if col not in self.columns_to_drop]
        self.categorical_columns = [col for col in self.categorical_columns if col not in self.columns_to_drop]

        self.frequency_maps = {}

        for col in self.categorical_columns:
            self.frequency_maps[col] = X[col].value_counts(normalize=True, dropna=False).to_dict()

        self.numeric_medians = {}

        for col in self.numeric_columns:

            self.numeric_medians[col] = X[col].median()

        return self

    def transform(self, X): 
        X = X.copy()

        if "isFraud" in X.columns:
            X = X.drop(columns=["isFraud"])
            
        X = X.drop(columns=self.columns_to_drop, errors='ignore')
        for col in self.categorical_columns:
            if col in X.columns:
                X[col] = X[col].map(self.frequency_maps[col])

        X = X.fillna(0)

        return X
    
    
    def _create_features(self, X: pd.DataFrame):

        X = X.copy()

        # --------------------------------------------
        # Transaction Hour
        # --------------------------------------------

        if "TransactionDT" in X.columns:

            X["TransactionHour"] = (
                X["TransactionDT"] // 3600
            ) % 24

        # --------------------------------------------
        # Log Transaction Amount
        # --------------------------------------------

        if "TransactionAmt" in X.columns:

            X["TransactionAmtLog"] = np.log1p(
                X["TransactionAmt"]
            )

        # --------------------------------------------
        # Email Domain Match
        # --------------------------------------------

        if (
            "P_emaildomain" in X.columns
            and
            "R_emaildomain" in X.columns
        ):

            X["EmailDomainMatch"] = (
                X["P_emaildomain"]
                ==
                X["R_emaildomain"]
            ).astype(int)

        # --------------------------------------------
        # Missing Card Information
        # --------------------------------------------

        card_columns = [
            "card1",
            "card2",
            "card3",
            "card4",
            "card5",
            "card6"
        ]

        existing = [
            c
            for c in card_columns
            if c in X.columns
        ]

        if existing:

            X["CardMissingCount"] = (
                X[existing]
                .isnull()
                .sum(axis=1)
            )

        return X