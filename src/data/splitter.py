"""
Dataset Splitter

Supports:
- Stratified train/validation split
- Chronological split (fraud-safe)
- Logging
- Split metadata

Author:
Enterprise Fraud Detection Platform
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict

import pandas as pd
from sklearn.model_selection import train_test_split

from src.utils.config import get_config
from src.utils.logger import get_logger

logger = get_logger(__name__)


@dataclass
class SplitResult:
    X_train: pd.DataFrame
    X_valid: pd.DataFrame
    y_train: pd.Series
    y_valid: pd.Series
    metadata: Dict


class DataSplitter:

    def __init__(self):

        self.config = get_config()

        training = self.config["training"]

        self.target = training["target_column"]
        self.validation_size = training["validation_size"]
        self.random_state = training["random_state"]
        self.strategy = training["split_strategy"]
        self.shuffle = training["shuffle"]
        self.time_column = training["transaction_time_column"]

    # -----------------------------------------------------
    # Public
    # -----------------------------------------------------

    def split(self, df: pd.DataFrame) -> SplitResult:

        if self.target not in df.columns:
            raise ValueError(
                f"Target column '{self.target}' not found."
            )

        if self.strategy == "chronological":
            return self._chronological_split(df)

        return self._stratified_split(df)

    # -----------------------------------------------------
    # Stratified
    # -----------------------------------------------------

    def _stratified_split(
        self,
        df: pd.DataFrame
    ) -> SplitResult:

        logger.info("Using stratified train/validation split.")

        X = df.drop(columns=self.target)

        y = df[self.target]

        X_train, X_valid, y_train, y_valid = train_test_split(

            X,

            y,

            test_size=self.validation_size,

            random_state=self.random_state,

            stratify=y,

            shuffle=self.shuffle,
        )

        metadata = self._metadata(
            X_train,
            X_valid,
            y_train,
            y_valid,
            "stratified"
        )

        return SplitResult(
            X_train,
            X_valid,
            y_train,
            y_valid,
            metadata
        )

    # -----------------------------------------------------
    # Chronological
    # -----------------------------------------------------

    def _chronological_split(
        self,
        df: pd.DataFrame
    ) -> SplitResult:

        if self.time_column not in df.columns:
            raise ValueError(
                f"{self.time_column} missing."
            )

        logger.info("Using chronological split.")

        df = df.sort_values(self.time_column)

        split_index = int(
            len(df) * (1 - self.validation_size)
        )

        train = df.iloc[:split_index]

        valid = df.iloc[split_index:]

        X_train = train.drop(columns=self.target)
        y_train = train[self.target]

        X_valid = valid.drop(columns=self.target)
        y_valid = valid[self.target]

        metadata = self._metadata(
            X_train,
            X_valid,
            y_train,
            y_valid,
            "chronological"
        )

        return SplitResult(
            X_train,
            X_valid,
            y_train,
            y_valid,
            metadata
        )

    # -----------------------------------------------------
    # Metadata
    # -----------------------------------------------------

    def _metadata(
        self,
        X_train,
        X_valid,
        y_train,
        y_valid,
        strategy
    ) -> Dict:

        metadata = {

            "split_strategy": strategy,

            "train_rows": len(X_train),

            "validation_rows": len(X_valid),

            "train_fraud_rate": round(
                float(y_train.mean()),
                6
            ),

            "validation_fraud_rate": round(
                float(y_valid.mean()),
                6
            )
        }

        logger.info("-" * 60)
        logger.info("Dataset Split Summary")
        logger.info("-" * 60)
        logger.info(f"Strategy           : {strategy}")
        logger.info(f"Training Rows      : {len(X_train):,}")
        logger.info(f"Validation Rows    : {len(X_valid):,}")
        logger.info(f"Training Fraud %   : {y_train.mean():.2%}")
        logger.info(f"Validation Fraud % : {y_valid.mean():.2%}")

        return metadata