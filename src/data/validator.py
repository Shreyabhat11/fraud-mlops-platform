"""
Enterprise Dataset Validator

Validates datasets before they enter the ML pipeline.

Author:
Enterprise Fraud Detection Platform
"""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


class DatasetValidator:
    """
    Enterprise dataset validator.
    """

    def __init__(

        self,

        target_column: str = "isFraud",

        transaction_column: str = "TransactionID",

        missing_threshold: float = 0.90,

        cardinality_threshold: int = 100,

    ):

        self.target_column = target_column

        self.transaction_column = transaction_column

        self.missing_threshold = missing_threshold

        self.cardinality_threshold = cardinality_threshold

    # ---------------------------------------------------------
    # Main Entry
    # ---------------------------------------------------------

    def validate(self, df: pd.DataFrame):

        logger.info("Running dataset validation...")

        self.check_empty(df)

        self.check_duplicates(df)

        self.check_transaction_ids(df)

        self.check_target(df)

        self.check_infinite(df)

        self.check_constant_columns(df)

        self.check_missing(df)

        self.check_cardinality(df)

        self.summary(df)

        logger.info("Dataset validation completed successfully.")

    # ---------------------------------------------------------
    # Empty
    # ---------------------------------------------------------

    def check_empty(self, df):

        if df.empty:

            raise ValueError("Dataset is empty.")

    # ---------------------------------------------------------
    # Duplicate Rows
    # ---------------------------------------------------------

    def check_duplicates(self, df):

        duplicates = df.duplicated().sum()

        if duplicates > 0:

            logger.warning(

                f"{duplicates:,} duplicate rows detected."

            )

    # ---------------------------------------------------------
    # Transaction IDs
    # ---------------------------------------------------------

    def check_transaction_ids(self, df):

        if self.transaction_column not in df.columns:

            logger.warning(

                "TransactionID column missing."

            )

            return

        duplicate_ids = (

            df[self.transaction_column]

            .duplicated()

            .sum()

        )

        if duplicate_ids > 0:

            logger.warning(

                f"{duplicate_ids:,} duplicate TransactionIDs."

            )

    # ---------------------------------------------------------
    # Target
    # ---------------------------------------------------------

    def check_target(self, df):

        if self.target_column not in df.columns:

            logger.warning(

                "Target column missing. Assuming inference dataset."

            )

            return

        values = sorted(

            df[self.target_column]

            .dropna()

            .unique()

            .tolist()

        )

        if values != [0, 1]:

            raise ValueError(

                f"Target should be binary. Found {values}"

            )

        fraud_rate = df[self.target_column].mean()

        logger.info(

            f"Fraud Rate : {fraud_rate:.2%}"

        )

    # ---------------------------------------------------------
    # Infinite
    # ---------------------------------------------------------

    def check_infinite(self, df):

        numeric = df.select_dtypes(

            include=np.number

        )

        inf_count = np.isinf(

            numeric

        ).sum().sum()

        if inf_count > 0:

            raise ValueError(

                f"{inf_count:,} infinite values detected."

            )

    # ---------------------------------------------------------
    # Constant Columns
    # ---------------------------------------------------------

    def check_constant_columns(self, df):

        constant = [

            c

            for c in df.columns

            if df[c].nunique(dropna=False) <= 1

        ]

        if constant:

            logger.warning(

                f"{len(constant)} constant columns detected."

            )

    # ---------------------------------------------------------
    # Missing Values
    # ---------------------------------------------------------

    def check_missing(self, df):

        percent = (

            df

            .isna()

            .mean()

        )

        high_missing = percent[

            percent > self.missing_threshold

        ]

        if len(high_missing):

            logger.warning(

                f"{len(high_missing)} columns have "

                f">{self.missing_threshold:.0%} missing values."

            )

    # ---------------------------------------------------------
    # Cardinality
    # ---------------------------------------------------------

    def check_cardinality(self, df):

        categorical = df.select_dtypes(

            exclude=np.number

        )

        high = {}

        for column in categorical.columns:

            unique = categorical[column].nunique()

            if unique > self.cardinality_threshold:

                high[column] = unique

        if high:

            logger.info(

                "High-cardinality categorical columns:"

            )

            for col, value in high.items():

                logger.info(

                    f"{col:<25} {value}"

                )

    # ---------------------------------------------------------
    # Summary
    # ---------------------------------------------------------

    def summary(self, df):

        numerical = df.select_dtypes(

            include=np.number

        ).columns

        categorical = df.select_dtypes(

            exclude=np.number

        ).columns

        logger.info("-" * 60)

        logger.info("Validation Summary")

        logger.info("-" * 60)

        logger.info(

            f"Rows               : {len(df):,}"

        )

        logger.info(

            f"Columns            : {df.shape[1]}"

        )

        logger.info(

            f"Numerical Features : {len(numerical)}"

        )

        logger.info(

            f"Categorical Features : {len(categorical)}"

        )