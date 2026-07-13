"""
Dataset Fingerprinting Utility

Creates a reproducible fingerprint of a dataset for
versioning, experiment tracking, governance, and reproducibility.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import pandas as pd


class DatasetFingerprint:
    """
    Generates metadata and fingerprints for datasets.
    """

    def __init__(
        self,
        dataframe: pd.DataFrame,
        file_path: str | Path
    ):

        self.df = dataframe

        self.file_path = Path(file_path)

    # ---------------------------------------------------------
    # SHA256
    # ---------------------------------------------------------

    def sha256(self) -> str:

        sha = hashlib.sha256()

        with open(self.file_path, "rb") as file:

            while True:

                block = file.read(8192)

                if not block:
                    break

                sha.update(block)

        return sha.hexdigest()

    # ---------------------------------------------------------
    # Dataset Signature
    # ---------------------------------------------------------

    def dataset_signature(self) -> str:
        """
        Creates a fingerprint based on schema and shape.
        """

        signature = str(list(self.df.columns))
        signature += str(self.df.shape)

        return hashlib.sha256(
            signature.encode()
        ).hexdigest()

    # ---------------------------------------------------------
    # Schema
    # ---------------------------------------------------------

    def schema(self) -> dict:

        return {
            column: str(dtype)
            for column, dtype
            in self.df.dtypes.items()
        }

    # ---------------------------------------------------------
    # Missing Values
    # ---------------------------------------------------------

    def missing_values(self):

        return (
            self.df
            .isna()
            .sum()
            .sort_values(ascending=False)
            .to_dict()
        )

    # ---------------------------------------------------------
    # Missing Percentage
    # ---------------------------------------------------------

    def missing_percentage(self):

        percent = (
            self.df
            .isna()
            .mean()
            * 100
        )

        return (
            percent
            .round(2)
            .sort_values(ascending=False)
            .to_dict()
        )

    # ---------------------------------------------------------
    # Target Distribution
    # ---------------------------------------------------------

    def target_distribution(self):

        if "isFraud" not in self.df.columns:
            return {}

        target = self.df["isFraud"]

        return {

            "negative": int((target == 0).sum()),

            "positive": int((target == 1).sum()),

            "positive_ratio": round(
                target.mean(),
                6
            )
        }

    # ---------------------------------------------------------
    # Column Types
    # ---------------------------------------------------------

    def numerical_columns(self):

        return self.df.select_dtypes(
            include="number"
        ).columns.tolist()

    def categorical_columns(self):

        return self.df.select_dtypes(
            exclude="number"
        ).columns.tolist()

    # ---------------------------------------------------------
    # Unique Values
    # ---------------------------------------------------------

    def unique_counts(self):

        return self.df.nunique().to_dict()

    # ---------------------------------------------------------
    # Metadata
    # ---------------------------------------------------------

    def metadata(self) -> dict[str, Any]:

        metadata = {

            "dataset_name": self.file_path.name,

            "dataset_path": str(self.file_path),

            "sha256": self.sha256(),

            "dataset_signature": self.dataset_signature(),

            "rows": int(self.df.shape[0]),

            "columns": int(self.df.shape[1]),

            "memory_mb": round(

                self.df.memory_usage(deep=True).sum()

                / 1024

                / 1024,

                2,

            ),

            "file_size_mb": round(

                self.file_path.stat().st_size

                / 1024

                / 1024,

                2,

            ),

            "duplicate_rows": int(

                self.df.duplicated().sum()

            ),

            "duplicate_transaction_ids": int(

                self.df["TransactionID"]

                .duplicated()

                .sum()

            )

            if "TransactionID" in self.df.columns

            else 0,

            "schema": self.schema(),

            "missing_values": self.missing_values(),

            "missing_percentage": self.missing_percentage(),

            "numerical_columns": self.numerical_columns(),

            "categorical_columns": self.categorical_columns(),

            "unique_counts": self.unique_counts(),

            "target_distribution": self.target_distribution(),
        }

        return metadata

    # ---------------------------------------------------------
    # Save
    # ---------------------------------------------------------

    def save(
        self,
        output_directory: str | Path
    ):

        output_directory = Path(output_directory)

        output_directory.mkdir(
            parents=True,
            exist_ok=True
        )

        with open(

            output_directory /
            "dataset_fingerprint.json",

            "w",

            encoding="utf-8"

        ) as file:

            json.dump(

                self.metadata(),

                file,

                indent=4

            )