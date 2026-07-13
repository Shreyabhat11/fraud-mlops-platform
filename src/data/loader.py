"""
Dataset Loader

Loads datasets, validates them and generates metadata
required for experiment tracking.

Author:
Enterprise Fraud Detection Platform
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, Tuple
import hashlib
import logging

import pandas as pd

from src.utils.config import get_config

logger = logging.getLogger(__name__)


class DataLoader:
    """
    Production dataset loader.

    Responsibilities
    ----------------
    - Load CSV files
    - Validate file existence
    - Generate SHA256 fingerprint
    - Generate dataset metadata
    - Verify target column
    - Detect duplicate Transaction IDs
    """

    def __init__(self):

        self.config = get_config()

        self.target_column = self.config.training.target_column
        self.project_root = Path(__file__).resolve().parents[2]

    # ----------------------------------------------------
    # Private
    # ----------------------------------------------------

    @staticmethod
    def _calculate_sha256(file_path: Path) -> str:
        """
        Calculates SHA256 checksum.
        """

        sha = hashlib.sha256()

        with open(file_path, "rb") as f:

            while True:

                block = f.read(8192)

                if not block:
                    break

                sha.update(block)

        return sha.hexdigest()

    # ----------------------------------------------------
    # Load CSV
    # ----------------------------------------------------

    def load_csv(
        self,
        file_path: str | Path
    ) -> Tuple[pd.DataFrame, Dict]:

        file_path = Path(file_path)

        if not file_path.exists():

            raise FileNotFoundError(
                f"Dataset not found:\n{file_path}"
            )

        logger.info(f"Loading dataset: {file_path}")

        df = pd.read_csv(file_path)

        metadata = self.generate_metadata(
            df,
            file_path
        )

        self.validate(df)

        logger.info(
            f"Dataset loaded successfully "
            f"({df.shape[0]:,} rows × {df.shape[1]} columns)"
        )

        return df, metadata

    # ----------------------------------------------------
    # Metadata
    # ----------------------------------------------------

    def generate_metadata(
        self,
        df: pd.DataFrame,
        file_path: Path
    ) -> Dict:

        metadata = {

            "dataset_name": file_path.name,

            "dataset_path": str(file_path),

            "sha256": self._calculate_sha256(file_path),

            "rows": int(df.shape[0]),

            "columns": int(df.shape[1]),

            "memory_mb": round(
                df.memory_usage(deep=True).sum()
                / 1024
                / 1024,
                2,
            ),

            "duplicates": int(
                df.duplicated().sum()
            ),
        }

        if self.target_column in df.columns:

            target = df[self.target_column]

            metadata["target_distribution"] = {

                "negative": int((target == 0).sum()),

                "positive": int((target == 1).sum()),

                "positive_ratio": round(
                    target.mean(),
                    6
                ),
            }

        return metadata

    # ----------------------------------------------------
    # Validation
    # ----------------------------------------------------

    def validate(
        self,
        df: pd.DataFrame
    ):

        logger.info("Running dataset validation")

        if "TransactionID" not in df.columns:

            raise ValueError(
                "Missing TransactionID column."
            )

        duplicate_ids = df["TransactionID"].duplicated().sum()

        if duplicate_ids > 0:

            logger.warning(
                f"{duplicate_ids} duplicate TransactionIDs detected."
            )

        if self.target_column not in df.columns:

            logger.warning(
                "Target column not found. "
                "Inference mode assumed."
            )

        logger.info("Validation completed successfully.")

    # ----------------------------------------------------
    # Default Training Dataset
    # ----------------------------------------------------

    def load_training_dataset(self):

        processed_path = (
            Path(
                self.config.paths.processed_data
            )
            / "train_merged.csv"
        )

        return self.load_csv(processed_path)