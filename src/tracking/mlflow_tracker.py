"""
Enterprise MLflow Tracker

Handles experiment tracking, model logging,
artifact logging and model registry.

Author:
Enterprise Fraud Detection Platform
"""

from __future__ import annotations

import json
import os
import platform
import subprocess
from pathlib import Path
from typing import Any

import mlflow
import mlflow.sklearn
import mlflow.lightgbm
import mlflow.xgboost
import mlflow.catboost

from lightgbm import LGBMClassifier
from xgboost import XGBClassifier
from catboost import CatBoostClassifier

from pathlib import Path
class MLflowTracker:
    def __init__(
        self,
        experiment_name: str,
        tracking_uri: str = "sqlite:///mlflow.db",
    ):
        if tracking_uri.startswith("sqlite:///"):

            db_path = Path("mlflow.db").resolve()

            tracking_uri = f"sqlite:///{db_path.as_posix()}"
        mlflow.set_tracking_uri(tracking_uri)
        mlflow.set_experiment(experiment_name)

    # -----------------------------------------------------
    def start_run(
        self,
        run_name: str,
    ):
        mlflow.start_run(run_name=run_name)

    # -----------------------------------------------------

    def end_run(self):
        mlflow.end_run()

    # -----------------------------------------------------

    @staticmethod
    def log_params(
        params: dict[str, Any],
    ):
        for key, value in params.items():
            if isinstance(value, dict):
                for k, v in value.items():
                    mlflow.log_param(
                        f"{key}.{k}",
                        v,
                    )
            else:
                mlflow.log_param(
                    key,
                    value,
                )

    # -----------------------------------------------------

    @staticmethod
    def log_metrics(
        metrics: dict[str, float],
    ):
        mlflow.log_metrics(metrics)

    # -----------------------------------------------------

    @staticmethod
    def log_artifact(
        file_path,
    ):
        if Path(file_path).exists():
            mlflow.log_artifact(file_path)

    # -----------------------------------------------------

    @staticmethod
    def log_directory(
        directory,
    ):
        if Path(directory).exists():
            mlflow.log_artifacts(directory)

    # -----------------------------------------------------

    @staticmethod
    def log_model(
        model,
        name="best_model",
    ):
        """
        Log models using their native MLflow flavor.

        This avoids MLflow 3.x serialization issues and
        preserves framework-specific metadata.
        """

        if isinstance(model, LGBMClassifier):

            mlflow.lightgbm.log_model(
                lgb_model=model,
                name=name,
            )

        elif isinstance(model, XGBClassifier):

            mlflow.xgboost.log_model(
                xgb_model=model,
                name=name,
            )

        elif isinstance(model, CatBoostClassifier):

            mlflow.catboost.log_model(
                cb_model=model,
                name=name,
            )

        else:

            mlflow.sklearn.log_model(
                sk_model=model,
                name=name,
            )

    # -----------------------------------------------------

    @staticmethod
    def log_tags():
        tags = {
            "python": platform.python_version(),
            "platform": platform.platform(),
        }
        try:
            commit = subprocess.check_output(
                ["git", "rev-parse", "HEAD"]
            ).decode().strip()
            tags["git_commit"] = commit
        except Exception:
            tags["git_commit"] = "unknown"
        mlflow.set_tags(tags)

    # -----------------------------------------------------
    @staticmethod
    def log_dataset_metadata(
        dataframe,
        dataset_name,
    ):
        metadata = {
            "dataset": dataset_name,
            "rows": dataframe.shape[0],
            "columns": dataframe.shape[1],
            "missing_values": int(
                dataframe.isna().sum().sum()
            ),
        }
        mlflow.log_dict(
            metadata,
            "dataset_metadata.json",
        )

    # -----------------------------------------------------

    @staticmethod
    def log_config(
        config,
    ):
        mlflow.log_dict(
            config,
            "config.json",
        )