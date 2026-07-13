"""
Enterprise Configuration Manager

Loads config.yaml into strongly typed dataclasses.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

import yaml


# ==========================================================
# General
# ==========================================================

@dataclass(frozen=True)
class ProjectConfig:
    name: str
    version: float


@dataclass(frozen=True)
class PathsConfig:
    raw_data: str
    processed_data: str
    reports: str
    artifacts: str
    models: str
    logs: str


# ==========================================================
# Training
# ==========================================================

@dataclass(frozen=True)
class TrainingConfig:
    random_state: int
    validation_size: float
    split_strategy: str
    shuffle: bool
    target_column: str
    transaction_time_column: str


# ==========================================================
# Models
# ==========================================================

@dataclass(frozen=True)
class RandomForestConfig:
    enabled: bool
    n_estimators: int
    max_depth: int
    n_jobs: int


@dataclass(frozen=True)
class XGBoostConfig:
    enabled: bool
    n_estimators: int
    learning_rate: float
    max_depth: int
    subsample: float
    colsample_bytree: float
    tree_method: str
    random_state: int


@dataclass(frozen=True)
class LightGBMConfig:
    enabled: bool
    n_estimators: int
    learning_rate: float
    num_leaves: int
    random_state: int


@dataclass(frozen=True)
class CatBoostConfig:
    enabled: bool
    iterations: int
    learning_rate: float
    depth: int
    verbose: bool


@dataclass(frozen=True)
class ModelsConfig:
    random_forest: RandomForestConfig
    xgboost: XGBoostConfig
    lightgbm: LightGBMConfig
    catboost: CatBoostConfig


# ==========================================================
# MLflow
# ==========================================================

@dataclass(frozen=True)
class MLflowConfig:
    experiment_name: str
    tracking_uri: str


# ==========================================================
# Monitoring
# ==========================================================

@dataclass(frozen=True)
class MonitoringConfig:
    drift_threshold: float


# ==========================================================
# Artifacts
# ==========================================================

@dataclass(frozen=True)
class ArtifactsConfig:
    root_dir: str
    keep_last: int
    save_config: bool
    save_environment: bool
    save_git_commit: bool
    save_metadata: bool


# ==========================================================
# App
# ==========================================================

@dataclass(frozen=True)
class AppConfig:

    project: ProjectConfig
    paths: PathsConfig
    training: TrainingConfig
    models: ModelsConfig
    mlflow: MLflowConfig
    monitoring: MonitoringConfig
    artifacts: ArtifactsConfig


# ==========================================================
# Config Loader
# ==========================================================

class ConfigManager:

    def __init__(self):

        root = Path(__file__).resolve().parents[2]

        self.config_path = root / "config" / "config.yaml"

    def load(self) -> AppConfig:

        with open(self.config_path, "r", encoding="utf-8") as f:
            cfg = yaml.safe_load(f)

        return AppConfig(

            project=ProjectConfig(**cfg["project"]),

            paths=PathsConfig(**cfg["paths"]),

            training=TrainingConfig(**cfg["training"]),

            models=ModelsConfig(

                random_forest=RandomForestConfig(
                    **cfg["models"]["random_forest"]
                ),

                xgboost=XGBoostConfig(
                    **cfg["models"]["xgboost"]
                ),

                lightgbm=LightGBMConfig(
                    **cfg["models"]["lightgbm"]
                ),

                catboost=CatBoostConfig(
                    **cfg["models"]["catboost"]
                ),

            ),

            mlflow=MLflowConfig(
                **cfg["mlflow"]
            ),

            monitoring=MonitoringConfig(
                **cfg["monitoring"]
            ),

            artifacts=ArtifactsConfig(
                **cfg["artifacts"]
            ),

        )


@lru_cache(maxsize=1)
def get_config():

    return ConfigManager().load()