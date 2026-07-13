"""
Enterprise Fraud Detection Training Pipeline

Author:
Enterprise Fraud Detection Platform

Responsibilities
----------------
1. Load Dataset
2. Train Feature Pipeline
3. Train Multiple Models
4. Evaluate Models
5. Generate Reports
6. Save Artifacts
7. Log MLflow Experiment
8. Bundle Best Model
"""

from __future__ import annotations

import json
import time
from datetime import datetime
from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split

from src.data.loader import DataLoader

from src.features.pipeline import FeaturePipeline

from src.models.bundle import ModelBundle
from src.models.evaluator import ModelEvaluator
from src.models.trainer import (
    ModelTrainer,
    TrainingContext,
)
from src.models.visualizer import ModelVisualizer

from src.tracking.mlflow_tracker import MLflowTracker

from src.utils.config import get_config
from src.utils.logger import get_logger


logger = get_logger(__name__)

ROOT = Path(__file__).resolve().parent

ARTIFACT_DIR = ROOT / "artifacts"

CONFIG = get_config()

# ==========================================================
# Helpers
# ==========================================================


def print_banner():

    print("\n" + "=" * 70)
    print("IEEE-CIS FRAUD DETECTION")
    print("Enterprise MLOps Platform")
    print("=" * 70 + "\n")


def ensure_directories():

    ARTIFACT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )


def dataframe_summary(df):

    logger.info(
        "Rows : %s",
        len(df),
    )

    logger.info(
        "Columns : %s",
        len(df.columns),
    )

    logger.info(
        "Memory : %.2f MB",
        df.memory_usage(deep=True).sum() / 1024**2,
    )

# ==========================================================
# Main
# ==========================================================


def main():

    total_start = time.perf_counter()

    print_banner()

    ensure_directories()

    logger.info("Loading dataset...")

    loader = DataLoader()

    df = loader.load_training_dataset()[0]

    dataframe_summary(df)

    cfg = get_config()

    tracker = MLflowTracker(
        experiment_name=cfg.mlflow.experiment_name,
        tracking_uri=cfg.mlflow.tracking_uri,
    )

    tracker.start_run(

        run_name=datetime.now().strftime(
            "%Y%m%d_%H%M%S"
        )

    )
    tracker.log_tags()
    tracker.log_dataset_metadata(
        df,
        "IEEE-CIS Fraud Detection",

    )

    target = CONFIG.training.target_column

    X = df.drop(columns=[target])

    y = df[target]
    
    X_train, X_valid, y_train, y_valid = train_test_split(
        X,
        y,
        test_size=CONFIG.training.validation_size,
        shuffle=CONFIG.training.shuffle,
        stratify=y,
        random_state=CONFIG.training.random_state,
    )
    logger.info(
        "Training rows : %d",
        len(X_train),
    )
    logger.info(
        "Validation rows : %d",
        len(X_valid),
    )
    target = CONFIG.training.target_column
    X = df.drop(columns=[target])
    y = df[target]

    X_train, X_valid, y_train, y_valid = train_test_split(
        X,
        y,
        test_size=CONFIG.training.validation_size,
        shuffle=CONFIG.training.shuffle,
        stratify=y,
        random_state=CONFIG.training.random_state,
    )
    logger.info(
        "Training rows : %d",
        len(X_train),
    )
    logger.info(
        "Validation rows : %d",
        len(X_valid),
    )
    logger.info("=" * 70)
    logger.info("Training Feature Pipeline")
    feature_pipeline = FeaturePipeline()
    X_train = feature_pipeline.fit_transform(
        X_train,
        y_train,
    )
    X_valid = feature_pipeline.transform(
        X_valid,
    )

    logger.info(
        "Final Feature Count : %d",
        X_train.shape[1],
    )

    context = TrainingContext(
        X_train=X_train,
        y_train=y_train,
        X_valid=X_valid,
        y_valid=y_valid,

    )

    # ==========================================================
    # Model Training
    # ==========================================================

    logger.info("=" * 70)
    logger.info("Training Models")

    trainer = ModelTrainer()

    model_results = trainer.train_all(context)

    logger.info(
        "%d models trained successfully.",
        len(model_results),
    )

    # ==========================================================
    # Model Evaluation
    # ==========================================================

    logger.info("=" * 70)
    logger.info("Evaluating Models")
    evaluator = ModelEvaluator()
    leaderboard = evaluator.evaluate(
        model_results,
        y_valid,
    )
    evaluator.save(
        ARTIFACT_DIR,
    )
    best_model = evaluator.best_model(
        model_results,
    )
    logger.info(
        "Best Model : %s",
        best_model.name,
    )

    print("\n")
    print("=" * 70)
    print("MODEL LEADERBOARD")
    print("=" * 70)
    print(leaderboard)
    print("=" * 70)
    logger.info("Logging metrics to MLflow...")
    tracker.log_metrics(
        best_model.metrics,
    )
    tracker.log_params(
        {
            "best_model": best_model.name,
            "training_rows": len(X_train),
            "validation_rows": len(X_valid),
            "feature_count": X_train.shape[1],
        }
    )

    # ==========================================================
    # Visualizations
    # ==========================================================

    logger.info("=" * 70)
    logger.info("Generating Visual Reports")
    visualizer = ModelVisualizer(
        ARTIFACT_DIR,
    )
    visualizer.roc_curve(
        best_model.model,
        X_valid,
        y_valid,
        best_model.name,
    )
    visualizer.precision_recall(
        best_model.model,
        X_valid,
        y_valid,
        best_model.name,
    )
    visualizer.confusion_matrix(
        best_model.model,
        X_valid,
        y_valid,
        best_model.name,
    )
    visualizer.feature_importance(
        best_model.feature_importance,
    )
    metrics_path = ARTIFACT_DIR / "metrics.json"
    with open(
        metrics_path,
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            best_model.metrics,
            file,
            indent=4,
        )
    report_path = (
        ARTIFACT_DIR /
        "classification_report.txt"
    )
    with open(
        report_path,
        "w",
        encoding="utf-8",
    ) as file:
        for metric, value in best_model.metrics.items():
            file.write(
                f"{metric:<25}: {value}\n"
            )
    logger.info(
        "Uploading artifacts to MLflow..."
    )
    tracker.log_directory(
        ARTIFACT_DIR,
    )
    logger.info(
        "Registering model in MLflow..."
    )
    tracker.log_model(
        best_model.model,
    )
    logger.info(
        "Registering model in MLflow..."
    )
    tracker.log_model(
        best_model.model,

    )
    print("=" * 60)
    print(type(best_model.model))
    print(best_model.model.__class__.__module__)
    print(best_model.model.__class__.__name__)
    print("=" * 60)

    # ==========================================================
    # Create Inference Bundle
    # ==========================================================

    logger.info("=" * 70)
    logger.info("Creating Deployment Bundle")
    bundle = ModelBundle(
        ARTIFACT_DIR / "bundle"
    )
    bundle.save(
        model=best_model.model,
        pipeline=feature_pipeline,
        metrics=best_model.metrics,
        threshold=best_model.threshold,
        feature_names=feature_pipeline.feature_names,
    )
    logger.info("Bundle Created Successfully")
    leaderboard.to_csv(
        ARTIFACT_DIR / "leaderboard.csv",
        index=False,
    )

    # ==========================================================
    # Training Summary
    # ==========================================================
    summary = {
        "best_model": best_model.name,
        "roc_auc": best_model.metrics["ROC_AUC"],
        "pr_auc": best_model.metrics["PR_AUC"],
        "training_time": best_model.training_time,
        "feature_count": len(
            feature_pipeline.feature_names
        ),
        "timestamp": datetime.now().isoformat(),
    }
    with open(
        ARTIFACT_DIR / "training_summary.json",
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            summary,
            file,
            indent=4,
        )
    # ==========================================================
    # Finish MLflow Run
    # ==========================================================
    tracker.end_run()
    total_time = round(
        time.perf_counter()
        - total_start,
        2,
    )
    logger.info("=" * 70)
    logger.info(
        "Training Completed Successfully"
    )
    logger.info(
        "Best Model : %s",
        best_model.name,
    )
    logger.info(
        "ROC-AUC : %.5f",
        best_model.metrics["ROC_AUC"],
    )
    logger.info(
        "Execution Time : %.2f sec",
        total_time,
    )
    logger.info("=" * 70)
    print()
    print("=" * 70)
    print("TRAINING COMPLETED")
    print("=" * 70)
    print(f"Best Model      : {best_model.name}")
    print(
        f"ROC-AUC         : "
        f"{best_model.metrics['ROC_AUC']:.5f}"
    )
    print(
        f"PR-AUC          : "
        f"{best_model.metrics['PR_AUC']:.5f}"
    )
    print(
        f"Training Time   : "
        f"{best_model.training_time:.2f} sec"
    )
    print(
        f"Total Runtime   : "
        f"{total_time:.2f} sec"
    )
    print()
    print("Artifacts Saved")
    print("--------------------------")
    print("✔ Model Bundle")
    print("✔ Feature Pipeline")
    print("✔ Leaderboard")
    print("✔ Metrics")
    print("✔ Visualizations")
    print("✔ MLflow Run")
    print("=" * 70)

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        logger.warning(
            "Training interrupted by user."
        )
    except Exception as error:
        logger.exception(
            "Training failed."
        )
        raise