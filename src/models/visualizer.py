"""
Enterprise Visualization Module

Creates publication-quality plots for model evaluation.

Author:
Enterprise Fraud Detection Platform
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

from sklearn.metrics import (

    RocCurveDisplay,

    PrecisionRecallDisplay,

    ConfusionMatrixDisplay,

)

from src.utils.logger import get_logger

logger = get_logger(__name__)


class ModelVisualizer:

    def __init__(

        self,

        output_directory,

    ):

        self.output_directory = Path(

            output_directory

        )

        self.output_directory.mkdir(

            parents=True,

            exist_ok=True,

        )

    # -------------------------------------------------

    def roc_curve(

        self,

        model,

        X,

        y,

        name,

    ):

        plt.figure(figsize=(8, 6))

        RocCurveDisplay.from_estimator(

            model,

            X,

            y,

        )

        plt.title(

            f"ROC Curve - {name}"

        )

        plt.tight_layout()

        plt.savefig(

            self.output_directory /

            f"{name}_roc.png"

        )

        plt.close()

    # -------------------------------------------------

    def precision_recall(

        self,

        model,

        X,

        y,

        name,

    ):

        plt.figure(figsize=(8, 6))

        PrecisionRecallDisplay.from_estimator(

            model,

            X,

            y,

        )

        plt.title(

            f"Precision Recall - {name}"

        )

        plt.tight_layout()

        plt.savefig(

            self.output_directory /

            f"{name}_pr.png"

        )

        plt.close()

    # -------------------------------------------------

    def confusion_matrix(

        self,

        model,

        X,

        y,

        name,

    ):

        plt.figure(figsize=(6, 6))

        ConfusionMatrixDisplay.from_estimator(

            model,

            X,

            y,

        )

        plt.title(

            f"Confusion Matrix - {name}"

        )

        plt.tight_layout()

        plt.savefig(

            self.output_directory /

            f"{name}_cm.png"

        )

        plt.close()

    # -------------------------------------------------

    def feature_importance(

        self,

        importance,

        top_n=20,

    ):

        if importance is None:

            return

        data = (

            pd.Series(importance)

            .sort_values(

                ascending=False

            )

            .head(top_n)

        )

        plt.figure(

            figsize=(10, 7)

        )

        data[::-1].plot.barh()

        plt.title(

            "Top Feature Importance"

        )

        plt.tight_layout()

        plt.savefig(

            self.output_directory /

            "feature_importance.png"

        )

        plt.close()

        logger.info(

            "Visualization saved."

        )