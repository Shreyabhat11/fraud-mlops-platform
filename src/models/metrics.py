"""
Enterprise Metrics Module

Computes every metric used in fraud detection.

Author:
Enterprise Fraud Detection Platform
"""

from __future__ import annotations

import numpy as np

from sklearn.metrics import (

    accuracy_score,

    balanced_accuracy_score,

    f1_score,

    matthews_corrcoef,

    precision_recall_curve,

    precision_score,

    recall_score,

    roc_auc_score,

    average_precision_score,

)


class MetricsCalculator:

    @staticmethod
    def best_threshold(

        y_true,

        probabilities,

    ):

        precision, recall, thresholds = (

            precision_recall_curve(

                y_true,

                probabilities,

            )

        )

        f1 = (

            2

            * precision[:-1]

            * recall[:-1]

        ) / (

            precision[:-1]

            + recall[:-1]

            + 1e-8

        )

        index = np.argmax(f1)

        return thresholds[index]

    @staticmethod
    def evaluate(

        y_true,

        probabilities,

    ):

        threshold = MetricsCalculator.best_threshold(

            y_true,

            probabilities,

        )

        prediction = (

            probabilities >= threshold

        ).astype(int)

        return {

            "ROC_AUC":

                roc_auc_score(

                    y_true,

                    probabilities,

                ),

            "PR_AUC":

                average_precision_score(

                    y_true,

                    probabilities,

                ),

            "Accuracy":

                accuracy_score(

                    y_true,

                    prediction,

                ),

            "Balanced_Accuracy":

                balanced_accuracy_score(

                    y_true,

                    prediction,

                ),

            "Precision":

                precision_score(

                    y_true,

                    prediction,

                ),

            "Recall":

                recall_score(

                    y_true,

                    prediction,

                ),

            "F1":

                f1_score(

                    y_true,

                    prediction,

                ),

            "MCC":

                matthews_corrcoef(

                    y_true,

                    prediction,

                ),

            "Threshold":

                float(threshold),

        }