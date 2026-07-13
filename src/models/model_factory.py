"""
Enterprise Model Factory

Creates configured models.

Author:
Enterprise Fraud Detection Platform
"""

from __future__ import annotations

from dataclasses import dataclass

from catboost import CatBoostClassifier
from lightgbm import LGBMClassifier
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier

from src.utils.config import get_config


@dataclass(slots=True)
class ModelSpec:

    name: str

    estimator: object


class ModelFactory:

    def __init__(self):

        self.cfg = get_config().models

    # ---------------------------------------------------

    def create(self):

        models = []

        rf = self.cfg.random_forest

        if rf.enabled:

            models.append(

                ModelSpec(

                    "RandomForest",

                    RandomForestClassifier(

                        n_estimators=rf.n_estimators,

                        max_depth=rf.max_depth,

                        class_weight="balanced_subsample",

                        n_jobs=rf.n_jobs,

                        random_state=42,

                    ),

                )

            )

        xgb = self.cfg.xgboost

        if xgb.enabled:

            models.append(

                ModelSpec(

                    "XGBoost",

                    XGBClassifier(

                        objective="binary:logistic",

                        eval_metric="auc",

                        n_estimators=xgb.n_estimators,

                        learning_rate=xgb.learning_rate,

                        max_depth=xgb.max_depth,

                        subsample=xgb.subsample,

                        colsample_bytree=xgb.colsample_bytree,

                        tree_method=xgb.tree_method,

                        random_state=xgb.random_state,

                        n_jobs=-1,

                    ),

                )

            )

        lgb = self.cfg.lightgbm

        if lgb.enabled:

            models.append(

                ModelSpec(

                    "LightGBM",

                    LGBMClassifier(

                        n_estimators=lgb.n_estimators,

                        learning_rate=lgb.learning_rate,

                        num_leaves=lgb.num_leaves,

                        random_state=lgb.random_state,

                        n_jobs=-1,

                        verbosity=-1,

                    ),

                )

            )

        cat = self.cfg.catboost

        if cat.enabled:

            models.append(

                ModelSpec(

                    "CatBoost",

                    CatBoostClassifier(

                        iterations=cat.iterations,

                        learning_rate=cat.learning_rate,

                        depth=cat.depth,

                        verbose=cat.verbose,

                    ),

                )

            )

        # ---------- Additional Enterprise Models ----------

        models.append(

            ModelSpec(

                "ExtraTrees",

                ExtraTreesClassifier(

                    n_estimators=300,

                    class_weight="balanced",

                    n_jobs=-1,

                    random_state=42,

                ),

            )

        )

        models.append(

            ModelSpec(

                "HistGradientBoosting",

                HistGradientBoostingClassifier(

                    max_iter=300,

                    learning_rate=0.05,

                    random_state=42,

                ),

            )

        )

        return models