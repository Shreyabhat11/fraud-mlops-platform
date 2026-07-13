"""
Enterprise Bundle Manager

Stores everything required for inference.

Author:
Enterprise Fraud Detection Platform
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import joblib


class ModelBundle:

    VERSION = "1.0.0"

    def __init__(

        self,

        bundle_directory,

    ):

        self.bundle = Path(

            bundle_directory

        )

        self.bundle.mkdir(

            parents=True,

            exist_ok=True,

        )

    # -----------------------------------------------------

    @staticmethod
    def checksum(path):

        sha = hashlib.sha256()

        with open(path, "rb") as f:

            while True:

                block = f.read(8192)

                if not block:

                    break

                sha.update(block)

        return sha.hexdigest()

    # -----------------------------------------------------

    def save(

        self,

        model,

        pipeline,

        metrics,

        threshold,

        feature_names,

    ):

        model_file = self.bundle / "model.pkl"

        pipeline_file = self.bundle / "pipeline.pkl"

        joblib.dump(

            model,

            model_file,

        )

        joblib.dump(

            pipeline,

            pipeline_file,

        )

        manifest = {

            "version": self.VERSION,

            "threshold": threshold,

            "feature_count": len(

                feature_names

            ),

        }

        with open(

            self.bundle /

            "manifest.json",

            "w",

            encoding="utf-8",

        ) as f:

            json.dump(

                manifest,

                f,

                indent=4,

            )

        with open(

            self.bundle /

            "metrics.json",

            "w",

            encoding="utf-8",

        ) as f:

            json.dump(

                metrics,

                f,

                indent=4,

            )

        with open(

            self.bundle /

            "feature_names.json",

            "w",

            encoding="utf-8",

        ) as f:

            json.dump(

                feature_names,

                f,

                indent=4,

            )

        checksums = {

            "model.pkl":

                self.checksum(

                    model_file

                ),

            "pipeline.pkl":

                self.checksum(

                    pipeline_file

                ),

        }

        with open(

            self.bundle /

            "checksums.json",

            "w",

            encoding="utf-8",

        ) as f:

            json.dump(

                checksums,

                f,

                indent=4,

            )

    # -----------------------------------------------------

    def load(self):

        model = joblib.load(

            self.bundle /

            "model.pkl"

        )

        pipeline = joblib.load(

            self.bundle /

            "pipeline.pkl"

        )

        with open(

            self.bundle /

            "manifest.json",

            "r",

        ) as f:

            manifest = json.load(f)

        return {

            "model": model,

            "pipeline": pipeline,

            "manifest": manifest,

        }