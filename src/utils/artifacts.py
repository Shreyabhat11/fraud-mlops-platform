"""
Artifact Management Utility

Creates versioned experiment directories and stores all
artifacts generated during a training run.

Example
-------
artifacts/
    2026-07-11_20-14-53/
        config.yaml
        metadata.json
        training.log
        fraud_bundle.pkl
        metrics.json
        ...
"""

from __future__ import annotations

import json
import platform
import shutil
from datetime import datetime
from pathlib import Path
from typing import Any, Dict

from src.utils.config import get_config


class ArtifactManager:
    """
    Handles experiment directory creation and artifact storage.
    """

    def __init__(self):

        self.config = get_config()

        self.root = Path(
            self.config["artifacts"]["root_dir"]
        )

        self.root.mkdir(
            parents=True,
            exist_ok=True
        )

        self.run_dir = self._create_run_directory()

    # ----------------------------------------------------
    # Create Experiment Folder
    # ----------------------------------------------------

    def _create_run_directory(self) -> Path:

        timestamp = datetime.now().strftime(
            "%Y-%m-%d_%H-%M-%S"
        )

        run_dir = self.root / timestamp

        run_dir.mkdir(
            parents=True,
            exist_ok=True
        )

        return run_dir

    # ----------------------------------------------------
    # Public Getters
    # ----------------------------------------------------

    def get_run_directory(self) -> Path:

        return self.run_dir

    def models_dir(self) -> Path:

        path = self.run_dir / "models"

        path.mkdir(exist_ok=True)

        return path

    def reports_dir(self) -> Path:

        path = self.run_dir / "reports"

        path.mkdir(exist_ok=True)

        return path

    def logs_dir(self) -> Path:

        path = self.run_dir / "logs"

        path.mkdir(exist_ok=True)

        return path

    # ----------------------------------------------------
    # Save Config
    # ----------------------------------------------------

    def save_config(self):

        if not self.config["artifacts"]["save_config"]:
            return

        shutil.copy(

            "config/config.yaml",

            self.run_dir / "config.yaml"

        )

    # ----------------------------------------------------
    # Save Metadata
    # ----------------------------------------------------

    def save_metadata(

        self,

        metadata: Dict[str, Any]

    ):

        if not self.config["artifacts"]["save_metadata"]:
            return

        metadata["python_version"] = platform.python_version()

        metadata["platform"] = platform.platform()

        with open(

            self.run_dir / "metadata.json",

            "w",

            encoding="utf-8"

        ) as file:

            json.dump(

                metadata,

                file,

                indent=4

            )

    # ----------------------------------------------------
    # Cleanup
    # ----------------------------------------------------

    def cleanup_old_runs(self):

        keep_last = self.config["artifacts"]["keep_last"]

        runs = sorted(

            [

                p for p in self.root.iterdir()

                if p.is_dir()

            ]

        )

        if len(runs) <= keep_last:

            return

        remove = runs[:-keep_last]

        for directory in remove:

            shutil.rmtree(directory)

    # ----------------------------------------------------
    # Initialize
    # ----------------------------------------------------

    def initialize(

        self,

        metadata: Dict[str, Any]

    ):

        self.save_config()

        self.save_metadata(metadata)

        self.cleanup_old_runs()