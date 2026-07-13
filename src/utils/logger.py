"""
Logging Utility

Creates a project-wide logger that logs both to the console
and to a file.

Every module should obtain its logger using:

    logger = get_logger(__name__)
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

from src.utils.config import get_config

# Keeps track of whether logging has already been initialized
_INITIALIZED = False


def setup_logger(log_file: Optional[Path] = None) -> None:
    """
    Configure the root logger.

    Parameters
    ----------
    log_file : Path, optional
        Path to the log file.
    """

    global _INITIALIZED

    if _INITIALIZED:
        return

    config = get_config()

    log_dir = Path(config.paths.logs)
    log_dir.mkdir(parents=True, exist_ok=True)

    if log_file is None:
        log_file = log_dir / "training.log"

    formatter = logging.Formatter(
        fmt="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    root_logger = logging.getLogger()

    root_logger.setLevel(logging.INFO)

    # Console handler
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)

    # File handler
    file_handler = logging.FileHandler(
        log_file,
        encoding="utf-8"
    )

    file_handler.setFormatter(formatter)

    root_logger.addHandler(console_handler)
    root_logger.addHandler(file_handler)

    _INITIALIZED = True


def get_logger(name: str) -> logging.Logger:
    """
    Returns a configured logger.

    Parameters
    ----------
    name : str
        Usually __name__

    Returns
    -------
    logging.Logger
    """

    setup_logger()

    return logging.getLogger(name)