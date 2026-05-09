"""Unified logging for console and file output.

Delegates to the standard logging module with a project-specific format.
"""

import logging
import sys
from pathlib import Path


def setup_logging(
    log_dir: str = "results/logs",
    experiment_name: str = "aml-gnn",
    level: int = logging.INFO,
) -> logging.Logger:
    """Create a logger that writes to both console and a timestamped file.

    Args:
        log_dir: Directory for log files.
        experiment_name: Name of the experiment (used in filename).
        level: Logging level.

    Returns:
        Configured root logger.
    """
    log_path = Path(log_dir)
    log_path.mkdir(parents=True, exist_ok=True)

    # Deduplicate handlers if called multiple times -----------------------
    root = logging.getLogger()
    root.handlers.clear()
    root.setLevel(level)

    fmt = logging.Formatter(
        "[%(asctime)s] %(levelname)-8s %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    # Console handler ----------------------------------------------------
    console = logging.StreamHandler(sys.stdout)
    console.setLevel(level)
    console.setFormatter(fmt)
    root.addHandler(console)

    # File handler -------------------------------------------------------
    file_handler = logging.FileHandler(log_path / f"{experiment_name}.log", mode="a")
    file_handler.setLevel(level)
    file_handler.setFormatter(fmt)
    root.addHandler(file_handler)

    return root


def get_logger(name: str) -> logging.Logger:
    """Return a child logger with the given name."""
    return logging.getLogger(name)
