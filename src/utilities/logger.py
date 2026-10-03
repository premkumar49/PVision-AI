"""
PVision AI: Logging Utility
Provides formatted logging output for terminal and file.
"""

import logging
import sys
from pathlib import Path
from src.utilities.config import config

def get_logger(name: str = "PVision_AI", log_file: str = "pvision_ai.log") -> logging.Logger:
    """Creates or retrieves a configured logger instance."""
    logger = logging.getLogger(name)
    if logger.handlers:
        return logger

    logger.setLevel(logging.INFO)
    formatter = logging.Formatter(
        "[%(asctime)s] [%(levelname)s] [%(name)s]: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S"
    )

    # Console Handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    # File Handler
    try:
        config.ensure_directories()
        log_path = config.OUTPUTS_DIR / log_file
        file_handler = logging.FileHandler(log_path, encoding="utf-8")
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
    except Exception:
        pass

    return logger
