# -*- coding: utf-8 -*-
"""
Description: This module contains function to generate a simple logger
Author: SO, AB
Project: prism-1
"""

import logging
import os
from datetime import datetime

def get_simple_logger(app_name: str, log_level: str, nom_experience:str) -> logging.Logger:
    """
    Initializes and returns a logger with a specified logging level.
    Logs are written both to the console and to a file named <app_name>.log.

    Args:
        app_name (str): Name of the application or the logger.
        log_level (str): Desired log level as a string.

    Returns:
        logging.Logger: Configured logger object.
    """
    valid_levels = ("debug", "info", "warning", "error", "critical")
    if log_level.lower() not in valid_levels:
        raise ValueError(
            f'Invalid log level "{log_level}". '
            f'Log level must be one of {valid_levels}.'
        )

    log_level = log_level.upper()
    logger = logging.getLogger(app_name)
    logger.setLevel(log_level)

    # Remove all handlers if they already exist (avoid duplicate logs)
    if logger.hasHandlers():
        logger.handlers.clear()

    formatter = logging.Formatter("%(asctime)s %(levelname)s:%(message)s")

    # File handler with timestamped subfolder
    timestamp = datetime.now().strftime("%Y%m%d_%H%M")
    log_dir = os.path.join("logs", "exp_"+str(timestamp)+str(nom_experience))
    os.makedirs(log_dir, exist_ok=True)
    file_handler = logging.FileHandler(os.path.join(log_dir, f"{app_name}.log"))
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    # Console handler
    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(formatter)
    logger.addHandler(stream_handler)

    return logger