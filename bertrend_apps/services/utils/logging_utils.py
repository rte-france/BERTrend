#  Copyright (c) 2024, RTE (https://www.rte-france.com)
#  See AUTHORS.txt
#  SPDX-License-Identifier: MPL-2.0
#  This file is part of BERTrend.
from datetime import datetime

from loguru import logger

from bertrend import BERTREND_LOG_PATH


def get_file_logger(id: str, user_name: str = "", model_id: str = "") -> int:
    """Create a unique log file for this call"""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = (
        BERTREND_LOG_PATH
        / user_name
        / model_id
        / f"{id}_{user_name}_{model_id}_{timestamp}.log"
    )
    log_file.parent.mkdir(parents=True, exist_ok=True)
    # Add a specific handler file for this execution
    logger_id = logger.add(
        log_file,
        level="DEBUG",
        format="<green>{time:YYYY-MM-DD HH:mm:ss.SSS}</green> | <level>{level: <8}</level> | <cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - <level>{message}</level>",
        colorize=True,
        retention="60 days",  # Remove automatically logs after x days
    )
    return logger_id
