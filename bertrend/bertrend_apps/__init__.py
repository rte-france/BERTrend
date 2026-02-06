#  Copyright (c) 2024, RTE (https://www.rte-france.com)
#  See AUTHORS.txt
#  SPDX-License-Identifier: MPL-2.0
#  This file is part of BERTrend.
import os

from loguru import logger

from bertrend.bertrend_apps.common.apscheduler_utils import APSchedulerUtils
from bertrend.bertrend_apps.common.crontab_utils import CrontabSchedulerUtils

# Lazy initialization to avoid fork issues with multiple workers
_scheduler_utils = None


def _get_scheduler_utils():
    """Get or create the scheduler utils instance for the current process."""
    global _scheduler_utils
    if _scheduler_utils is None:
        # Note: .env is already loaded in bertrend/__init__.py
        # Load scheduler type and URL from environment variables
        scheduler_type = os.getenv("SCHEDULER_SERVICE_TYPE", "crontab").lower()

        if scheduler_type == "apscheduler":
            _scheduler_utils = APSchedulerUtils()
        else:  # assume default="crontab":
            _scheduler_utils = CrontabSchedulerUtils()
            logger.info("Using CrontabScheduler")

    return _scheduler_utils


# Provide SCHEDULER_UTILS as a property-like access for backward compatibility
class _SchedulerUtilsProxy:
    """Proxy to provide lazy initialization while maintaining the same API."""

    def __getattr__(self, name):
        return getattr(_get_scheduler_utils(), name)


SCHEDULER_UTILS = _SchedulerUtilsProxy()
