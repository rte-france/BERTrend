#  Copyright (c) 2024, RTE (https://www.rte-france.com)
#  See AUTHORS.txt
#  SPDX-License-Identifier: MPL-2.0
#  This file is part of BERTrend.
import locale
import os
import re
import subprocess
import sys
from pathlib import Path

from cron_descriptor import (
    Options,
    CasingTypeEnum,
    ExpressionDescriptor,
    DescriptionTypeEnum,
)
from dotenv import load_dotenv
from loguru import logger

from bertrend import BEST_CUDA_DEVICE, BERTREND_LOG_PATH, load_toml_config
from bertrend.demos.demos_utils.i18n import get_current_internationalization_language

load_dotenv(override=True)


def get_understandable_cron_description(cron_expression: str) -> str:
    """Returns a human understandable crontab description."""
    # Save current locale
    saved_locale = locale.setlocale(locale.LC_ALL)

    options = Options()
    options.casing_type = CasingTypeEnum.Sentence
    options.use_24hour_time_format = True

    locale_code = (
        "fr_FR.UTF-8"
        if get_current_internationalization_language() == "fr"
        else "en_US.UTF-8"
    )
    crontab_locale_code = (
        "fr_FR" if get_current_internationalization_language() == "fr" else "en"
    )

    options.locale_code = crontab_locale_code

    try:
        # Set temporary locale to specific locale
        locale.setlocale(locale.LC_ALL, locale_code)
        descriptor = ExpressionDescriptor(cron_expression, options)
        description = descriptor.get_description(DescriptionTypeEnum.FULL)

    finally:
        # Restore original locale
        locale.setlocale(locale.LC_ALL, saved_locale)

    return description


def add_job_to_crontab(schedule, command, env_vars="") -> bool:
    """Add the specified job to the crontab."""
    logger.debug(f"Adding to crontab: {schedule} {command}")
    home = os.getenv("HOME")
    # Create crontab, add command - NB: we use the .bashrc to source all environment variables that may be required by the command
    cmd = f'(crontab -l; echo "{schedule} umask 002; source {home}/.bashrc; {env_vars} {command}" ) | crontab -'
    returned_value = subprocess.call(cmd, shell=True)  # returns the exit code in unix
    return returned_value == 0


def check_cron_job(pattern: str) -> bool:
    """Check if a specific pattern (expressed as a regular expression) matches crontab entries."""
    try:
        # Run `crontab -l` and capture the output
        result = subprocess.run(
            ["crontab", "-l"], capture_output=True, text=True, check=True
        )

        # Search for the regex pattern in the crontab output
        if re.search(pattern, result.stdout):
            return True
        else:
            return False
    except subprocess.CalledProcessError:
        # If crontab fails (e.g., no crontab for the user), return False
        return False


def remove_from_crontab(pattern: str) -> bool:
    """Removes from the crontab the job matching the provided pattern (expressed as a regular expression)"""
    if not (check_cron_job(pattern)):
        logger.warning("No job matching the provided pattern")
        return False
    try:
        # Retrieve current crontab
        output = subprocess.check_output(
            f"crontab -l | grep -Ev '{pattern}' | crontab -", shell=True
        )
        return output == 0
    except subprocess.CalledProcessError:
        return False


def schedule_scrapping(feed_cfg: Path, user: str = None):
    """Schedule data scrapping on the basis of a feed configuration file"""
    data_feed_cfg = load_toml_config(feed_cfg)
    schedule = data_feed_cfg["data-feed"]["update_frequency"]
    id = data_feed_cfg["data-feed"]["id"]
    log_path = BERTREND_LOG_PATH if not user else BERTREND_LOG_PATH / "users" / user
    log_path.mkdir(parents=True, exist_ok=True)
    command = f"{sys.executable} -m bertrend_apps.data_provider scrape-feed {feed_cfg.resolve()} > {log_path}/cron_feed_{id}.log 2>&1"
    add_job_to_crontab(schedule, command, "")


def schedule_newsletter(
    newsletter_cfg_path: Path,
    data_feed_cfg_path: Path,
    cuda_devices: str = BEST_CUDA_DEVICE,
):
    """Schedule data scrapping on the basis of a feed configuration file"""
    newsletter_cfg = load_toml_config(newsletter_cfg_path)
    schedule = newsletter_cfg["newsletter"]["update_frequency"]
    id = newsletter_cfg["newsletter"]["id"]
    command = f"{sys.executable} -m bertrend_apps.newsletters newsletters {newsletter_cfg_path.resolve()} {data_feed_cfg_path.resolve()} > {BERTREND_LOG_PATH}/cron_newsletter_{id}.log 2>&1"
    env_vars = f"CUDA_VISIBLE_DEVICES={cuda_devices}"
    add_job_to_crontab(schedule, command, env_vars)


def check_if_scrapping_active_for_user(feed_id: str, user: str = None) -> bool:
    """Checks if a given scrapping feed is active (registered in the crontab"""
    if user:
        return check_cron_job(rf"scrape-feed.*/users/{user}/{feed_id}_feed.toml")
    else:
        return check_cron_job(rf"scrape-feed.*/{feed_id}_feed.toml")


def remove_scrapping_for_user(feed_id: str, user: str = None):
    """Removes from the crontab the job matching the provided feed_id"""
    if user:
        return remove_from_crontab(rf"scrape-feed.*/users/{user}/{feed_id}_feed.toml")
    else:
        return remove_from_crontab(rf"scrape-feed.*/{feed_id}_feed.toml")
