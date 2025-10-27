#  Copyright (c) 2024, RTE (https://www.rte-france.com)
#  See AUTHORS.txt
#  SPDX-License-Identifier: MPL-2.0
#  This file is part of BERTrend.
import os
import re
import subprocess
import sys
from pathlib import Path

from dotenv import load_dotenv
from loguru import logger

from bertrend import BEST_CUDA_DEVICE, BERTREND_LOG_PATH, load_toml_config
from bertrend_apps.common.scheduler_utils import SchedulerUtils

load_dotenv(override=True)


class CrontabSchedulerUtils(SchedulerUtils):
    def add_job_to_crontab(
        self,
        schedule: str,
        command: str,
        env_vars: str = None,
        command_kwargs: dict = None,
    ) -> bool:
        if env_vars is None:
            env_vars = ""
        """Add the specified job to the crontab."""
        logger.debug(f"Adding to crontab: {schedule} {command}")
        home = os.getenv("HOME")
        # Create crontab, add command - NB: we use the .bashrc to source all environment variables that may be required by the command
        cmd = f'(crontab -l; echo "{schedule} umask 002; source {home}/.bashrc; {env_vars} {command}" ) | crontab -'
        returned_value = subprocess.call(
            cmd, shell=True
        )  # returns the exit code in unix
        return returned_value == 0

    def _check_cron_job(self, pattern: str) -> bool:
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

    def _remove_from_crontab(self, pattern: str) -> bool:
        """Removes from the crontab the job matching the provided pattern (expressed as a regular expression)"""
        if not (self._check_cron_job(pattern)):
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

    def schedule_scrapping(self, feed_cfg: Path, user: str = None):
        """Schedule data scrapping on the basis of a feed configuration file"""
        data_feed_cfg = load_toml_config(feed_cfg)
        schedule = data_feed_cfg["data-feed"]["update_frequency"]
        id = data_feed_cfg["data-feed"]["id"]
        log_path = BERTREND_LOG_PATH if not user else BERTREND_LOG_PATH / "users" / user
        log_path.mkdir(parents=True, exist_ok=True)
        command = f"{sys.executable} -m bertrend_apps.data_provider scrape-feed {feed_cfg.resolve()} > {log_path}/cron_feed_{id}.log 2>&1"
        self.add_job_to_crontab(schedule, command, "")

    def schedule_newsletter(
        self,
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
        self.add_job_to_crontab(schedule, command, env_vars)

    def schedule_training_for_user(self, schedule: str, model_id: str, user: str):
        """Schedule data scrapping on the basis of a feed configuration file"""
        logpath = BERTREND_LOG_PATH / "users" / user
        logpath.mkdir(parents=True, exist_ok=True)
        command = (
            f"{sys.executable} -m bertrend_apps.prospective_demo.process_new_data train-new-model {user} {model_id} "
            f"> {logpath}/learning_{model_id}.log 2>&1"
        )
        env_vars = f"CUDA_VISIBLE_DEVICES={BEST_CUDA_DEVICE}"
        return self.add_job_to_crontab(schedule, command, env_vars)

    def schedule_report_generation_for_user(
        self, schedule: str, model_id: str, user: str, report_config: dict
    ) -> bool:
        """Schedule automated report generation based on model configuration"""
        auto_send = report_config.get("auto_send", False)
        recipients = report_config.get("email_recipients", [])

        if not auto_send:
            logger.info(f"auto_send is disabled for model {model_id}")
            return False

        if not recipients:
            logger.warning(f"No email recipients configured for model {model_id}")
            return False

        logpath = BERTREND_LOG_PATH / "users" / user
        logpath.mkdir(parents=True, exist_ok=True)

        command = (
            f"{sys.executable} -m bertrend_apps.prospective_demo.automated_report_generation {user} {model_id} "
            f"> {logpath}/report_{model_id}.log 2>&1"
        )

        return self.add_job_to_crontab(schedule, command, "")

    def remove_scrapping_for_user(self, feed_id: str, user: str | None = None):
        """Removes from the scheduler service the job matching the provided feed_id"""
        if user:
            return self._remove_from_crontab(
                rf"scrape-feed.*/users/{user}/{feed_id}_feed.toml"
            )
        else:
            return self._remove_from_crontab(rf"scrape-feed.*/{feed_id}_feed.toml")

    def remove_scheduled_training_for_user(self, model_id: str, user: str):
        """Removes from the crontab the training job matching the provided model_id"""
        if user:
            return self._remove_from_crontab(
                rf"process_new_data train-new-model {user} {model_id}"
            )
        return False

    def remove_scheduled_report_generation_for_user(
        self, model_id: str, user: str
    ) -> bool:
        """Removes from the crontab the report generation job matching the provided model_id"""
        if user:
            return self._remove_from_crontab(
                rf"automated_report_generation {user} {model_id}"
            )
        return False

    def check_if_scrapping_active_for_user(
        self, feed_id: str, user: str | None = None
    ) -> bool:
        """Checks if a given scrapping feed is active (registered with the service)."""
        if user:
            return self._check_cron_job(
                rf"scrape-feed.*/users/{user}/{feed_id}_feed.toml"
            )
        else:
            return self._check_cron_job(rf"scrape-feed.*/{feed_id}_feed.toml")

    def check_if_learning_active_for_user(self, model_id: str, user: str):
        """Checks if a given scrapping feed is active (registered in the crontab"""
        if user:
            return self._check_cron_job(
                rf"process_new_data train-new-model.*{user}.*{model_id}"
            )
        else:
            return False

    def check_if_report_generation_active_for_user(
        self, model_id: str, user: str
    ) -> bool:
        """Checks if automated report generation is active (registered in the crontab)"""
        if user:
            return self._check_cron_job(
                rf"automated_report_generation.*{user}.*{model_id}"
            )
        else:
            return False
