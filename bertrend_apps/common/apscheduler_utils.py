#  Copyright (c) 2024, RTE (https://www.rte-france.com)
#  See AUTHORS.txt
#  SPDX-License-Identifier: MPL-2.0
#  This file is part of BERTrend.

from __future__ import annotations

import hashlib
import os
from pathlib import Path

from dotenv import load_dotenv
from loguru import logger

from bertrend import load_toml_config
import requests
from urllib.parse import urljoin

from bertrend_apps.common.scheduler_utils import SchedulerUtils

load_dotenv(override=True)

# Base URL for the scheduling service (FastAPI). Can be overridden via env var.
SCHEDULER_SERVICE_URL = os.getenv("SCHEDULER_SERVICE_URL", "http://localhost:8882/")

BERTREND_APPS_SERVICE_URL = os.getenv(
    "BERTREND_APPS_SERVICE_URL", "http://localhost:8881/"
)

# Single shared session for connection pooling
_session = requests.Session()
_REQUEST_TIMEOUT = float(os.getenv("SCHEDULER_HTTP_TIMEOUT", "5"))


def _request(method: str, path: str, *, json: dict | None = None):
    url = urljoin(SCHEDULER_SERVICE_URL, path)
    resp = _session.request(method.upper(), url, json=json, timeout=_REQUEST_TIMEOUT)
    return resp


def _job_id_from_string(s: str) -> str:
    """Generate a deterministic, short job id from an arbitrary string."""
    h = hashlib.sha1(s.encode("utf-8")).hexdigest()[:10]
    return f"job_{h}"


def _list_jobs() -> list[dict]:
    """Return list of jobs using the HTTP API."""
    r = _request("GET", "/jobs")
    if r.status_code != 200:
        logger.error(f"Failed to list jobs: {r.status_code} {r.text}")
        return []
    try:
        return r.json()
    except Exception:
        return []


class APSchedulerUtils(SchedulerUtils):

    def __init__(self):
        super().__init__()
        # check if service is available
        try:
            r = _request("GET", "/health")
            if r.status_code != 200:
                logger.warning(
                    f"Error from to scheduler service: {r.status_code} {r.text}."
                )
            else:
                logger.info(
                    f"Connected to scheduler service at {SCHEDULER_SERVICE_URL}."
                )
        except Exception:
            logger.error(
                f"Failed to connect to scheduler service at {SCHEDULER_SERVICE_URL}. Start it if you want to to use scheduled jobs."
            )

    @staticmethod
    def find_jobs(patterns: dict, match_all: bool = True) -> list[str]:
        """Find jobs matching the provided patterns and return their ids."""
        payload = {"match_all": match_all, "kwargs_patterns": patterns}
        r = _request("POST", "/jobs/find", json=payload)
        if not r.status_code in (200, 201):
            logger.error(f"Failed to find jobs: {r.status_code} {r.text}")
            raise Exception(f"Failed to find jobs: {r.status_code} {r.text}")
        # Process results
        results_d = r.json()
        if results_d["matches_found"] == 0:
            logger.trace("No jobs found matching the provided patterns")
            return []
        return [job["job_id"] for job in results_d["jobs"]]

    @staticmethod
    def remove_jobs(job_ids: list[str]):
        """Remove jobs matching the provided ids from the scheduler service."""
        for job_id in job_ids:
            r = _request("DELETE", f"/jobs/{job_id}")
            if r.status_code != 200:
                logger.error(f"Failed to delete job {job_id}: {r.status_code} {r.text}")

    def add_job_to_crontab(
        self,
        schedule: str,
        command: str,
        env_vars: str = None,
        command_kwargs: dict = None,
    ) -> bool:
        """Add the specified job to the scheduler service via HTTP.
        We preserve the original signature; internally we create a cron job calling
        the service's http_request with the command embedded as a message. To allow
        regex-based discovery like the legacy crontab, we embed the command text in
        the job name (which is searchable) and use a hash-based job_id (URL-safe).
        """
        if command_kwargs is None:
            command_kwargs = {}
        logger.debug(f"Scheduling via service (HTTP): {schedule} {env_vars} {command}")
        # Use hash for job_id (URL-safe), but put full command in name for regex checks
        job_id = _job_id_from_string(f"{command}|{schedule}|{command_kwargs}")
        job_name = f"{command}|{command_kwargs}"
        payload = {
            "job_id": job_id,
            "job_name": job_name,
            "job_type": "cron",
            "function_name": "http_request",  # curl-like job
            "cron_expression": schedule,
            "kwargs": {
                "url": urljoin(BERTREND_APPS_SERVICE_URL, command),
                "method": command_kwargs.get("method", "GET"),
                "json_data": command_kwargs.get("json_data", {}),
            },
            "max_instances": 3,
            "coalesce": True,
            "replace_existing": True,
        }
        r = _request("POST", "/jobs", json=payload)
        if r.status_code in (200, 201):
            return True
        # Consider duplicates as success
        try:
            detail = r.json().get("detail", "")
        except Exception:
            detail = r.text
        logger.error(f"Failed to create job: {r.status_code} {detail}")
        return False

    def schedule_scrapping(self, feed_cfg: Path, user: str | None = None):
        """Schedule data scrapping based on a feed configuration file using the service."""
        data_feed_cfg = load_toml_config(feed_cfg)
        schedule = data_feed_cfg["data-feed"]["update_frequency"]
        id = data_feed_cfg["data-feed"]["id"].removeprefix("feed_")
        command = "/scrape-feed"
        command_kwargs = {
            "method": "POST",
            "json_data": {
                "feed_cfg": feed_cfg.resolve().as_posix(),
                "user": user,
                "model_id": id,
            },
        }
        # Use schedule+command in job_id to keep determinism and allow pattern search
        self.add_job_to_crontab(
            schedule=schedule, command=command, command_kwargs=command_kwargs
        )

    def schedule_newsletter(
        self,
        newsletter_cfg_path: Path,
        data_feed_cfg_path: Path,
        cuda_devices: str = "0",
    ):
        """Schedule newsletter generation based on configuration using the service."""
        newsletter_cfg = load_toml_config(newsletter_cfg_path)
        schedule = newsletter_cfg["newsletter"]["update_frequency"]
        command = "/schedule-newsletters"
        command_kwargs = {
            "method": "POST",
            "json_data": {
                "newsletter_toml_cfg_path": newsletter_cfg_path.resolve().as_posix(),
                "data_feed_toml_cfg_path": data_feed_cfg_path.resolve().as_posix(),
                "cuda_devices": cuda_devices,
            },
        }
        self.add_job_to_crontab(
            schedule=schedule, command=command, command_kwargs=command_kwargs
        )

    def schedule_training_for_user(self, schedule: str, model_id: str, user: str):
        """Schedule data scrapping on the basis of a feed configuration file"""
        command = "/train-new-model"
        command_kwargs = {
            "method": "POST",
            "json_data": {"user": user, "model_id": model_id},
        }
        return self.add_job_to_crontab(
            schedule=schedule, command=command, command_kwargs=command_kwargs
        )

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
        command = "/generate-report"
        command_kwargs = {
            "method": "POST",
            "json_data": {
                "user": user,
                "model_id": model_id,
                "reference_date": None,
            },
        }
        return self.add_job_to_crontab(
            schedule=schedule, command=command, command_kwargs=command_kwargs
        )

    def remove_scrapping_for_user(self, feed_id: str, user: str | None = None):
        """Removes from the scheduler service the job matching the provided feed_id"""
        try:
            job_ids = self.find_jobs(
                patterns={
                    "url": ".*/scrape-feed.*",
                    "json_data": {"user": user, "model_id": feed_id},
                }
            )
            self.remove_jobs(job_ids)
            return True
        except Exception as e:
            logger.error(f"Error occurred while removing scrapping job: {e}")
            return False

    def remove_scheduled_training_for_user(self, model_id: str, user: str):
        """Removes from the crontab the training job matching the provided model_id"""
        try:
            job_ids = self.find_jobs(
                patterns={
                    "url": ".*/train-new-model.*",
                    "json_data": {"user": user, "model_id": model_id},
                }
            )
            self.remove_jobs(job_ids)
            return True
        except Exception as e:
            logger.error(f"Error occurred while removing scrapping job: {e}")
            return False

    def remove_scheduled_report_generation_for_user(
        self, model_id: str, user: str
    ) -> bool:
        """Removes from the crontab the report generation job matching the provided model_id"""
        try:
            job_ids = self.find_jobs(
                patterns={
                    "url": ".*/generate-report.*",
                    "json_data": {"user": user, "model_id": model_id},
                }
            )
            self.remove_jobs(job_ids)
            return True
        except Exception as e:
            logger.error(f"Error occurred while removing scrapping job: {e}")
            return False

    def check_if_scrapping_active_for_user(
        self, feed_id: str, user: str | None = None
    ) -> bool:
        """Checks if a given scrapping feed is active (registered with the service)."""
        try:
            job_ids = self.find_jobs(
                patterns={
                    "url": ".*/scrape-feed.*",
                    "json_data": {"user": user, "model_id": feed_id},
                }
            )
            return len(job_ids) > 0
        except Exception as e:
            logger.error(
                f"Error occurred while checking if scrapping is active for {user},{feed_id}: {e}"
            )
            return False

    def check_if_learning_active_for_user(self, model_id: str, user: str):
        """Checks if a given scrapping feed is active (registered in the crontab"""
        try:
            job_ids = self.find_jobs(
                patterns={
                    "url": ".*/train-new-model.*",
                    "json_data": {"user": user, "model_id": model_id},
                }
            )
            return len(job_ids) > 0
        except Exception as e:
            logger.error(
                f"Error occurred while checking if learning is active for {user},{model_id}: {e}"
            )
            return False

    def check_if_report_generation_active_for_user(
        self, model_id: str, user: str
    ) -> bool:
        """Checks if automated report generation is active (registered in the crontab)"""
        try:
            job_ids = self.find_jobs(
                patterns={
                    "url": ".*/generate-report.*",
                    "json_data": {"user": user, "model_id": model_id},
                }
            )
            return len(job_ids) > 0
        except Exception as e:
            logger.error(
                f"Error occurred while checking if reporting is active for {user},{model_id}: {e}"
            )
            return False
