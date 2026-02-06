#  Copyright (c) 2024, RTE (https://www.rte-france.com)
#  See AUTHORS.txt
#  SPDX-License-Identifier: MPL-2.0
#  This file is part of BERTrend.

from __future__ import annotations

import hashlib
import os
from contextlib import contextmanager
from datetime import datetime
from pathlib import Path
from urllib.parse import urljoin

import requests
from loguru import logger
from requests.adapters import HTTPAdapter

from bertrend import load_toml_config
from bertrend.bertrend_apps.common.scheduler_utils import SchedulerUtils

# Note: .env is already loaded in bertrend/__init__.py

# Base URL for the scheduling service (FastAPI). Can be overridden via env var.
SCHEDULER_SERVICE_URL = os.getenv("SCHEDULER_SERVICE_URL", "http://localhost:8882/")

BERTREND_APPS_SERVICE_URL = os.getenv(
    "BERTREND_APPS_SERVICE_URL", "http://localhost:8881/"
)

DEFAULT_COMMAND_TIMEOUT = 60  # in secs

# commands
SCRAPE_FEED_COMMAND = "/scrape-feed"
TRAIN_NEW_MODEL = "/train-new-model"
GENERATE_NEWSLETTERS = "/generate-newsletters"
GENERATE_REPORT = "/generate-report"

# default timeouts for BERTrend jobs
BERTREND_COMMANDS_TIMEOUTS = {
    SCRAPE_FEED_COMMAND: 1200,
    TRAIN_NEW_MODEL: 1800,
    GENERATE_NEWSLETTERS: 600,
    GENERATE_REPORT: 600,
}

_REQUEST_TIMEOUT = float(os.getenv("SCHEDULER_HTTP_TIMEOUT", "5"))


@contextmanager
def _get_session():
    """Context manager that creates and properly closes a session."""
    session = requests.Session()
    # Configure connection pooling limits
    adapter = HTTPAdapter(
        pool_connections=10, pool_maxsize=20, max_retries=0, pool_block=False
    )
    session.mount("http://", adapter)
    session.mount("https://", adapter)
    try:
        yield session
    finally:
        # Ensure all connections are properly closed
        try:
            # Clear adapters which closes the connection pools
            for adapter in session.adapters.values():
                adapter.close()
            session.adapters.clear()
        except Exception as e:
            logger.error(f"Error closing session adapters: {e}")


def _request(method: str, path: str, *, json: dict | None = None):
    """Make HTTP request and ensure response body is consumed."""
    url = urljoin(SCHEDULER_SERVICE_URL, path)
    with _get_session() as session:
        try:
            resp = session.request(
                method.upper(), url, json=json, timeout=_REQUEST_TIMEOUT
            )
            # Force connection release by accessing content
            # This ensures the response is fully read
            _ = resp.content
            return resp
        except requests.exceptions.RequestException as e:
            logger.error(f"Request failed for {method} {url}: {e}")
            raise


def _job_id_from_string(s: str) -> str:
    """Generate a deterministic, short job id from an arbitrary string."""
    h = hashlib.sha1(s.encode("utf-8")).hexdigest()[:10]
    return f"job_{h}"


def _list_jobs() -> list[dict]:
    """Return list of jobs using the HTTP API."""
    try:
        r = _request("GET", "/jobs")
        if r.status_code != 200:
            logger.error(f"Failed to list jobs: {r.status_code} {r.text}")
            return []
        return r.json()
    except Exception as e:
        logger.error(f"Error listing jobs: {e}")
        return []


class APSchedulerUtils(SchedulerUtils):
    def __init__(self):
        super().__init__()
        # Health check removed from __init__ to avoid blocking worker startup
        # The service connection will be validated on first actual use

    @staticmethod
    def find_jobs(patterns: dict, match_all: bool = True) -> list[str]:
        """Find jobs matching the provided patterns and return their ids."""
        payload = {"match_all": match_all, "kwargs_patterns": patterns}
        try:
            r = _request("POST", "/jobs/find", json=payload)
            if r.status_code not in (200, 201):
                logger.error(f"Failed to find jobs: {r.status_code} {r.text}")
                raise Exception(f"Failed to find jobs: {r.status_code} {r.text}")
            # Process results
            results_d = r.json()
            if results_d["matches_found"] == 0:
                logger.trace("No jobs found matching the provided patterns")
                return []
            return [job["job_id"] for job in results_d["jobs"]]
        except requests.exceptions.RequestException as e:
            logger.error(f"Request error finding jobs: {e}")
            raise

    @staticmethod
    def find_jobs_description(patterns: dict, match_all: bool = True):
        """Find jobs matching the provided patterns and return their full description."""
        payload = {"match_all": match_all, "kwargs_patterns": patterns}
        try:
            r = _request("POST", "/jobs/find", json=payload)
            if r.status_code not in (200, 201):
                logger.error(f"Failed to find jobs: {r.status_code} {r.text}")
                raise Exception(f"Failed to find jobs: {r.status_code} {r.text}")
            # Process results
            results_d = r.json()
            if results_d["matches_found"] == 0:
                logger.trace("No jobs found matching the provided patterns")
                return []
            return results_d["jobs"]
        except requests.exceptions.RequestException as e:
            logger.error(f"Request error finding jobs: {e}")
            raise

    @staticmethod
    def remove_jobs(job_ids: list[str]):
        """Remove jobs matching the provided ids from the scheduler service."""
        for job_id in job_ids:
            try:
                r = _request("DELETE", f"/jobs/{job_id}")
                if r.status_code != 200:
                    logger.error(
                        f"Failed to delete job {job_id}: {r.status_code} {r.text}"
                    )
                # Response body already consumed by _request()
            except Exception as e:
                logger.error(f"Error deleting job {job_id}: {e}")

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
                "timeout": BERTREND_COMMANDS_TIMEOUTS.get(
                    command, DEFAULT_COMMAND_TIMEOUT
                ),
            },
            "max_instances": 3,
            "coalesce": True,
            "replace_existing": True,
        }
        try:
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
        except Exception as e:
            logger.error(f"Error creating job: {e}")
            return False

    def schedule_scrapping(self, feed_cfg: Path, user: str | None = None):
        """Schedule data scrapping based on a feed configuration file using the service."""
        data_feed_cfg = load_toml_config(feed_cfg)
        schedule = data_feed_cfg["data-feed"]["update_frequency"]
        id = data_feed_cfg["data-feed"]["id"].removeprefix("feed_")
        command = SCRAPE_FEED_COMMAND
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
    ):
        """Schedule newsletter generation based on configuration using the service."""
        newsletter_cfg = load_toml_config(newsletter_cfg_path)
        schedule = newsletter_cfg["newsletter"]["update_frequency"]
        command = GENERATE_NEWSLETTERS
        command_kwargs = {
            "method": "POST",
            "json_data": {
                "newsletter_toml_path": newsletter_cfg_path.resolve().as_posix(),
                "data_feed_toml_path": data_feed_cfg_path.resolve().as_posix(),
            },
        }
        self.add_job_to_crontab(
            schedule=schedule, command=command, command_kwargs=command_kwargs
        )

    def schedule_training_for_user(self, schedule: str, model_id: str, user: str):
        """Schedule data scrapping on the basis of a feed configuration file"""
        command = TRAIN_NEW_MODEL
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
        command = GENERATE_REPORT
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

    def check_if_learning_active_for_user(self, model_id: str, user: str) -> bool:
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

    def get_next_scrapping(
        self, feed_id: str, user: str | None = None
    ) -> datetime | None:
        """Return the next scrapping date for the given feed_id and user"""
        try:
            jobs = self.find_jobs_description(
                patterns={
                    "url": ".*/scrape-feed.*",
                    "json_data": {"user": user, "model_id": feed_id},
                }
            )
            if len(jobs) == 0:
                return None
            return datetime.fromisoformat(jobs[0]["next_run_time"])
        except Exception as e:
            logger.error(
                f"Error occurred while checking date of next scrapping for {user},{feed_id}: {e}"
            )
            return None

    def get_next_learning(
        self, model_id: str, user: str | None = None
    ) -> datetime | None:
        """Return the next scrapping date for the given feed_id and user"""
        try:
            jobs = self.find_jobs_description(
                patterns={
                    "url": ".*/train-new-model.*",
                    "json_data": {"user": user, "model_id": model_id},
                }
            )
            if len(jobs) == 0:
                return None
            return datetime.fromisoformat(jobs[0]["next_run_time"])
        except Exception as e:
            logger.error(
                f"Error occurred while checking date of next learning for {user},{model_id}: {e}"
            )
            return None
