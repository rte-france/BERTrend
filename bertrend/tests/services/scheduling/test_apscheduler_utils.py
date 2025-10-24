#  Copyright (c) 2024, RTE (https://www.rte-france.com)
#  See AUTHORS.txt
#  SPDX-License-Identifier: MPL-2.0
#  This file is part of BERTrend.

from datetime import datetime, timedelta
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from bertrend.services.scheduling import scheduling_service as svc
from bertrend_apps.common import apscheduler_utils as su

scheduler = su.APSchedulerUtils()


class FakeJob:
    def __init__(
        self,
        job_id,
        func=None,
        name=None,
        trigger=None,
        args=None,
        kwargs=None,
        max_instances=3,
    ):
        self.id = job_id
        self.name = name or job_id
        self.func = func or (lambda *a, **k: None)
        self.trigger = trigger or "trigger"
        self.args = args or []
        self.kwargs = kwargs or {}
        self.max_instances = max_instances
        self.executor = "default"
        self.next_run_time = datetime.now() + timedelta(seconds=1)


class FakeScheduler:
    def __init__(self):
        self.jobs = {}

    def get_job(self, job_id):
        return self.jobs.get(job_id)

    def get_jobs(self):
        return list(self.jobs.values())

    def add_job(
        self,
        func,
        trigger,
        id,
        name,
        args,
        kwargs,
        max_instances,
        coalesce,
        replace_existing,
    ):
        if id in self.jobs:
            raise Exception("Job already exists")
        self.jobs[id] = FakeJob(
            id,
            func=func,
            name=name,
            trigger=trigger,
            args=args,
            kwargs=kwargs,
            max_instances=max_instances,
        )

    def reschedule_job(self, job_id, trigger):
        job = self.get_job(job_id)
        if not job:
            raise Exception("Job not found")
        job.trigger = trigger

    def modify_job(self, job_id, **changes):
        job = self.get_job(job_id)
        if not job:
            raise Exception("Job not found")
        for k, v in changes.items():
            setattr(job, k, v)

    def remove_job(self, job_id):
        if job_id not in self.jobs:
            raise Exception("Job not found")
        del self.jobs[job_id]

    def pause_job(self, job_id):
        if job_id not in self.jobs:
            raise Exception("Job not found")

    def resume_job(self, job_id):
        if job_id not in self.jobs:
            raise Exception("Job not found")


@pytest.fixture()
def client_and_http_adapter(monkeypatch):
    # Use a fake in-memory scheduler inside the service
    fake = FakeScheduler()
    monkeypatch.setattr(svc, "scheduler", fake)

    client = TestClient(svc.app)

    # Route scheduler_utils HTTP requests to the FastAPI TestClient
    def _adapter(method, url, **kwargs):
        from urllib.parse import urlsplit

        split = urlsplit(url)
        path = split.path
        if split.query:
            path = f"{path}?{split.query}"
        json = kwargs.get("json")
        return client.request(method, path, json=json)

    monkeypatch.setattr(
        su, "_session", type("S", (), {"request": staticmethod(_adapter)})()
    )
    monkeypatch.setenv("SCHEDULER_SERVICE_URL", "http://testserver/")
    return client


def _make_tmp_feed(
    tmp_path: Path,
    feed_id: str = "abc",
    cron: str = "0 9 * * *",
    user: str | None = None,
) -> Path:
    # Create directories reflecting potential user path usage
    base = tmp_path if not user else tmp_path / "users" / user
    base.mkdir(parents=True, exist_ok=True)
    p = base / f"{feed_id}_feed.toml"
    p.write_text(
        """
[data-feed]
update_frequency = "0 9 * * *"
id = "abc"
""".strip()
    )
    return p


def test_schedule_and_check_and_remove_scrapping_non_user(
    tmp_path, client_and_http_adapter
):
    feed_path = _make_tmp_feed(tmp_path, feed_id="feedX")

    # Initially not present
    assert scheduler.check_if_scrapping_active_for_user("feedX") is False

    # Schedule
    scheduler.schedule_scrapping(feed_path)

    # Now it should be present (pattern is based on command string with path)
    assert scheduler.check_if_scrapping_active_for_user("feedX") is True

    # Remove
    assert scheduler.remove_scrapping_for_user("feedX") is True
    assert scheduler.check_if_scrapping_active_for_user("feedX") is False


def test_schedule_and_check_and_remove_scrapping_user(
    tmp_path, client_and_http_adapter
):
    user = "alice"
    feed_path = _make_tmp_feed(tmp_path, feed_id="ufeed", user=user)

    assert scheduler.check_if_scrapping_active_for_user("ufeed", user=user) is False

    scheduler.schedule_scrapping(feed_path, user=user)

    assert scheduler.check_if_scrapping_active_for_user("ufeed", user=user) is True

    assert scheduler.remove_scrapping_for_user("ufeed", user=user) is True
    assert scheduler.check_if_scrapping_active_for_user("ufeed", user=user) is False


def test_schedule_newsletter(tmp_path, client_and_http_adapter):
    # Prepare minimal newsletter and feed config files
    newsletter_cfg = tmp_path / "n1.toml"
    newsletter_cfg.write_text(
        """
[newsletter]
update_frequency = "*/15 * * * *"
id = "nl1"
""".strip()
    )
    feed_cfg = tmp_path / "f1.toml"
    feed_cfg.write_text(
        """
[data-feed]
update_frequency = "0 9 * * *"
id = "f1"
""".strip()
    )

    # Schedule newsletter
    scheduler.schedule_newsletter(newsletter_cfg, feed_cfg, cuda_devices="0")

    # Check job exists by looking for file names in scheduled job payloads
    assert scheduler.check_cron_job(r"newsletters .*n1.toml.*f1.toml") is True
