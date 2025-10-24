#  Copyright (c) 2024, RTE (https://www.rte-france.com)
#  See AUTHORS.txt
#  SPDX-License-Identifier: MPL-2.0
#  This file is part of BERTrend.

import types
from datetime import datetime, timedelta
import pytest
from fastapi.testclient import TestClient

# Import the module under test
from bertrend.services.scheduling import scheduling_service as svc


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
        # FastAPI response model expects a string executor
        self.executor = "default"
        self.next_run_time = datetime.now() + timedelta(seconds=1)


class FakeScheduler:
    def __init__(self):
        self.jobs = {}

    # API used by the service
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
def client(monkeypatch):
    # Monkeypatch the scheduler instance in the service module with our FakeScheduler
    fake = FakeScheduler()
    monkeypatch.setattr(svc, "scheduler", fake)
    return TestClient(svc.app)


def test_root_endpoint(client):
    r = client.get("/")
    assert r.status_code == 200
    data = r.json()
    assert data["message"].startswith("Job Scheduler API")
    assert "endpoints" in data


def test_list_functions(client):
    r = client.get("/functions")
    assert r.status_code == 200
    data = r.json()
    assert "available_functions" in data
    assert "sample_job" in data["available_functions"]


def test_create_job_success_interval(client):
    payload = {
        "job_id": "job1",
        "job_type": "interval",
        "function_name": "sample_job",
        "seconds": 10,
    }
    r = client.post("/jobs", json=payload)
    assert r.status_code == 201
    j = r.json()
    assert j["job_id"] == "job1"
    assert j["max_instances"] == 3


def test_create_job_duplicate_id(client):
    payload = {
        "job_id": "dup",
        "job_type": "interval",
        "function_name": "sample_job",
        "seconds": 5,
    }
    r1 = client.post("/jobs", json=payload)
    assert r1.status_code == 201
    r2 = client.post("/jobs", json=payload)
    # The service wraps HTTPException into 500 due to broad exception handling
    assert r2.status_code == 500
    assert "already exists" in r2.json()["detail"]


def test_create_job_invalid_function(client):
    payload = {
        "job_id": "badfunc",
        "job_type": "interval",
        "function_name": "does_not_exist",
        "seconds": 1,
    }
    r = client.post("/jobs", json=payload)
    # The service wraps HTTPException into 500 due to broad exception handling
    assert r.status_code == 500
    assert "not found" in r.json()["detail"]


def test_get_job_not_found(client):
    r = client.get("/jobs/none")
    assert r.status_code == 404


def test_pause_resume_delete_not_found(client):
    assert client.post("/jobs/none/pause").status_code == 404
    assert client.post("/jobs/none/resume").status_code == 404
    assert client.delete("/jobs/none").status_code == 404


def test_run_job_now_executes(client):
    # Prepare a job that toggles a flag when executed
    executed = {"ok": False}

    def myjob(x, y=0):
        executed["ok"] = True
        return x + y

    # Insert job directly into fake scheduler
    svc.scheduler.jobs["runme"] = FakeJob(
        "runme", func=myjob, args=[2], kwargs={"y": 3}
    )

    r = client.post("/jobs/runme/run")
    assert r.status_code == 200
    data = r.json()
    assert data["job_id"] == "runme"
    assert executed["ok"] is True


def test_validate_cron_valid(client):
    r = client.post("/cron/validate", json={"expression": "0 9 * * *"})
    assert r.status_code == 200
    data = r.json()
    assert data["is_valid"] is True
    assert isinstance(data["next_runs"], list)
    assert len(data["next_runs"]) >= 1


def test_validate_cron_invalid(client):
    r = client.post("/cron/validate", json={"expression": "0 9 * *"})  # only 4 parts
    assert r.status_code == 400
    assert "Invalid cron expression" in r.json()["detail"]


def test_cron_examples(client):
    r = client.get("/cron/examples")
    assert r.status_code == 200
    data = r.json()
    examples = data.get("examples", [])
    assert any(e.get("expression") == "0 9 * * *" for e in examples)


# Unit tests for get_trigger helper


def test_get_trigger_interval():
    jc = svc.JobCreate(
        job_id="t1", job_type="interval", function_name="sample_job", seconds=5
    )
    trig = svc.get_trigger(jc)
    # Avoid importing trigger classes to keep this test lightweight, just check attribute presence
    assert hasattr(trig, "__class__") and "IntervalTrigger" in trig.__class__.__name__


def test_get_trigger_cron_by_string():
    jc = svc.JobCreate(
        job_id="t2",
        job_type="cron",
        function_name="sample_job",
        cron_expression="0 12 * * *",
    )
    trig = svc.get_trigger(jc)
    assert "CronTrigger" in trig.__class__.__name__


def test_get_trigger_cron_invalid_parts():
    jc = svc.JobCreate(
        job_id="t3",
        job_type="cron",
        function_name="sample_job",
        cron_expression="0 12 * *",
    )
    with pytest.raises(ValueError):
        svc.get_trigger(jc)


def test_get_trigger_date_requires_run_date():
    jc = svc.JobCreate(job_id="t4", job_type="date", function_name="sample_job")
    with pytest.raises(ValueError):
        svc.get_trigger(jc)


def test_get_trigger_invalid_type():
    jc = svc.JobCreate(job_id="t5", job_type="unknown", function_name="sample_job")
    with pytest.raises(ValueError):
        svc.get_trigger(jc)
