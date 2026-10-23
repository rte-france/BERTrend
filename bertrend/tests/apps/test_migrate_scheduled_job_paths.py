#  Copyright (c) 2024-2026, RTE (https://www.rte-france.com)
#  See AUTHORS.txt
#  SPDX-License-Identifier: MPL-2.0
#  This file is part of BERTrend.
from unittest.mock import MagicMock, patch

import pytest
import typer
from typer.testing import CliRunner

from bertrend.bertrend_apps import migrate_scheduled_job_paths as migration
from bertrend.bertrend_apps.common import portable_paths

MODULE = "bertrend.bertrend_apps.migrate_scheduled_job_paths"


@pytest.fixture
def anchors(tmp_path, monkeypatch):
    base, package = tmp_path / "base", tmp_path / "bertrend"
    for f in (
        base / "config/users/u/abc_feed.toml",
        package / "bertrend_apps/config/feeds/rte_feed.toml",
        package / "bertrend_apps/config/newsletters/rte_newsletter_from_last.toml",
    ):
        f.parent.mkdir(parents=True, exist_ok=True)
        f.write_text("")
    monkeypatch.setattr(portable_paths, "BASE_PATH", base)
    monkeypatch.setattr(portable_paths, "PACKAGE_PATH", package)
    return base, package


def _job(job_id: str, json_data: dict) -> dict:
    kwargs = {"url": "http://bertrend:8881/x", "method": "POST", "json_data": json_data}
    return {"job_id": job_id, "name": f"/x|{json_data}", "kwargs": kwargs}


def test_plan_rewrites_legacy_paths_and_skips_portable_jobs(anchors, tmp_path):
    old_host = tmp_path / "old-host"
    jobs = [
        _job(
            "scrape",
            {
                "feed_cfg": f"{old_host}/nlp/bertrend/config/users/u/abc_feed.toml",
                "user": "u",
                "model_id": "abc",
            },
        ),
        _job(
            "newsletter",
            {
                "newsletter_toml_path": f"{old_host}/BERTrend/bertrend/bertrend_apps/config/newsletters/rte_newsletter_from_last.toml",
                "data_feed_toml_path": f"{old_host}/BERTrend/bertrend/bertrend_apps/config/feeds/rte_feed.toml",
            },
        ),
        _job("portable", {"feed_cfg": "config/users/u/abc_feed.toml"}),
        _job("training", {"user": "u", "model_id": "abc"}),
    ]

    plan = migration.plan_migration(jobs)

    assert [m.job_id for m in plan] == ["scrape", "newsletter"]
    scrape, newsletter = plan
    assert scrape.new_kwargs["json_data"] == {
        "feed_cfg": "config/users/u/abc_feed.toml",
        "user": "u",
        "model_id": "abc",
    }
    assert newsletter.new_kwargs["json_data"] == {
        "newsletter_toml_path": "bertrend_apps/config/newsletters/rte_newsletter_from_last.toml",
        "data_feed_toml_path": "bertrend_apps/config/feeds/rte_feed.toml",
    }
    # The job name embeds the kwargs: it follows them
    assert str(old_host) not in scrape.new_name
    assert "config/users/u/abc_feed.toml" in scrape.new_name
    # The listed job is left untouched
    assert jobs[0]["kwargs"]["json_data"]["feed_cfg"].startswith(str(old_host))


def test_legacy_base_dir_applies_when_files_are_unreachable(anchors, tmp_path):
    old_base = tmp_path / "old-host" / "bertrend"
    jobs = [_job("j", {"feed_cfg": f"{old_base}/config/users/v/not_here_feed.toml"})]

    assert migration.plan_migration(jobs)[0].changes[0].new is None
    assert (
        migration.plan_migration(jobs, [old_base])[0].changes[0].new
        == "config/users/v/not_here_feed.toml"
    )


def test_unresolved_paths_are_left_unchanged(anchors, tmp_path):
    stored = f"{tmp_path}/old-host/x/unknown.toml"
    (m,) = migration.plan_migration([_job("j", {"feed_cfg": stored})])

    assert m.applicable_changes == []
    assert m.new_kwargs["json_data"]["feed_cfg"] == stored


def _response(status_code: int, payload=None):
    r = MagicMock(status_code=status_code, text="")
    r.json.return_value = payload
    return r


def test_apply_checks_the_stored_job(anchors, tmp_path):
    job = _job("j", {"feed_cfg": f"{tmp_path}/old/config/users/u/abc_feed.toml"})
    (m,) = migration.plan_migration([job])
    stored = {**job, "kwargs": m.new_kwargs}

    # Older scheduler versions store the update but answer the PUT with an error
    with patch(
        f"{MODULE}._request", side_effect=[_response(400), _response(200, stored)]
    ) as req:
        assert migration.apply_migration(m) is True

    put = req.call_args_list[0]
    assert put.args == ("PUT", "/jobs/j")
    assert put.kwargs["json"] == {"kwargs": m.new_kwargs, "job_name": m.new_name}

    with patch(f"{MODULE}._request", side_effect=[_response(200), _response(200, job)]):
        assert migration.apply_migration(m) is False


def _run(args: list[str]):
    app = typer.Typer()
    app.command()(migration.main)
    return CliRunner().invoke(app, args)


def test_dry_run_changes_nothing(anchors, tmp_path):
    job = _job("j", {"feed_cfg": f"{tmp_path}/old/config/users/u/abc_feed.toml"})
    with (
        patch(f"{MODULE}.list_jobs", return_value=[job]),
        patch(f"{MODULE}.apply_migration") as apply,
    ):
        result = _run([])

    assert result.exit_code == 0
    assert "-> config/users/u/abc_feed.toml" in result.output
    assert "Dry run" in result.output
    apply.assert_not_called()


def test_apply_updates_jobs_and_reports_failures(anchors, tmp_path):
    job = _job("j", {"feed_cfg": f"{tmp_path}/old/config/users/u/abc_feed.toml"})
    with (
        patch(f"{MODULE}.list_jobs", return_value=[job]),
        patch(f"{MODULE}.apply_migration", return_value=False) as apply,
    ):
        result = _run(["--apply"])

    apply.assert_called_once()
    assert result.exit_code == 1
