#  Copyright (c) 2024-2026, RTE (https://www.rte-france.com)
#  See AUTHORS.txt
#  SPDX-License-Identifier: MPL-2.0
#  This file is part of BERTrend.
"""Rewrite the config paths stored in scheduled jobs into their portable form.

Jobs registered before config paths were made portable (see
``bertrend_apps.common.portable_paths``) store absolute paths, valid only in the
environment that registered them. This rewrites them relative to
BERTREND_BASE_DIR or to the bertrend package, through the scheduler service
API (SCHEDULER_SERVICE_URL). Job ids, triggers and next run times are kept.

Dry run by default: it prints the planned changes and modifies nothing.

    python -m bertrend.bertrend_apps.migrate_scheduled_job_paths [--apply] \\
        [--legacy-base-dir /old/bertrend/base/dir ...]
"""

import copy
from dataclasses import dataclass, field
from pathlib import Path

import typer

from bertrend.bertrend_apps.common.apscheduler_utils import (
    SCHEDULER_SERVICE_URL,
    _request,
)
from bertrend.bertrend_apps.common.portable_paths import (
    remap_legacy_path,
    to_portable_path,
)

# Keys of a job's json_data holding config paths
PATH_KEYS = ("feed_cfg", "newsletter_toml_path", "data_feed_toml_path")


@dataclass
class PathChange:
    key: str
    old: str
    # None when no portable form could be determined: the path is left unchanged
    new: str | None


@dataclass
class JobMigration:
    job_id: str
    name: str
    kwargs: dict
    changes: list[PathChange] = field(default_factory=list)

    @property
    def applicable_changes(self) -> list[PathChange]:
        return [c for c in self.changes if c.new is not None]

    @property
    def new_kwargs(self) -> dict:
        kwargs = copy.deepcopy(self.kwargs)
        for change in self.applicable_changes:
            kwargs["json_data"][change.key] = change.new
        return kwargs

    @property
    def new_name(self) -> str:
        # Job names embed the command kwargs (see APSchedulerUtils.add_job_to_crontab)
        name = self.name
        for change in self.applicable_changes:
            name = name.replace(change.old, change.new)
        return name


def portable_form(path: str, legacy_base_dirs: list[Path] = ()) -> str | None:
    """Portable form of a path stored in a job, None if it cannot be determined.

    An absolute path is made relative to, in order: one of `legacy_base_dirs` (the
    BERTREND_BASE_DIR of the environment that registered the job, for when its
    files are not reachable from here), the local BERTREND_BASE_DIR or package, or
    the local anchor its trailing components designate (see `remap_legacy_path`).
    """
    p = Path(path)
    if not p.is_absolute():
        return path
    for legacy_base_dir in legacy_base_dirs:
        if p.is_relative_to(legacy_base_dir):
            return p.relative_to(legacy_base_dir).as_posix()
    portable = to_portable_path(p)
    if not Path(portable).is_absolute():
        return portable
    remapped = remap_legacy_path(p)
    return to_portable_path(remapped) if remapped else None


def plan_migration(
    jobs: list[dict], legacy_base_dirs: list[Path] = ()
) -> list[JobMigration]:
    """Return the jobs (as listed by the scheduler service) storing absolute paths."""
    migrations = []
    for job in jobs:
        kwargs = job.get("kwargs") or {}
        json_data = kwargs.get("json_data")
        if not isinstance(json_data, dict):
            continue
        migration = JobMigration(job_id=job["job_id"], name=job["name"], kwargs=kwargs)
        for key in PATH_KEYS:
            value = json_data.get(key)
            if isinstance(value, str) and Path(value).is_absolute():
                migration.changes.append(
                    PathChange(key, value, portable_form(value, legacy_base_dirs))
                )
        if migration.changes:
            migrations.append(migration)
    return migrations


def list_jobs() -> list[dict]:
    r = _request("GET", "/jobs")
    if r.status_code != 200:
        raise RuntimeError(f"Failed to list jobs: {r.status_code} {r.text}")
    return r.json()


def apply_migration(migration: JobMigration) -> bool:
    """Update the job through the service; True once its new kwargs are stored."""
    new_kwargs = migration.new_kwargs
    _request(
        "PUT",
        f"/jobs/{migration.job_id}",
        json={"kwargs": new_kwargs, "job_name": migration.new_name},
    )
    # Check the stored job rather than the PUT status: scheduler versions predating
    # this script store the update but answer it with an error.
    r = _request("GET", f"/jobs/{migration.job_id}")
    return r.status_code == 200 and r.json().get("kwargs") == new_kwargs


def main(
    apply: bool = typer.Option(
        False, "--apply", help="Update the jobs (default: only show the changes)."
    ),
    legacy_base_dir: list[Path] = typer.Option(
        [],
        help="BERTREND_BASE_DIR of the environment that registered the jobs, when "
        "their files are not reachable from here. Repeatable.",
    ),
):
    """Rewrite absolute config paths of scheduled jobs into portable ones."""
    migrations = plan_migration(list_jobs(), legacy_base_dir)
    if not migrations:
        typer.echo(f"No job stores absolute config paths ({SCHEDULER_SERVICE_URL}).")
        return

    failures = 0
    for migration in migrations:
        typer.echo(f"{migration.job_id}  {migration.name}")
        for change in migration.changes:
            new = (
                change.new
                or "UNCHANGED: no matching file under BERTREND_BASE_DIR or the package"
            )
            typer.echo(f"    {change.key}: {change.old}\n        -> {new}")
        if not apply or not migration.applicable_changes:
            continue
        if apply_migration(migration):
            typer.echo("    updated")
        else:
            failures += 1
            typer.echo("    FAILED to update", err=True)

    unresolved = sum(c.new is None for m in migrations for c in m.changes)
    if unresolved:
        typer.echo(
            f"{unresolved} path(s) could not be made portable; run from an environment "
            "where they exist, or pass --legacy-base-dir.",
            err=True,
        )
    if not apply:
        typer.echo(
            "Dry run: nothing was changed. Re-run with --apply to update the jobs."
        )
    if failures:
        raise typer.Exit(code=1)


if __name__ == "__main__":
    typer.run(main)
