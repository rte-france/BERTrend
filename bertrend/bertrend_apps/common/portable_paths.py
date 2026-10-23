#  Copyright (c) 2024-2026, RTE (https://www.rte-france.com)
#  See AUTHORS.txt
#  SPDX-License-Identifier: MPL-2.0
#  This file is part of BERTrend.
"""Config paths stored in scheduled jobs, independent of where BERTrend runs.

A scheduled job outlives the process that registered it and may be executed
somewhere else: a job registered on the host can be run by a worker in a
container, where BERTREND_BASE_DIR is mounted and the package installed at other
locations. Job payloads therefore store config paths relative to one of two
anchors, tried in this order:

- ``BASE_PATH`` (BERTREND_BASE_DIR), for user configs, e.g.
  ``config/users/<user>/<model_id>_feed.toml``;
- ``PACKAGE_PATH`` (the ``bertrend`` package), for the configs it ships, e.g.
  ``bertrend_apps/config/feeds/rte_feed.toml``.

Jobs registered before this scheme store absolute paths; they still resolve when
their file exists, and are otherwise remapped onto the anchors (see
``remap_legacy_path``). ``bertrend_apps.migrate_scheduled_job_paths`` rewrites them.
"""

from pathlib import Path

from loguru import logger

import bertrend
from bertrend import BASE_PATH

PACKAGE_PATH = Path(bertrend.__file__).resolve().parent

# Fewest trailing components of a legacy path matched against the anchors, so that
# a bare file name is never taken for an unrelated file that happens to share it.
MIN_REMAP_PARTS = 2


def _anchors() -> tuple[Path, ...]:
    return tuple(anchor.expanduser().resolve() for anchor in (BASE_PATH, PACKAGE_PATH))


def to_portable_path(path: str | Path) -> str:
    """Return the form of `path` to store in a scheduled job.

    Relative to the first anchor containing it; absolute (resolved against the
    current directory) when it is under none of them.
    """
    resolved = Path(path).expanduser().resolve()
    for anchor in _anchors():
        if resolved.is_relative_to(anchor):
            return resolved.relative_to(anchor).as_posix()
    return resolved.as_posix()


def remap_legacy_path(path: str | Path) -> Path | None:
    """Find the file a legacy absolute path designates under the local anchors.

    Tries the trailing components of `path`, longest first, under each anchor, so
    that ``/DSIA/nlp/bertrend/config/users/u/f_feed.toml`` is found as
    ``<BASE_PATH>/config/users/u/f_feed.toml`` and
    ``/home/x/BERTrend/bertrend/bertrend_apps/config/feeds/f.toml`` as
    ``<PACKAGE_PATH>/bertrend_apps/config/feeds/f.toml``.
    Returns None when no existing file matches.
    """
    parts = Path(path).parts[1:]  # drop the root
    for start in range(len(parts) - MIN_REMAP_PARTS + 1):
        suffix = Path(*parts[start:])
        for anchor in _anchors():
            candidate = anchor / suffix
            if candidate.is_file():
                return candidate
    return None


def resolve_portable_path(path: str | Path) -> Path:
    """Return the local file designated by a path stored in a scheduled job.

    - relative: under the first anchor where it exists, else as given (i.e.
      relative to the current directory, as for a path typed on the command line);
    - absolute: as given if it exists, else remapped onto the anchors (legacy jobs
      registered on another host or outside a container), else as given.
    """
    candidate = Path(path).expanduser()
    if not candidate.is_absolute():
        for anchor in _anchors():
            if (anchor / candidate).exists():
                return anchor / candidate
        return candidate
    if candidate.exists():
        return candidate
    remapped = remap_legacy_path(candidate)
    if remapped is None:
        return candidate
    logger.warning(
        f"Config path {candidate} does not exist here, using {remapped} instead. "
        "The job was registered with an absolute path from another environment; "
        "run `python -m bertrend.bertrend_apps.migrate_scheduled_job_paths` to rewrite it."
    )
    return remapped
