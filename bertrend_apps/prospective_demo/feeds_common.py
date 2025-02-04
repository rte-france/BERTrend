#  Copyright (c) 2024, RTE (https://www.rte-france.com)
#  See AUTHORS.txt
#  SPDX-License-Identifier: MPL-2.0
#  This file is part of BERTrend.
from pathlib import Path

from loguru import logger

from bertrend import FEED_BASE_PATH, load_toml_config

# Feed config path
USER_FEEDS_BASE_PATH = FEED_BASE_PATH / "users"
USER_FEEDS_BASE_PATH.mkdir(parents=True, exist_ok=True)


def get_user_feed_path(user_name: str, feed_id: str):
    feed_path = USER_FEEDS_BASE_PATH / user_name / f"{feed_id}_feed.toml"
    return feed_path


def read_user_feeds(username: str) -> tuple[dict[str, dict], dict[str, Path]]:
    """Read user feed config files"""
    user_feed_dir = USER_FEEDS_BASE_PATH / username
    user_feed_dir.mkdir(parents=True, exist_ok=True)
    logger.debug(f"Reading user feeds from: {user_feed_dir}")
    matching_files = user_feed_dir.rglob("*_feed.toml")

    user_feeds = {}
    feed_files = {}
    for f in matching_files:
        feed_id = f.name.split("_feed.toml")[0]
        user_feeds[feed_id] = load_toml_config(f)
        feed_files[feed_id] = f

    return user_feeds, feed_files


def get_all_files_for_feed(user_feeds: dict[str, dict], feed_id: str) -> list[Path]:
    """Returns the paths of all files associated to a feed for the current user."""
    feed_base_dir = user_feeds[feed_id]["data-feed"]["feed_dir_path"]
    list_all_files = list(
        Path(FEED_BASE_PATH, feed_base_dir).glob(
            f"*{user_feeds[feed_id]['data-feed'].get('id')}*.jsonl*"
        )
    )
    return list_all_files
