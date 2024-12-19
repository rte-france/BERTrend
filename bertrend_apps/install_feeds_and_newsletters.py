#  Copyright (c) 2024, RTE (https://www.rte-france.com)
#  See AUTHORS.txt
#  SPDX-License-Identifier: MPL-2.0
#  This file is part of BERTrend.
from pathlib import Path
from loguru import logger

from bertrend_apps.common.crontab_utils import schedule_scrapping, schedule_newsletter

CONFIG_PATH = Path(__file__).parent / "config"
CONFIG_FEEDS_PATH = CONFIG_PATH / "feeds"
CONFIGS_NEWSLETTERS_PATH = CONFIG_PATH / "newsletters"

if __name__ == "__main__":
    # Install feed crontabs
    logger.info("*** Installing feeds crontabs ***")
    for f in CONFIG_FEEDS_PATH.iterdir():
        logger.info(f"Installing crontab for {f}")
        schedule_scrapping(Path(f))

    # Install newsletters crontabs
    logger.info("*** Installing newsletters crontabs ***")
    for f in CONFIGS_NEWSLETTERS_PATH.iterdir():
        logger.info(f"Installing crontab for {f.stem}")
        associated_feed = CONFIG_FEEDS_PATH / (
            f.stem.split("_newsletter")[0] + "_feed.toml"
        )
        logger.debug(f"Associated feed: {associated_feed}")
        schedule_newsletter(Path(f), associated_feed)
