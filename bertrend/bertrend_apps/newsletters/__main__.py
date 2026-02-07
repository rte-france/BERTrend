#  Copyright (c) 2024-2026, RTE (https://www.rte-france.com)
#  See AUTHORS.txt
#  SPDX-License-Identifier: MPL-2.0
#  This file is part of BERTrend.
import os
from pathlib import Path

import typer

from bertrend.bertrend_apps import SCHEDULER_UTILS
from bertrend.bertrend_apps.newsletters.newsletter_generation import process_newsletter

# Ensures to write with +rw for both user and groups
os.umask(0o002)

if __name__ == "__main__":
    app = typer.Typer()

    @app.command("newsletters")
    def newsletter_from_feed(
        newsletter_toml_path: Path = typer.Argument(
            help="Path to newsletters toml config file"
        ),
        data_feed_toml_path: Path = typer.Argument(
            help="Path to data feed toml config file"
        ),
    ):
        """Creates a newsletter associated to a data feed."""
        process_newsletter(newsletter_toml_path, data_feed_toml_path)

    @app.command("schedule-newsletters")
    def automate_newsletter(
        newsletter_toml_cfg_path: Path = typer.Argument(
            help="Path to newsletters toml config file"
        ),
        data_feed_toml_cfg_path: Path = typer.Argument(
            help="Path to data feed toml config file"
        ),
    ):
        """Schedule data scrapping on the basis of a feed configuration file"""
        SCHEDULER_UTILS.schedule_newsletter(
            newsletter_toml_cfg_path, data_feed_toml_cfg_path
        )

    # Main app
    app()
