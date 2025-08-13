#  Copyright (c) 2024, RTE (https://www.rte-france.com)
#  See AUTHORS.txt
#  SPDX-License-Identifier: MPL-2.0
#  This file is part of BERTrend.
import re
import time
from pathlib import Path

import pandas as pd
import streamlit as st
import toml
from loguru import logger

from bertrend.config.parameters import LANGUAGES
from bertrend.demos.demos_utils.i18n import translate
from bertrend.demos.demos_utils.icons import (
    INFO_ICON,
    ERROR_ICON,
    ADD_ICON,
    EDIT_ICON,
    DELETE_ICON,
    WARNING_ICON,
    TOGGLE_ON_ICON,
    TOGGLE_OFF_ICON,
)
from bertrend.demos.streamlit_components.clickable_df_component import clickable_df
from bertrend_apps.common.crontab_utils import (
    get_understandable_cron_description,
    check_if_scrapping_active_for_user,
    remove_scrapping_for_user,
    schedule_scrapping,
)
from bertrend_apps.data_provider import URL_PATTERN
from bertrend_apps.prospective_demo.feeds_common import (
    read_user_feeds,
)
from bertrend_apps.prospective_demo import CONFIG_FEEDS_BASE_PATH
from bertrend_apps.prospective_demo.models_info import (
    remove_scheduled_training_for_user,
)

# Default feed configs
DEFAULT_CRONTAB_EXPRESSION = "1 0 * * 1"
DEFAULT_ATOM_CRONTAB_EXPRESSION = "42 0,6,12,18 * * *"  # 4 times a day
DEFAULT_MAX_RESULTS = 25
DEFAULT_MAX_RESULTS_ARXIV = 1000
DEFAULT_NUMBER_OF_DAYS = 7
FEED_SOURCES = ["google", "atom", "arxiv"]


@st.dialog(translate("feed_config_dialog_title"))
def edit_feed_monitoring(config: dict | None = None):
    """Create or update a feed monitoring configuration."""

    evaluate_articles_quality = False

    chosen_id = st.text_input(
        translate("feed_id_label") + " :red[*]",
        help=translate("feed_id_help"),
        value=None if not config else config["id"],
    )

    provider = st.segmented_control(
        translate("feed_source_label"),
        selection_mode="single",
        options=FEED_SOURCES,
        default=FEED_SOURCES[0] if not config else config["provider"],
        help=translate("feed_source_help"),
    )
    if provider == "google" or provider == "arxiv":
        query = st.text_input(
            translate("feed_query_label") + " :red[*]",
            value="" if not config else config["query"],
            help=translate("feed_query_help"),
        )
        language = st.segmented_control(
            translate("feed_language_label"),
            selection_mode="single",
            options=LANGUAGES,
            default=LANGUAGES[0] if provider == "google" else LANGUAGES[1],
            format_func=lambda lang: translate(f"language_{lang.lower()}"),
            help=translate("feed_language_help"),
        )
        if "update_frequency" not in st.session_state:
            st.session_state.update_frequency = (
                DEFAULT_CRONTAB_EXPRESSION if not config else config["update_frequency"]
            )
        new_freq = st.text_input(
            translate("feed_frequency_label"),
            value=st.session_state.update_frequency,
            help=translate("feed_frequency_help"),
        )
        st.session_state.update_frequency = new_freq
        st.write(display_crontab_description(st.session_state.update_frequency))

        if provider == "google":
            if "evaluate_articles_quality" not in st.session_state:
                st.session_state.evaluate_articles_quality = (
                    False
                    if not config
                    else config.get("evaluate_articles_quality", False)
                )
            evaluate_articles_quality = st.checkbox(
                translate("evaluate_articles_quality"),
                value=st.session_state.evaluate_articles_quality,
                help=translate("evaluate_articles_quality_help"),
            )
            st.session_state.evaluate_articles_quality = evaluate_articles_quality

    elif provider == "atom":
        query = st.text_input(
            translate("feed_atom_label") + " :red[*]",
            value="" if not config else config["query"],
            help=translate("feed_atom_help"),
        )

    try:
        get_understandable_cron_description(st.session_state.update_frequency)
        valid_cron = True
    except:
        valid_cron = False

    if st.button(
        translate("ok_button"),
        disabled=not chosen_id
        or not query
        or (query and provider == "atom" and not re.match(URL_PATTERN, query)),
    ):
        if not config:
            config = {}
        config["id"] = "feed_" + chosen_id
        config["feed_dir_path"] = (
            "users/" + st.session_state.username + "/feed_" + chosen_id
        )
        config["query"] = query
        config["provider"] = provider
        if not config.get("max_results"):
            config["max_results"] = (
                DEFAULT_MAX_RESULTS
                if provider != "arxiv"
                else DEFAULT_MAX_RESULTS_ARXIV
            )
        if not config.get("number_of_days"):
            config["number_of_days"] = DEFAULT_NUMBER_OF_DAYS
        if provider == "google" or provider == "arxiv":
            config["language"] = "fr" if language == "French" else "en"
            config["update_frequency"] = (
                st.session_state.update_frequency
                if valid_cron
                else DEFAULT_CRONTAB_EXPRESSION
            )
        elif provider == "atom":
            config["language"] = "fr"
            config["update_frequency"] = DEFAULT_ATOM_CRONTAB_EXPRESSION

        config["evaluate_articles_quality"] = evaluate_articles_quality

        if "update_frequency" in st.session_state:
            del st.session_state["update_frequency"]  # to avoid memory effect

        # Remove prevous crontab if any
        remove_scrapping_for_user(feed_id=chosen_id, user=st.session_state.username)

        # Save feed config and update crontab
        save_feed_config(chosen_id, config)


def save_feed_config(chosen_id, feed_config: dict):
    """Save the feed configuration to disk as a TOML file."""
    feed_path = (
        CONFIG_FEEDS_BASE_PATH / st.session_state.username / f"{chosen_id}_feed.toml"
    )
    # Save the dictionary to a TOML file
    with open(feed_path, "w") as toml_file:
        toml.dump({"data-feed": feed_config}, toml_file)
    logger.debug(f"Saved feed config {feed_config} to {feed_path}")
    schedule_scrapping(feed_path, user=st.session_state.username)
    st.rerun()


def display_crontab_description(crontab_expr: str) -> str:
    try:
        return f":blue[{INFO_ICON} {get_understandable_cron_description(crontab_expr)}]"
    except Exception:
        return f":red[{ERROR_ICON} {translate('cron_error_message')}]"


def configure_information_sources():
    """Configure Information Sources."""
    # if "user_feeds" not in st.session_state:
    st.session_state.user_feeds, st.session_state.feed_files = read_user_feeds(
        st.session_state.username
    )

    displayed_list = []
    for k, v in st.session_state.user_feeds.items():
        displayed_list.append(
            {
                "id": k,
                "provider": v["data-feed"]["provider"],
                "query": v["data-feed"]["query"],
                "language": v["data-feed"]["language"],
                "update_frequency": v["data-feed"]["update_frequency"],
            }
        )
    df = pd.DataFrame(displayed_list)
    if not df.empty:
        df = df.sort_values(by="id", inplace=False).reset_index(drop=True)

    if st.button(
        f":green[{ADD_ICON}]", type="tertiary", help=translate("new_feed_help")
    ):
        edit_feed_monitoring()

    clickable_df_buttons = [
        (EDIT_ICON, edit_feed_monitoring, "secondary"),
        (lambda x: toggle_icon(df, x), handle_toggle_feed, "secondary"),
        (DELETE_ICON, handle_delete, "primary"),
    ]
    clickable_df(df, clickable_df_buttons)


def toggle_icon(df: pd.DataFrame, index: int) -> str:
    """Switch the toggle icon depending on the status of the scrapping feed in the crontab"""
    feed_id = df["id"][index]
    return (
        f":green[{TOGGLE_ON_ICON}]"
        if check_if_scrapping_active_for_user(
            feed_id=feed_id, user=st.session_state.username
        )
        else f":red[{TOGGLE_OFF_ICON}]"
    )


def toggle_feed(cfg: dict):
    """Activate / deactivate the feed from the crontab"""
    feed_id = cfg["id"]
    if check_if_scrapping_active_for_user(
        feed_id=feed_id, user=st.session_state.username
    ):
        if remove_scrapping_for_user(feed_id=feed_id, user=st.session_state.username):
            st.toast(
                translate("feed_deactivated_message").format(feed_id=feed_id),
                icon=INFO_ICON,
            )
            logger.info(f"Flux {feed_id} désactivé !")
    else:
        schedule_scrapping(
            st.session_state.feed_files[feed_id], user=st.session_state.username
        )
        st.toast(
            translate("feed_activated_message").format(feed_id=feed_id),
            icon=WARNING_ICON,
        )
        logger.info(f"Flux {feed_id} activé !")
    time.sleep(0.2)
    st.rerun()


def delete_feed_config(feed_id: str):
    # remove config file
    file_path: Path = st.session_state.feed_files[feed_id]
    try:
        file_path.unlink()
        logger.debug(f"Feed file {file_path} has been removed.")
    except Exception as e:
        logger.error(f"An error occurred: {e}")


@st.dialog(translate("confirmation_dialog_title"))
def handle_delete(row_dict: dict):
    """Function to handle remove click events"""
    feed_id = row_dict["id"]
    st.write(
        f":orange[{WARNING_ICON}] {translate('delete_feed_confirmation').format(feed_id=feed_id)}"
    )
    col1, col2, _ = st.columns([2, 2, 8])
    with col1:
        if st.button(translate("yes_button"), type="primary"):
            remove_scrapping_for_user(feed_id=feed_id, user=st.session_state.username)
            delete_feed_config(feed_id)
            logger.info(f"Flux {feed_id} supprimé !")
            # Remove from crontab associated training
            remove_scheduled_training_for_user(
                model_id=feed_id, user=st.session_state.username
            )
            time.sleep(0.2)
            st.rerun()
    with col2:
        if st.button(translate("no_button")):
            st.rerun()


@st.dialog(translate("confirmation_dialog_title"))
def handle_toggle_feed(row_dict: dict):
    """Function to handle remove click events"""
    feed_id = row_dict["id"]
    if check_if_scrapping_active_for_user(
        feed_id=feed_id, user=st.session_state.username
    ):
        st.write(
            f":orange[{WARNING_ICON}] {translate('deactivate_feed_confirmation').format(feed_id=feed_id)}"
        )
        col1, col2, _ = st.columns([2, 2, 8])
        with col1:
            if st.button(translate("yes_button"), type="primary"):
                toggle_feed(row_dict)
                st.rerun()
        with col2:
            if st.button(translate("no_button")):
                st.rerun()
    else:
        st.write(
            f":blue[{INFO_ICON}] {translate('activate_feed_message').format(feed_id=feed_id)}"
        )
        toggle_feed(row_dict)
        st.rerun()
