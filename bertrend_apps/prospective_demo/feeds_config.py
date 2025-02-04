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
from bertrend_apps.common.crontab_utils import (
    get_understandable_cron_description,
    check_if_scrapping_active_for_user,
    remove_scrapping_for_user,
    schedule_scrapping,
)
from bertrend_apps.data_provider import URL_PATTERN
from bertrend_apps.prospective_demo.feeds_common import (
    USER_FEEDS_BASE_PATH,
    read_user_feeds,
)
from bertrend_apps.prospective_demo.streamlit_utils import clickable_df

# Default feed configs
DEFAULT_GNEWS_CRONTAB_EXPRESSION = "1 0 * * 1"
DEFAULT_CUREBOT_CRONTAB_EXPRESSION = "42 0,6,12,18 * * *"  # 4 times a day
DEFAULT_MAX_RESULTS = 20
DEFAULT_NUMBER_OF_DAYS = 14
FEED_SOURCES = ["google", "curebot"]
TRANSLATION = {"English": "Anglais", "French": "Français"}


@st.dialog("Configuration d'un nouveau flux de données")
def edit_feed_monitoring(config: dict | None = None):
    """Create or update a feed monitoring configuration."""
    chosen_id = st.text_input(
        "ID :red[*]",
        help="Identifiant du flux de données",
        value=None if not config else config["id"],
    )

    provider = st.segmented_control(
        "Source",
        selection_mode="single",
        options=FEED_SOURCES,
        default=FEED_SOURCES[0] if not config else config["provider"],
        help="Sélection de la source de données",
    )
    if provider == "google":
        query = st.text_input(
            "Requête :red[*]",
            value="" if not config else config["query"],
            help="Saisir ici la requête qui sera faite sur Google News",
        )
        language = st.segmented_control(
            "Langue",
            selection_mode="single",
            options=LANGUAGES,
            default=LANGUAGES[0],
            format_func=lambda lang: TRANSLATION[lang],
            help="Choix de la langue",
        )
        if "update_frequency" not in st.session_state:
            st.session_state.update_frequency = (
                DEFAULT_GNEWS_CRONTAB_EXPRESSION
                if not config
                else config["update_frequency"]
            )
        new_freq = st.text_input(
            f"Fréquence d'exécution",
            value=st.session_state.update_frequency,
            help=f"Fréquence de collecte des données",
        )
        st.session_state.update_frequency = new_freq
        st.write(display_crontab_description(st.session_state.update_frequency))

    elif provider == "curebot":
        query = st.text_input(
            "ATOM feed :red[*]",
            value="" if not config else config["query"],
            help="URL du flux de données Curebot",
        )

    try:
        get_understandable_cron_description(st.session_state.update_frequency)
        valid_cron = True
    except:
        valid_cron = False

    if st.button(
        "OK",
        disabled=not chosen_id
        or not query
        or (query and provider == "curebot" and not re.match(URL_PATTERN, query)),
    ):
        if not config:
            config = {}
        config["id"] = "feed_" + chosen_id
        config["feed_dir_path"] = st.session_state.username + "/feed_" + chosen_id
        config["query"] = query
        config["provider"] = provider
        if not config.get("max_results"):
            config["max_results"] = DEFAULT_MAX_RESULTS
        if not config.get("number_of_days"):
            config["number_of_days"] = DEFAULT_NUMBER_OF_DAYS
        if provider == "google":
            config["language"] = "fr" if language == "French" else "en"
            config["update_frequency"] = (
                st.session_state.update_frequency
                if valid_cron
                else DEFAULT_GNEWS_CRONTAB_EXPRESSION
            )
        elif provider == "curebot":
            config["language"] = "fr"
            config["update_frequency"] = DEFAULT_CUREBOT_CRONTAB_EXPRESSION

        if "update_frequency" in st.session_state:
            del st.session_state["update_frequency"]  # to avoid memory effect

        # Remove prevous crontab if any
        remove_scrapping_for_user(feed_id=chosen_id, user=st.session_state.username)

        # Save feed config and update crontab
        save_feed_config(chosen_id, config)


def save_feed_config(chosen_id, feed_config: dict):
    """Save the feed configuration to disk as a TOML file."""
    feed_path = (
        USER_FEEDS_BASE_PATH / st.session_state.username / f"{chosen_id}_feed.toml"
    )
    # Save the dictionary to a TOML file
    with open(feed_path, "w") as toml_file:
        toml.dump({"data-feed": feed_config}, toml_file)
    logger.debug(f"Saved feed config {feed_config} to {feed_path}")
    schedule_scrapping(feed_path)
    st.rerun()


def display_crontab_description(crontab_expr: str) -> str:
    try:
        return f":blue[{INFO_ICON} {get_understandable_cron_description(crontab_expr)}]"
    except Exception:
        return f":red[{ERROR_ICON} Expression mal écrite !]"


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

    if st.button(ADD_ICON, type="tertiary", help="Nouveau flux de veille"):
        edit_feed_monitoring()

    clickable_df_buttons = [
        (EDIT_ICON, edit_feed_monitoring, "secondary"),
        (lambda x: toggle_icon(df, x), toggle_feed, "secondary"),
        (DELETE_ICON, handle_delete, "primary"),
    ]
    clickable_df(df, clickable_df_buttons)


def toggle_icon(df: pd.DataFrame, index: int) -> str:
    """Switch the toggle icon depending on the statis of the scrapping feed in the crontab"""
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
            st.toast(f"Le flux **{feed_id}** est déactivé !", icon=INFO_ICON)
            logger.info(f"Flux {feed_id} désactivé !")
    else:
        schedule_scrapping(st.session_state.feed_files[feed_id])
        st.toast(f"Le flux **{feed_id}** est activé !", icon=WARNING_ICON)
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


@st.dialog("Confirmation")
def handle_delete(row_dict: dict):
    """Function to handle remove click events"""
    feed_id = row_dict["id"]
    st.write(
        f":orange[{WARNING_ICON}] Voulez-vous vraiment supprimer le flux de veille **{feed_id}** ?"
    )
    col1, col2, _ = st.columns([2, 2, 8])
    with col1:
        if st.button("Oui", type="primary"):
            remove_scrapping_for_user(feed_id=feed_id, user=st.session_state.username)
            delete_feed_config(feed_id)
            logger.info(f"Flux {feed_id} supprimé !")
            time.sleep(0.2)
            st.rerun()
    with col2:
        if st.button("Non"):
            st.rerun()
