#  Copyright (c) 2024, RTE (https://www.rte-france.com)
#  See AUTHORS.txt
#  SPDX-License-Identifier: MPL-2.0
#  This file is part of BERTrend.
import hmac

import streamlit as st

from bertrend.demos.demos_utils.icons import UNHAPPY_ICON


def login_form():
    """Form with widgets to collect user information"""
    with st.form("Credentials"):
        st.text_input("Username", key="username")
        st.text_input("Password", type="password", key="password")
        st.form_submit_button("Log in", on_click=password_entered)


def password_entered():
    """Checks whether a password entered by the user is correct."""
    if st.session_state["username"] in st.secrets["passwords"] and hmac.compare_digest(
        st.session_state["password"],
        st.secrets.passwords[st.session_state["username"]],
    ):
        st.session_state["password_correct"] = True
        del st.session_state["password"]  # Don't store the username or password.
        # del st.session_state["username"]
    else:
        st.session_state["password_correct"] = False


def check_password() -> str | None:
    """Returns the user name if the user had a correct password, otherwise None."""

    # Return True if the username + password is validated.
    if st.session_state.get("password_correct", False):
        return st.session_state["username"]

    # Show inputs for username + password.
    login_form()
    if "password_correct" in st.session_state:
        st.error(f"{UNHAPPY_ICON} User not known or password incorrect")
    return None
