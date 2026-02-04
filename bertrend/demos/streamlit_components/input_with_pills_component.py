from typing import Callable

import streamlit as st

from bertrend.demos.demos_utils.icons import PERSON_ADD_ICON


def input_with_pills(
    label: str,
    key: str,
    placeholder: str = "",
    validate_fn: Callable[[str], bool] | None = None,
    label_visibility: str = "hidden",
    help: str = None,
    value: list[str] = None,
) -> list[str] | None:
    """
    A simple Streamlit component that auto-adds valid input to pills.
    Dialog-safe version that doesn't use st.rerun()
    """

    # Define all keys
    all_items_key = f"{key}_all_items"
    input_key = f"{key}_input"
    pills_widget_key = f"{key}_pills_widget"
    selection_tracking_key = f"{key}_selection_tracking"

    if all_items_key not in st.session_state:
        initial_value = value if value is not None else []
        st.session_state[all_items_key] = initial_value
        st.session_state[selection_tracking_key] = initial_value[:]

    if input_key not in st.session_state:
        st.session_state[input_key] = ""

    col1, col2 = st.columns([3, 3])

    with col1:
        with st.form(key=f"{key}_form", clear_on_submit=True, border=False):
            current_input = st.text_input(
                label,
                placeholder=placeholder,
                key=input_key,
                label_visibility=label_visibility,
                help=help,
            )
            submitted = st.form_submit_button(PERSON_ADD_ICON)

            if submitted and current_input.strip():
                is_valid = True
                error_msg = ""
                stripped_input = current_input.strip()

                if validate_fn and not validate_fn(stripped_input):
                    is_valid = False
                    error_msg = "Invalid format"
                elif stripped_input in st.session_state[all_items_key]:
                    is_valid = False
                    error_msg = "Already exists"

                if is_valid:
                    # Add to the list of all items
                    st.session_state[all_items_key].append(stripped_input)

                    # Update tracking selection
                    st.session_state[selection_tracking_key].append(stripped_input)

                    # Update pills widget selection
                    st.session_state[pills_widget_key] = st.session_state[
                        selection_tracking_key
                    ][:]
                else:
                    st.error(error_msg)

    with col2:
        # Display pills
        if st.session_state[all_items_key]:
            st.markdown("<div style='margin-top: 8px;'></div>", unsafe_allow_html=True)

            # Initialize pills widget if not exists
            if pills_widget_key not in st.session_state:
                st.session_state[pills_widget_key] = st.session_state[
                    selection_tracking_key
                ][:]

            def on_pills_change():
                # Sync the widget selection to our tracking key
                st.session_state[selection_tracking_key] = st.session_state[
                    pills_widget_key
                ][:]

            selected = st.pills(
                "Items",
                st.session_state[all_items_key],
                selection_mode="multi",
                key=pills_widget_key,
                label_visibility="collapsed",
                on_change=on_pills_change,
            )

    st.session_state[key] = st.session_state[selection_tracking_key]
    return st.session_state[key]
