#  Copyright (c) 2024, RTE (https://www.rte-france.com)
#  See AUTHORS.txt
#  SPDX-License-Identifier: MPL-2.0
#  This file is part of BERTrend.
from pathlib import Path
from tempfile import TemporaryDirectory

import pandas as pd
import streamlit as st
from streamlit.runtime.uploaded_file_manager import UploadedFile

from bertrend import DATA_PATH
from bertrend.config.parameters import MIN_CHARS_DEFAULT, SAMPLE_SIZE_DEFAULT
from bertrend.demos.demos_utils.i18n import (
    get_current_internationalization_language,
    translate,
)
from bertrend.demos.demos_utils.icons import (
    CLIENT_STORAGE_ICON,
    CSV_ICON,
    JSON_ICON,
    PARQUET_ICON,
    SERVER_STORAGE_ICON,
    WARNING_ICON,
    XLSX_ICON,
)
from bertrend.demos.demos_utils.state_utils import (
    SessionStateManager,
    register_widget,
    save_widget_state,
)
from bertrend.utils.data_loading import (
    TEXT_COLUMN,
    TIMESTAMP_COLUMN,
    DataLoadingError,
    _check_data,
    _clean_data,
    _file_to_pd,
    find_compatible_files,
    load_data,
    split_data,
)

FORMAT_ICONS = {
    "xlsx": XLSX_ICON,
    "csv": CSV_ICON,
    "parquet": PARQUET_ICON,
    "json": JSON_ICON,
    "jsonl": JSON_ICON,
    "jsonlines": JSON_ICON,
    "jsonl.gz": JSON_ICON,
}


@st.dialog(translate("column_selection"))
def _select_alternative_columns(df: pd.DataFrame, message: str = ""):
    (st.warning(message, icon=WARNING_ICON),)
    if TEXT_COLUMN not in df.columns:
        text_column = st.selectbox(
            label=translate("text_column_selection"), options=df.columns
        )
    else:
        text_column = None
    if TIMESTAMP_COLUMN not in df.columns:
        timestamp_column = st.selectbox(
            label=translate("timestamp_column_selection"), options=df.columns
        )
    else:
        timestamp_column = None
    if st.button("Submit"):
        if text_column:
            df[TEXT_COLUMN] = df[text_column]
        if timestamp_column:
            df[TIMESTAMP_COLUMN] = df[timestamp_column]
        st.session_state["modified_df"] = df
        st.rerun()


#    @st.cache_data(ttl=60 * 60 * 24)
def _process_uploaded_files(
    files: list[UploadedFile],
) -> list[pd.DataFrame]:
    """Read a list of uploaded files and return a list of dataframes containing the associated data"""
    dataframes = []
    with TemporaryDirectory() as tmpdir:
        for f in files:
            with open(tmpdir + "/" + f.name, "wb") as tmp_file:
                tmp_file.write(f.getvalue())
            if tmp_file is not None:
                if "modified_df" in st.session_state:
                    df = st.session_state["modified_df"]
                    st.session_state.pop("modified_df")
                else:
                    df = _file_to_pd(Path(tmp_file.name))
                try:
                    _check_data(df, Path(tmp_file.name))
                except DataLoadingError as dle:
                    message = translate("error_loading_file").format(
                        file_name=f.name, error=dle
                    )
                    if "modified_df" in st.session_state:
                        st.session_state.pop("modified_df")
                    _select_alternative_columns(df, message)
                    df = None
                if df is not None:
                    df = _clean_data(df, SessionStateManager.get("language", "French"))
                    dataframes.append(df)
        return dataframes


@st.cache_data(ttl=60 * 60 * 24)
def _load_files(
    files: list[Path],
) -> list[pd.DataFrame]:
    """Read a list of files from storage and return a list of dataframes containing the associated data"""
    dfs = []
    for selected_file in files:
        file_path = DATA_PATH / selected_file
        try:
            df = load_data(
                file_path,
                SessionStateManager.get("language", "French"),
            )
        except DataLoadingError as dle:
            st.warning(
                translate("error_loading_file").format(
                    file_name=file_path.name, error=dle
                ),
                icon=WARNING_ICON,
            )
            _select_alternative_columns(df)
            df = (
                st.session_state["modified_df"]
                if "modified_df" in st.session_state
                else None
            )
        if df is not None:
            dfs.append(df)
    return dfs


# TODO: if loaded data column names do not match our default values, show a popup for column data mapping
def display_data_loading_component():
    """
    Component for a streamlit app about topic modeling. It allows choosing data to load and preprocess data.
    Preprocessing of data includes:
    - concatenation of data from different files
    - adding potentially missing columns to make all datasets homogeneous
    - removal of duplicates
    - splitting the dataset by paragraphs to avoid too long textes
    - filtering based on timestamp range
    - filtering based on a minimum number of characters

    The initial dataframe (one line per document) is stored after filtering of bad data inside a Streamlit state
    variable "initial_df". After split by paragraph, it is stored in "df_split". The final dataframe (possibly
    split by paragraph from initial documents and in all cases filtered by dates) is stored inside the Streamlit
    state variable "time_filtered_df".
    """
    # Data loading section
    st.header(translate("data_loading"))

    # Find files in the current directory and subdirectories
    tab1, tab2 = st.tabs(
        [
            CLIENT_STORAGE_ICON + " " + translate("local_data"),
            SERVER_STORAGE_ICON + " " + translate("remote_data"),
        ]
    )
    compatible_extensions = FORMAT_ICONS.keys()

    with tab1:
        st.file_uploader(
            label=translate("select_from_local_storage"),
            type=compatible_extensions,
            accept_multiple_files=True,
            help=translate("drag_drop_help"),
            on_change=save_widget_state,
            key="uploaded_files",
        )

    with tab2:
        register_widget("selected_files")
        st.multiselect(
            label=translate("select_from_remote_storage"),
            options=find_compatible_files(DATA_PATH, compatible_extensions),
            default=SessionStateManager.get("selected_files", []),
            key="selected_files",
            format_func=lambda x: FORMAT_ICONS[x.suffix.lstrip(".")] + " " + str(x),
            on_change=save_widget_state,
        )

    if (
        not SessionStateManager.get("uploaded_files", [])
        and not SessionStateManager.get("selected_files")
        and "initial_df" not in st.session_state
    ):
        st.warning(translate("no_dataset_warning"), icon=WARNING_ICON)
        st.stop()

    # Load each selected file, then concatenate them
    # Priority to local data if both are set
    dfs = None
    if SessionStateManager.get("uploaded_files"):
        dfs = _process_uploaded_files(
            SessionStateManager.get("uploaded_files"),
        )
    elif SessionStateManager.get("selected_files"):
        dfs = _load_files(
            SessionStateManager.get("selected_files"),
        )

    # If DataFrames found, concatenate them
    if dfs:
        # Concatenate DataFrames
        df = pd.concat(dfs, ignore_index=True)

        # Save state of initial DF (before split and data selection)
        st.session_state["initial_df"] = df.copy()
    # If dfs is None and there is no 'initial_df' in session_state, show a warning
    elif "initial_df" not in st.session_state:
        st.warning(
            translate("no_data_after_preprocessing_message"),
            icon=WARNING_ICON,
        )
        st.stop()
    # If dfs is None but there is initial_df, set it
    else:
        df = st.session_state["initial_df"]

    # Show raw data info
    st.write(
        translate("raw_documents_count").format(
            count=len(st.session_state["initial_df"])
        )
    )

    # Data filtering section
    st.header(translate("data_filtering"))

    # Display number input and checkbox for preprocessing options
    col1, col2, col3 = st.columns(3)
    with col1:
        register_widget("min_chars")
        st.number_input(
            translate("minimum_characters"),
            value=MIN_CHARS_DEFAULT,
            min_value=0,
            max_value=1000,
            key="min_chars",
            on_change=save_widget_state,
            help=translate("minimum_characters_help"),
        )
    with col2:
        register_widget("sample_size")
        sample_size = st.number_input(
            translate("sample_ratio"),
            value=SAMPLE_SIZE_DEFAULT,
            min_value=0.0,
            max_value=1.0,
            key="sample_size",
            on_change=save_widget_state,
            help=translate("sample_ratio_help"),
        )
    with col3:
        register_widget("split_by_paragraph")
        SessionStateManager.get_or_set("split_by_paragraph", "no")
        st.segmented_control(
            translate("split_by_paragraph"),
            key="split_by_paragraph",
            options=["yes", "no", "enhanced"],
            selection_mode="single",
            help=translate("split_help"),
            on_change=save_widget_state,
            format_func=lambda x: translate("split_option_" + x),
        )

    if get_current_internationalization_language() == "en":
        split_by_paragraph_param = SessionStateManager.get("split_by_paragraph")
    else:
        # option has been translated
        split_by_paragraph_param = (
            "no"
            if SessionStateManager.get("split_by_paragraph") == "non"
            else (
                "enhanced"
                if SessionStateManager.get("split_by_paragraph") == "amélioré"
                else "yes"
            )
        )

    df = split_data(
        df=df,
        min_chars=SessionStateManager.get("min_chars"),
        split_by_paragraph=SessionStateManager.get("split_by_paragraph"),
        embedding_model_name=SessionStateManager.get("embedding_model_name"),
    )

    # Deduplicate using all columns
    df = df.drop_duplicates()

    # Save state of split dataframe (before time-based filtering)
    st.session_state["split_df"] = df.copy()

    # Select timeframe
    min_date, max_date = st.session_state["initial_df"][TIMESTAMP_COLUMN].dt.date.agg(
        ["min", "max"]
    )
    register_widget("timeframe_slider")
    start_date, end_date = st.slider(
        translate("select_timeframe"),
        min_value=min_date,
        max_value=max_date,
        value=(min_date, max_date),
        key="timeframe_slider",
        on_change=save_widget_state,
    )

    # Filter and sample the DataFrame
    df_filtered = df[
        (df[TIMESTAMP_COLUMN].dt.date >= start_date)
        & (df[TIMESTAMP_COLUMN].dt.date <= end_date)
    ]
    df_filtered = df_filtered.sort_values(by=TIMESTAMP_COLUMN).reset_index(drop=True)

    if sample_size < 1:
        df_filtered = df_filtered.sample(frac=sample_size, random_state=42)

    df_filtered = df_filtered.sort_values(by=TIMESTAMP_COLUMN).reset_index(drop=True)

    SessionStateManager.set("time_filtered_df", df_filtered)
    st.write(
        translate("filtered_documents_count").format(
            count=len(SessionStateManager.get_dataframe("time_filtered_df"))
        )
    )
    st.dataframe(
        SessionStateManager.get_dataframe("time_filtered_df")[
            [TEXT_COLUMN, "timestamp"]
        ],
        width="stretch",
    )
