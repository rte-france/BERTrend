#  Copyright (c) 2024, RTE (https://www.rte-france.com)
#  See AUTHORS.txt
#  SPDX-License-Identifier: MPL-2.0
#  This file is part of BERTrend.

"""
Comparative analysis dashboard for comparing topic trends between different time periods.

This module provides functionality to:
- Select and compare two different time periods
- Analyze topic evolution (new, disappeared, stable topics)
- Compare popularity changes across topics
- Visualize source diversity changes
- Display comparative metrics with interactive charts
"""

from datetime import datetime
import uuid

import pandas as pd
import streamlit as st
import plotly.graph_objects as go

from bertrend.demos.demos_utils.icons import (
    WARNING_ICON,
    ANALYSIS_ICON,
    ERROR_ICON,
    TIMELINE_ICON,
    POPULARITY_ICON,
    SIGNAL_EVOLUTION_ICON,
)
from bertrend_apps.prospective_demo.i18n import translate
from bertrend_apps.prospective_demo import (
    NOISE,
    WEAK_SIGNALS,
    STRONG_SIGNALS,
    LLM_TOPIC_TITLE_COLUMN,
    get_model_interpretation_path,
)
from bertrend_apps.prospective_demo.dashboard_common import (
    get_df_topics,
    update_key,
)
from bertrend_apps.prospective_demo.models_info import get_models_info

# Constants
DEFAULT_CHART_HEIGHT = 400
TOP_TOPICS_DISPLAY_LIMIT = 10
DATE_FORMAT = "%d/%m/%Y"
COLUMN_TOPIC = "Topic"
COLUMN_POPULARITY = "Latest_Popularity"
COLUMN_SOURCE_DIVERSITY = "Source_Diversity"
SIGNAL_CATEGORIES = [WEAK_SIGNALS, STRONG_SIGNALS]


def choose_two_periods() -> tuple[str, datetime, datetime]:
    """
    Allow user to select two different time periods for comparison.

    This function displays UI controls for selecting a feed and two analysis periods.
    It validates that at least 2 periods are available before allowing selection.

    Returns:
        tuple[str, datetime, datetime]: A tuple containing:
            - model_id: The selected feed/model identifier
            - period_1: The first selected analysis period
            - period_2: The second selected analysis period

    Raises:
        st.stop(): If no models are available or fewer than 2 periods exist
    """
    col1, col2 = st.columns(2)

    # Get available models/timestamps with validation
    try:
        options = sorted(st.session_state.user_feeds.keys())
        if not options:
            st.error(translate("no_available_model_warning"), icon=ERROR_ICON)
            st.stop()
    except (AttributeError, KeyError) as e:
        st.error(
            translate("error_accessing_user_feeds").format(error=e), icon=ERROR_ICON
        )
        st.stop()

    # Initialize model_id in session state if not present
    if "model_id" not in st.session_state:
        st.session_state.model_id = options[0]

    # Model selection
    model_id = st.selectbox(
        translate("select_feed"),
        options=options,
        index=options.index(st.session_state.model_id),
        key="comparative_model_id",
    )
    st.session_state.model_id = model_id

    # Get available periods for selected model
    try:
        list_models = get_models_info(model_id, st.session_state.username)
    except Exception as e:
        st.error(translate("error_loading_model_info").format(error=e), icon=ERROR_ICON)
        st.stop()

    if not list_models:
        st.warning(translate("no_available_model_warning"), icon=WARNING_ICON)
        st.stop()
    elif len(list_models) < 2:
        st.warning(translate("at_least_2models_warning"), icon=WARNING_ICON)
        st.stop()

    # Period 1 selection
    with col1:
        period1_key = uuid.uuid4()
        if "period_1" not in st.session_state:
            # Default to second oldest period if available
            st.session_state.period_1 = (
                list_models[1] if len(list_models) >= 2 else list_models[0]
            )
        # Validate that period_1 exists in current list_models
        elif st.session_state.period_1 not in list_models:
            # Reset to default if stored period is not in current list
            st.session_state.period_1 = (
                list_models[1] if len(list_models) >= 2 else list_models[0]
            )
        period_1 = st.selectbox(
            translate("period_1"),
            options=list_models,
            index=list_models.index(st.session_state.period_1),
            format_func=lambda ts: ts.strftime(DATE_FORMAT),
            key=period1_key,
            on_change=lambda: update_key("period_1", st.session_state[period1_key]),
        )

    # Period 2 selection
    with col2:
        period2_key = uuid.uuid4()
        if "period_2" not in st.session_state:
            # Default to most recent period
            st.session_state.period_2 = list_models[-1]
        # Validate that period_2 exists in current list_models
        elif st.session_state.period_2 not in list_models:
            # Reset to default if stored period is not in current list
            st.session_state.period_2 = list_models[-1]
        period_2 = st.selectbox(
            translate("period_2"),
            options=list_models,
            index=list_models.index(st.session_state.period_2),
            format_func=lambda ts: ts.strftime(DATE_FORMAT),
            key=period2_key,
            on_change=lambda: update_key("period_2", st.session_state[period2_key]),
        )

    return model_id, period_1, period_2


@st.cache_data(ttl=3600, show_spinner=False)
def get_period_data(
    model_id: str, period: datetime, username: str
) -> dict[str, pd.DataFrame]:
    """
    Load topic data for a specific period with caching.

    This function retrieves topic data (weak signals, strong signals, noise) for a given
    model and time period. Results are cached for 1 hour to improve performance.

    Args:
        model_id: The feed/model identifier
        period: The analysis period timestamp
        username: The username for accessing user-specific data

    Returns:
        dict[str, pd.DataFrame]: Dictionary containing DataFrames for each signal type:
            - WEAK_SIGNALS: DataFrame of weak signal topics
            - STRONG_SIGNALS: DataFrame of strong signal topics
            - NOISE: DataFrame of noise topics

    Raises:
        Exception: If data cannot be loaded or path is invalid
    """
    try:
        model_interpretation_path = get_model_interpretation_path(
            user_name=username,
            model_id=model_id,
            reference_ts=period,
        )
        return get_df_topics(model_interpretation_path)
    except Exception as e:
        st.error(
            translate("error_loading_data_for_period").format(
                period=period.strftime(DATE_FORMAT), error=e
            ),
            icon=ERROR_ICON,
        )
        # Return empty dataframes to allow graceful degradation
        return {
            WEAK_SIGNALS: pd.DataFrame(),
            STRONG_SIGNALS: pd.DataFrame(),
            NOISE: pd.DataFrame(),
        }


def plot_topic_popularity_over_time(
    model_id: str, period_1: datetime, period_2: datetime, username: str
) -> None:
    """
    Plot the popularity of topics over time across multiple periods.

    This function retrieves all available time periods between period_1 and period_2,
    loads topic data for each period, and displays an interactive line chart showing
    how topic popularity evolves over time for the top topics.

    Args:
        model_id: The feed/model identifier
        period_1: Start period for the analysis
        period_2: End period for the analysis
        username: The username for accessing user-specific data
    """
    st.subheader(POPULARITY_ICON + " " + translate("topic_popularity_over_time"))

    try:
        # Get all available periods for the model
        all_periods = get_models_info(model_id, username)

        # Filter periods between period_1 and period_2
        filtered_periods = [p for p in all_periods if period_1 <= p <= period_2]

        if len(filtered_periods) < 2:
            st.warning(translate("insufficient_periods_for_trend"), icon=WARNING_ICON)
            return

        # Collect popularity data for all topics across all periods
        topic_popularity_data = {}  # {topic_id: {period: popularity}}
        topic_titles = {}  # {topic_id: title}

        with st.spinner(translate("loading_historical_data")):
            for period in filtered_periods:
                period_data = get_period_data(model_id, period, username)

                for category in SIGNAL_CATEGORIES:
                    df = period_data.get(category, pd.DataFrame())
                    if df.empty or COLUMN_TOPIC not in df.columns:
                        continue

                    for _, row in df.iterrows():
                        topic_id = row[COLUMN_TOPIC]
                        popularity = row.get(COLUMN_POPULARITY, 0)

                        if topic_id not in topic_popularity_data:
                            topic_popularity_data[topic_id] = {}
                            # Store title if available
                            if LLM_TOPIC_TITLE_COLUMN in row:
                                topic_titles[topic_id] = row[LLM_TOPIC_TITLE_COLUMN]

                        topic_popularity_data[topic_id][period] = popularity

        if not topic_popularity_data:
            st.info(translate("no_data"))
            return

        # Calculate average popularity for each topic to identify top topics
        topic_avg_popularity = {
            topic_id: sum(periods.values()) / len(periods)
            for topic_id, periods in topic_popularity_data.items()
        }

        # Select top N topics by average popularity
        top_topics = sorted(
            topic_avg_popularity.items(), key=lambda x: x[1], reverse=True
        )[:TOP_TOPICS_DISPLAY_LIMIT]

        # Create line chart
        fig = go.Figure()

        for topic_id, _ in top_topics:
            periods_list = sorted(topic_popularity_data[topic_id].keys())
            popularity_list = [
                topic_popularity_data[topic_id].get(p, 0) for p in periods_list
            ]

            # Format topic label
            if topic_id in topic_titles:
                raw_title = topic_titles.get(topic_id)
                # Handle None/NaN and ensure string
                if raw_title is None or pd.isna(raw_title):
                    safe_title = translate("not_available")
                else:
                    safe_title = str(raw_title)
                topic_label = translate("topic_with_title").format(
                    topic_id=topic_id,
                    title=safe_title[:50],
                )
            else:
                topic_label = translate("topic_without_title").format(topic_id=topic_id)

            fig.add_trace(
                go.Scatter(
                    x=periods_list,  # Use datetime objects directly
                    y=popularity_list,
                    mode="lines+markers",
                    name=topic_label,
                    hovertemplate="<b>%{fullData.name}</b><br>"
                    + translate("date_label")
                    + ": %{x|%d/%m/%Y}<br>"
                    + translate("popularity")
                    + ": %{y:.2f}<br>"
                    + "<extra></extra>",
                )
            )

        fig.update_layout(
            xaxis_title=translate("analysis_date"),
            yaxis_title=translate("popularity"),
            height=500,
            hovermode="x unified",
            showlegend=True,
            legend=dict(orientation="v", yanchor="top", y=1, xanchor="left", x=1.02),
            xaxis=dict(
                tickformat="%d/%m/%Y",  # Format dates on x-axis
                type="date",  # Ensure x-axis is treated as dates
            ),
        )

        st.plotly_chart(fig, use_container_width=True)

        # Display summary statistics
        st.caption(
            translate("showing_top_topics_summary").format(
                count=len(top_topics), periods=len(filtered_periods)
            )
        )

    except Exception as e:
        st.error(
            translate("error_plotting_popularity").format(error=e), icon=ERROR_ICON
        )
        import traceback

        with st.expander(translate("error_details_debugging")):
            st.code(traceback.format_exc())


def _extract_topic_ids(data: dict[str, pd.DataFrame]) -> set[int]:
    """
    Extract all topic IDs from weak and strong signal DataFrames.

    Args:
        data: Dictionary containing signal DataFrames

    Returns:
        set[int]: Set of unique topic IDs found in the data
    """
    topics = set()
    for category in SIGNAL_CATEGORIES:
        df = data.get(category, pd.DataFrame())
        if not df.empty and COLUMN_TOPIC in df.columns:
            topics.update(df[COLUMN_TOPIC].tolist())
    return topics


def analyze_topic_evolution(
    data1: dict[str, pd.DataFrame],
    data2: dict[str, pd.DataFrame],
    period1: datetime,
    period2: datetime,
) -> None:
    """
    Analyze topic evolution between two periods.

    Identifies and displays metrics for:
    - New topics that appeared in period 2
    - Topics that disappeared from period 1
    - Stable topics present in both periods

    Args:
        data1: Dictionary of DataFrames for the first period
        data2: Dictionary of DataFrames for the second period
        period1: First period timestamp
        period2: Second period timestamp
    """
    st.subheader(SIGNAL_EVOLUTION_ICON + " " + translate("signal_evolution"))

    try:
        # Extract topic IDs from both periods
        topics1 = _extract_topic_ids(data1)
        topics2 = _extract_topic_ids(data2)

        # Calculate evolution metrics
        new_topics = topics2 - topics1
        disappeared_topics = topics1 - topics2
        stable_topics = topics1 & topics2

        # Display metrics in columns
        col1, col2, col3 = st.columns(3)

        with col1:
            st.metric(
                label=translate("new_topics"),
                value=len(new_topics),
                delta=f"+{len(new_topics)}",
            )

        with col2:
            st.metric(
                label=translate("disappeared_topics"),
                value=len(disappeared_topics),
                delta=f"-{len(disappeared_topics)}",
                delta_color="inverse",
            )

        with col3:
            st.metric(
                label=translate("stable_topics"),
                value=len(stable_topics),
            )

        # Show detailed information in expanders
        if new_topics:
            with st.expander(f"📈 {translate('new_topics')} ({len(new_topics)})"):
                display_topic_list(data2, list(new_topics))

        if disappeared_topics:
            with st.expander(
                f"📉 {translate('disappeared_topics')} ({len(disappeared_topics)})"
            ):
                display_topic_list(data1, list(disappeared_topics))

        if stable_topics:
            with st.expander(f"🔄 {translate('stable_topics')} ({len(stable_topics)})"):
                compare_stable_topics(
                    data1, data2, list(stable_topics), period1, period2
                )

    except Exception as e:
        st.error(
            translate("error_analyzing_evolution").format(error=e), icon=ERROR_ICON
        )


def display_topic_list(data: dict[str, pd.DataFrame], topic_ids: list[int]) -> None:
    """
    Display a list of topics with their titles and metadata in a table.
    Rows are colored based on signal type: green for strong signals, orange for weak signals.

    Args:
        data: Dictionary of DataFrames containing topic information
        topic_ids: List of topic IDs to display
    """
    try:
        topics_info = []

        for category in SIGNAL_CATEGORIES:
            df = data.get(category, pd.DataFrame())
            if df.empty or COLUMN_TOPIC not in df.columns:
                continue

            for topic_id in topic_ids:
                topic_row = df[df[COLUMN_TOPIC] == topic_id]
                if not topic_row.empty:
                    title = (
                        topic_row[LLM_TOPIC_TITLE_COLUMN].values[0]
                        if LLM_TOPIC_TITLE_COLUMN in topic_row.columns
                        else translate("not_available")
                    )
                    popularity = (
                        topic_row[COLUMN_POPULARITY].values[0]
                        if COLUMN_POPULARITY in topic_row.columns
                        else 0
                    )
                    topics_info.append(
                        {
                            COLUMN_TOPIC: topic_id,
                            translate("title"): title,
                            translate("popularity"): popularity,
                            translate("category"): (
                                translate("weak_signals")
                                if category == WEAK_SIGNALS
                                else translate("strong_signals")
                            ),
                            "_signal_type": category,  # Internal field for styling
                        }
                    )

        if topics_info:
            df = pd.DataFrame(topics_info)
            # Sort by popularity in descending order
            df = df.sort_values(by=translate("popularity"), ascending=False)

            # Store signal type info before removing the column
            signal_types = df["_signal_type"].copy()

            # Remove internal field before applying styles
            df_display = df.drop(columns=["_signal_type"])

            # Apply text coloring based on signal type
            def color_rows(row):
                idx = row.name
                if signal_types.loc[idx] == WEAK_SIGNALS:
                    return ["color: orange"] * len(row)  # Orange text for weak signals
                else:  # STRONG_SIGNALS
                    return ["color: green"] * len(row)  # Green text for strong signals

            styled_df = df_display.style.apply(color_rows, axis=1)

            st.dataframe(styled_df, width="stretch")
        else:
            st.info(translate("no_data"))

    except Exception as e:
        st.error(
            translate("error_displaying_topic_list").format(error=e), icon=ERROR_ICON
        )


def compare_stable_topics(
    data1: dict[str, pd.DataFrame],
    data2: dict[str, pd.DataFrame],
    topic_ids: list[int],
    period1: datetime,
    period2: datetime,
) -> None:
    """
    Compare popularity changes for topics that appear in both periods.
    Rows are colored based on signal type: green for strong signals, orange for weak signals.

    Displays a table and bar chart showing how popularity changed for stable topics
    between the two periods.

    Args:
        data1: Dictionary of DataFrames for the first period
        data2: Dictionary of DataFrames for the second period
        topic_ids: List of topic IDs that appear in both periods
        period1: First period timestamp
        period2: Second period timestamp
    """
    try:
        comparison_data = []

        for category in SIGNAL_CATEGORIES:
            df1 = data1.get(category, pd.DataFrame())
            df2 = data2.get(category, pd.DataFrame())

            if df1.empty or df2.empty:
                continue

            for topic_id in topic_ids:
                topic1 = df1[df1[COLUMN_TOPIC] == topic_id]
                topic2 = df2[df2[COLUMN_TOPIC] == topic_id]

                if not topic1.empty and not topic2.empty:
                    title = (
                        topic2[LLM_TOPIC_TITLE_COLUMN].values[0]
                        if LLM_TOPIC_TITLE_COLUMN in topic2.columns
                        else translate("topic_without_title").format(topic_id=topic_id)
                    )
                    pop1 = (
                        topic1[COLUMN_POPULARITY].values[0]
                        if COLUMN_POPULARITY in topic1.columns
                        else 0
                    )
                    pop2 = (
                        topic2[COLUMN_POPULARITY].values[0]
                        if COLUMN_POPULARITY in topic2.columns
                        else 0
                    )
                    change = pop2 - pop1
                    change_percent = (change / pop1 * 100) if pop1 > 0 else 0

                    comparison_data.append(
                        {
                            COLUMN_TOPIC: topic_id,
                            translate("title"): title,
                            period1.strftime(DATE_FORMAT): pop1,
                            period2.strftime(DATE_FORMAT): pop2,
                            translate("popularity_change"): change,
                            translate("change_percent"): change_percent,
                            "_signal_type": category,  # Internal field for styling
                        }
                    )

        if comparison_data:
            df = pd.DataFrame(comparison_data)
            df = df.sort_values(by=translate("popularity_change"), ascending=False)

            # Store signal type info before removing the column
            signal_types = df["_signal_type"].copy()

            # Remove internal field before applying styles
            df_display = df.drop(columns=["_signal_type"])

            # Apply text coloring based on signal type
            def color_rows(row):
                idx = row.name
                if signal_types.loc[idx] == WEAK_SIGNALS:
                    return ["color: orange"] * len(row)  # Orange text for weak signals
                else:  # STRONG_SIGNALS
                    return ["color: green"] * len(row)  # Green text for strong signals

            styled_df = df_display.style.apply(color_rows, axis=1)

            # Display table with formatted columns
            st.dataframe(
                styled_df,
                width="stretch",
                column_config={
                    translate("popularity_change"): st.column_config.NumberColumn(
                        translate("popularity_change"),
                        format="%.2f",
                        help=translate("popularity_change"),
                    ),
                    translate("change_percent"): st.column_config.NumberColumn(
                        translate("change_percent"),
                        format="%.1f%%",
                        help=translate("percentage_change_help"),
                    ),
                },
            )

            # Visualize top changes
            top_n = min(TOP_TOPICS_DISPLAY_LIMIT, len(df))
            top_changes = df.head(top_n)

            fig = go.Figure()
            fig.add_trace(
                go.Bar(
                    x=top_changes[translate("title")],
                    y=top_changes[translate("popularity_change")],
                    marker_color=top_changes[translate("popularity_change")].apply(
                        lambda x: "green" if x > 0 else "red"
                    ),
                    text=top_changes[translate("popularity_change")].round(2),
                    textposition="outside",
                )
            )

            fig.update_layout(
                title=translate("popularity_change")
                + " "
                + translate("top_n_topics").format(n=top_n),
                xaxis_title=translate("topic"),
                yaxis_title=translate("popularity_change"),
                height=DEFAULT_CHART_HEIGHT,
                xaxis={"tickangle": -45},
                hovermode="x",
            )

            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info(translate("no_data"))

    except Exception as e:
        st.error(
            translate("error_comparing_stable_topics").format(error=e), icon=ERROR_ICON
        )


@st.fragment()
def dashboard_comparative() -> None:
    """
    Main dashboard for comparative analysis between different time periods.

    This function orchestrates the comparative analysis workflow:
    1. Period selection for comparison
    2. Data loading with validation
    3. Topic evolution analysis (new, disappeared, stable)
    4. Topic popularity trends over time

    The dashboard provides interactive visualizations and metrics to help users
    understand how topics and their characteristics have changed between two periods.
    """
    st.header(TIMELINE_ICON + " " + translate("comparative_analysis_title"))

    try:
        # Select two periods to compare
        model_id, period_1, period_2 = choose_two_periods()

        # Add a button to explicitly trigger rendering to avoid useless computations
        # Reset render state when selection changes
        current_selection = (model_id, period_1, period_2)
        prev_selection = st.session_state.get("comparative_prev_selection")
        if prev_selection != current_selection:
            st.session_state["comparative_prev_selection"] = current_selection
            st.session_state["comparative_render"] = False

        # Render button
        if st.button(
            ANALYSIS_ICON + " " + translate("render_comparison"),
            key="comparative_render_button",
            type="primary",
        ):
            st.session_state["comparative_render"] = True

        if not st.session_state.get("comparative_render", False):
            st.info(translate("click_to_render_comparison"))
            st.stop()

        # Validate that periods are different
        if period_1 == period_2:
            st.warning(translate("select_two_periods"), icon=WARNING_ICON)
            st.stop()

        # Load data for both periods with progress indication
        username = st.session_state.get("username", "")
        if not username:
            st.error(translate("username_not_found"), icon=ERROR_ICON)
            st.stop()

        with st.spinner(translate("loading_data")):
            data_period_1 = get_period_data(model_id, period_1, username)
            data_period_2 = get_period_data(model_id, period_2, username)

        # Validate data availability
        period_1_has_data = any(not df.empty for df in data_period_1.values())
        period_2_has_data = any(not df.empty for df in data_period_2.values())

        if not period_1_has_data or not period_2_has_data:
            st.error(translate("no_data_for_comparison"), icon=WARNING_ICON)
            if not period_1_has_data:
                st.info(
                    translate("no_data_for_period").format(
                        period=period_1.strftime(DATE_FORMAT)
                    )
                )
            if not period_2_has_data:
                st.info(
                    translate("no_data_for_period").format(
                        period=period_2.strftime(DATE_FORMAT)
                    )
                )
            st.stop()

        # Display comparison sections
        st.divider()

        # Section 1: Topic popularity over time
        with st.container():
            plot_topic_popularity_over_time(model_id, period_1, period_2, username)

        st.divider()

        # Section 2: Topic evolution analysis
        with st.container():
            analyze_topic_evolution(data_period_1, data_period_2, period_1, period_2)

    except Exception as e:
        st.error(
            translate("error_in_comparative_dashboard").format(error=e), icon=ERROR_ICON
        )
        # Log the error for debugging (in production, use proper logging)
        import traceback

        with st.expander(translate("error_details_debugging")):
            st.code(traceback.format_exc())
