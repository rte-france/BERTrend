#  Copyright (c) 2024-2026, RTE (https://www.rte-france.com)
#  See AUTHORS.txt
#  SPDX-License-Identifier: MPL-2.0
#  This file is part of BERTrend.

from pathlib import Path
from unittest.mock import patch

import pandas as pd

from bertrend.bertrend_apps.prospective_demo import (
    LLM_TOPIC_DESCRIPTION_COLUMN,
    LLM_TOPIC_TITLE_COLUMN,
)
from bertrend.bertrend_apps.prospective_demo import dashboard_analysis
from bertrend.bertrend_apps.prospective_demo import automated_report_generation
from bertrend.bertrend_apps.prospective_demo import report_generation
from bertrend.bertrend_apps.prospective_demo import dashboard_signals
from bertrend.bertrend_apps.prospective_demo.dashboard_signals import (
    _prepare_topics_for_display,
)
from bertrend.bertrend_apps.prospective_demo.topic_feedback import (
    HIDDEN_TOPIC,
    PROMOTED_TOPIC,
    apply_topic_feedback,
    set_topic_feedback,
)


def test_signal_table_hides_and_marks_saved_topic_feedback():
    topics = pd.DataFrame(
        {
            "Topic": [1, 2, 3],
            LLM_TOPIC_TITLE_COLUMN: ["Most popular", "Hidden", "Preferred"],
            "Latest_Popularity": [100, 50, 10],
            "Documents": [["one"], ["two"], ["three"]],
        }
    )
    columns = ["Topic", LLM_TOPIC_TITLE_COLUMN, "Latest_Popularity", "Documents"]

    displayed = _prepare_topics_for_display(
        topics,
        columns,
        {2: HIDDEN_TOPIC, 3: PROMOTED_TOPIC},
    )

    assert displayed["Topic"].tolist() == [3, 1]
    assert displayed[LLM_TOPIC_TITLE_COLUMN].tolist() == [
        "⭐ Preferred",
        "Most popular",
    ]


def test_signal_table_path_matches_raw_input_not_preapplied_frames():
    """Tables must apply feedback after popularity sort on raw frames.

    Pre-applying feedback before _prepare_topics_for_display is redundant but
    must not change outcomes; raw frames are the intended table input.
    """
    topics = pd.DataFrame(
        {
            "Topic": [1, 2, 3],
            LLM_TOPIC_TITLE_COLUMN: ["Most popular", "Hidden", "Preferred"],
            "Latest_Popularity": [100, 50, 10],
            "Documents": [["one"], ["two"], ["three"]],
        }
    )
    columns = ["Topic", LLM_TOPIC_TITLE_COLUMN, "Latest_Popularity", "Documents"]
    feedback = {2: HIDDEN_TOPIC, 3: PROMOTED_TOPIC}

    from_raw = _prepare_topics_for_display(topics, columns, feedback)
    preapplied = apply_topic_feedback(topics, feedback)
    from_preapplied = _prepare_topics_for_display(preapplied, columns, feedback)

    assert from_raw["Topic"].tolist() == [3, 1]
    assert from_raw["Topic"].tolist() == from_preapplied["Topic"].tolist()
    assert (
        from_raw[LLM_TOPIC_TITLE_COLUMN].tolist()
        == from_preapplied[LLM_TOPIC_TITLE_COLUMN].tolist()
    )


def test_signal_analysis_passes_raw_to_tables_and_applied_to_explore():
    """Structural + behavioral: signal_analysis wiring matches review fix."""
    source = Path(dashboard_signals.__file__).read_text(encoding="utf-8")
    analysis = source.split("def signal_analysis")[1].split("def explore_topic_sources")[
        0
    ]
    assert "raw_topics = get_df_topics(" in analysis
    assert "raw_topics[NOISE]" in analysis
    assert "raw_topics[WEAK_SIGNALS]" in analysis
    assert "raw_topics[STRONG_SIGNALS]" in analysis
    assert "explore_topic_sources(dfs_topics, feedback)" in analysis
    assert "display_translated_signal_categories(" in analysis
    # Tables must not receive the feedback-applied dict entries.
    assert "dfs_topics[NOISE]" not in analysis
    assert "dfs_topics[WEAK_SIGNALS]" not in analysis
    assert "dfs_topics[STRONG_SIGNALS]" not in analysis

    prepare_src = source.split("def _prepare_topics_for_display")[1].split(
        "def display_topic_links"
    )[0]
    assert "sort_values" in prepare_src
    assert "apply_topic_feedback(displayed_topics, feedback)" in prepare_src


def test_report_picker_omits_hidden_topics_and_prioritizes_promoted_topics():
    topics = pd.DataFrame(
        {
            "Topic": [1, 2, 3],
            LLM_TOPIC_TITLE_COLUMN: ["Normal", "Hidden", "Preferred"],
            LLM_TOPIC_DESCRIPTION_COLUMN: ["One", "Two", "Three"],
        }
    )
    displayed = {}

    def capture_data_editor(dataframe, **_kwargs):
        displayed["topics"] = dataframe.copy()
        return dataframe

    with patch.object(
        report_generation.st, "data_editor", side_effect=capture_data_editor
    ):
        selected = report_generation.choose_from_df(
            topics,
            {2: HIDDEN_TOPIC, 3: PROMOTED_TOPIC},
        )

    assert selected == [3, 1]
    assert displayed["topics"]["Topic"].tolist() == [3, 1]
    assert displayed["topics"]["Sujet"].tolist() == ["⭐ Preferred", "Normal"]


def test_automated_report_respects_feedback_before_topic_limits(tmp_path):
    model_path = tmp_path / "model"
    interpretation_path = tmp_path / "interpretation"
    interpretation_path.mkdir()
    pd.DataFrame({"Topic": [1, 2, 3]}).to_parquet(
        interpretation_path / "weak_signals.parquet"
    )
    set_topic_feedback(model_path, 2, HIDDEN_TOPIC)
    set_topic_feedback(model_path, 3, PROMOTED_TOPIC)

    with (
        patch.object(
            automated_report_generation,
            "get_model_interpretation_path",
            return_value=interpretation_path,
        ),
        patch.object(
            automated_report_generation,
            "get_user_models_path",
            return_value=model_path,
        ),
    ):
        weak_signals, strong_signals = automated_report_generation.load_signal_data(
            "user",
            "monitoring-feed",
            pd.Timestamp("2026-07-31"),
            max_emerging_topics=2,
        )

    assert weak_signals["Topic"].tolist() == [3, 1]
    assert strong_signals is None


def test_feedback_control_saves_a_changed_preference():
    interpretation_path = Path("/tmp/analysis/2026-07-31")

    with (
        patch.object(
            dashboard_analysis.st,
            "segmented_control",
            return_value=HIDDEN_TOPIC,
        ),
        patch.object(dashboard_analysis.st, "session_state", {}),
        patch.object(dashboard_analysis.st, "rerun") as rerun,
        patch.object(dashboard_analysis, "translate", side_effect=lambda key: key),
        patch.object(dashboard_analysis, "set_topic_feedback") as save_feedback,
    ):
        dashboard_analysis._display_topic_feedback_control(
            "monitoring-feed",
            interpretation_path,
            7,
            {},
        )

    save_feedback.assert_called_once_with(interpretation_path, 7, HIDDEN_TOPIC)
    rerun.assert_called_once_with(scope="app")


def test_feedback_control_clears_an_existing_preference():
    interpretation_path = Path("/tmp/analysis/2026-07-31")

    with (
        patch.object(
            dashboard_analysis.st,
            "segmented_control",
            return_value=dashboard_analysis.NO_TOPIC_FEEDBACK,
        ),
        patch.object(dashboard_analysis.st, "session_state", {}),
        patch.object(dashboard_analysis.st, "rerun") as rerun,
        patch.object(dashboard_analysis, "translate", side_effect=lambda key: key),
        patch.object(dashboard_analysis, "set_topic_feedback") as save_feedback,
    ):
        dashboard_analysis._display_topic_feedback_control(
            "monitoring-feed",
            interpretation_path,
            7,
            {7: HIDDEN_TOPIC},
        )

    save_feedback.assert_called_once_with(interpretation_path, 7, None)
    rerun.assert_called_once_with(scope="app")
