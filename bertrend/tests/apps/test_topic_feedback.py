#  Copyright (c) 2024-2026, RTE (https://www.rte-france.com)
#  See AUTHORS.txt
#  SPDX-License-Identifier: MPL-2.0
#  This file is part of BERTrend.

import json

import pandas as pd
import pytest

from bertrend.bertrend_apps.prospective_demo.topic_feedback import (
    HIDDEN_TOPIC,
    PROMOTED_TOPIC,
    apply_topic_feedback,
    get_topic_feedback_icon,
    load_topic_feedback,
    order_topic_ids,
    set_topic_feedback,
)


def test_topic_feedback_round_trip_and_clear(tmp_path):
    assert load_topic_feedback(tmp_path) == {}

    set_topic_feedback(tmp_path, 12, PROMOTED_TOPIC)
    set_topic_feedback(tmp_path, 7, HIDDEN_TOPIC)

    assert load_topic_feedback(tmp_path) == {
        7: HIDDEN_TOPIC,
        12: PROMOTED_TOPIC,
    }

    set_topic_feedback(tmp_path, 12, None)

    assert load_topic_feedback(tmp_path) == {7: HIDDEN_TOPIC}


def test_topic_feedback_is_saved_as_simple_json(tmp_path):
    set_topic_feedback(tmp_path, 3, PROMOTED_TOPIC)

    contents = json.loads((tmp_path / "topic_feedback.json").read_text())

    assert contents == {"3": PROMOTED_TOPIC}


def test_invalid_topic_feedback_is_rejected(tmp_path):
    with pytest.raises(ValueError, match="Unsupported topic feedback"):
        set_topic_feedback(tmp_path, 1, "maybe")


def test_malformed_topic_feedback_does_not_break_the_dashboard(tmp_path):
    (tmp_path / "topic_feedback.json").write_text("not json")

    assert load_topic_feedback(tmp_path) == {}


def test_apply_topic_feedback_hides_and_prioritizes_without_mutating_input():
    topics = pd.DataFrame(
        {
            "Topic": [1, 2, 3, 4],
            "LLM Title": ["One", "Two", "Three", "Four"],
        }
    )
    feedback = {2: HIDDEN_TOPIC, 3: PROMOTED_TOPIC, 4: PROMOTED_TOPIC}

    visible_topics = apply_topic_feedback(topics, feedback)

    assert visible_topics["Topic"].tolist() == [3, 4, 1]
    assert topics["Topic"].tolist() == [1, 2, 3, 4]
    assert visible_topics.columns.tolist() == topics.columns.tolist()


def test_apply_topic_feedback_can_keep_hidden_topics_for_management():
    topics = pd.DataFrame({"Topic": [1, 2, 3]})
    feedback = {1: HIDDEN_TOPIC, 2: PROMOTED_TOPIC}

    all_topics = apply_topic_feedback(topics, feedback, include_hidden=True)

    assert all_topics["Topic"].tolist() == [2, 3, 1]


def test_order_topic_ids_matches_dataframe_feedback_ordering():
    feedback = {2: HIDDEN_TOPIC, 3: PROMOTED_TOPIC}

    assert order_topic_ids([1, 2, 3, 4], feedback) == [3, 1, 4]
    assert order_topic_ids([1, 2, 3, 4], feedback, include_hidden=True) == [3, 1, 4, 2]


def test_topic_feedback_icons_make_saved_preferences_visible():
    feedback = {2: HIDDEN_TOPIC, 3: PROMOTED_TOPIC}

    assert get_topic_feedback_icon(1, feedback) == ""
    assert get_topic_feedback_icon(2, feedback) == "🚫"
    assert get_topic_feedback_icon(3, feedback) == "⭐"
