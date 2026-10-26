#  Copyright (c) 2024-2026, RTE (https://www.rte-france.com)
#  See AUTHORS.txt
#  SPDX-License-Identifier: MPL-2.0
#  This file is part of BERTrend.

import json
from pathlib import Path
from typing import Literal

import pandas as pd
from loguru import logger

PROMOTED_TOPIC = "promoted"
HIDDEN_TOPIC = "hidden"
TOPIC_FEEDBACK_FILE = "topic_feedback.json"

# Literal values must stay in sync with PROMOTED_TOPIC / HIDDEN_TOPIC above
# (typing.Literal cannot reference those constants).
TopicFeedback = Literal["promoted", "hidden"]
VALID_TOPIC_FEEDBACK = {PROMOTED_TOPIC, HIDDEN_TOPIC}


def load_topic_feedback(model_path: Path) -> dict[int, TopicFeedback]:
    """Load the user's topic feedback for one monitored model."""
    feedback_path = model_path / TOPIC_FEEDBACK_FILE
    if not feedback_path.exists():
        return {}

    try:
        raw_feedback = json.loads(feedback_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        logger.warning(f"Unable to load topic feedback from {feedback_path}: {exc}")
        return {}

    if not isinstance(raw_feedback, dict):
        logger.warning(f"Ignoring invalid topic feedback in {feedback_path}")
        return {}

    feedback = {}
    for topic_id, status in raw_feedback.items():
        try:
            normalized_topic_id = int(topic_id)
        except (TypeError, ValueError):
            continue
        if status in VALID_TOPIC_FEEDBACK:
            feedback[normalized_topic_id] = status
    return feedback


def save_topic_feedback(model_path: Path, feedback: dict[int, TopicFeedback]) -> None:
    """Save the user's topic feedback for one monitored model."""
    model_path.mkdir(parents=True, exist_ok=True)
    normalized_feedback = {
        str(int(topic_id)): status
        for topic_id, status in sorted(feedback.items())
        if status in VALID_TOPIC_FEEDBACK
    }
    feedback_path = model_path / TOPIC_FEEDBACK_FILE
    feedback_path.write_text(
        json.dumps(normalized_feedback, indent=2) + "\n", encoding="utf-8"
    )


def set_topic_feedback(
    model_path: Path,
    topic_id: int,
    status: TopicFeedback | None,
) -> dict[int, TopicFeedback]:
    """Set or clear feedback for one topic and return the updated mapping."""
    if status is not None and status not in VALID_TOPIC_FEEDBACK:
        raise ValueError(f"Unsupported topic feedback: {status}")

    feedback = load_topic_feedback(model_path)
    normalized_topic_id = int(topic_id)
    if status is None:
        feedback.pop(normalized_topic_id, None)
    else:
        feedback[normalized_topic_id] = status
    save_topic_feedback(model_path, feedback)
    return feedback


def _feedback_rank(topic_id: int, feedback: dict[int, TopicFeedback]) -> int:
    status = feedback.get(int(topic_id))
    if status == PROMOTED_TOPIC:
        return 0
    if status == HIDDEN_TOPIC:
        return 2
    return 1


def get_topic_feedback_icon(topic_id: int, feedback: dict[int, TopicFeedback]) -> str:
    """Return the compact icon used to expose topic feedback in the UI."""
    status = feedback.get(int(topic_id))
    if status == PROMOTED_TOPIC:
        return "⭐"
    if status == HIDDEN_TOPIC:
        return "🚫"
    return ""


def order_topic_ids(
    topic_ids: list[int],
    feedback: dict[int, TopicFeedback],
    *,
    include_hidden: bool = False,
) -> list[int]:
    """Order promoted topics first and hidden topics last or omit them."""
    visible_topic_ids = [
        topic_id
        for topic_id in topic_ids
        if include_hidden or feedback.get(int(topic_id)) != HIDDEN_TOPIC
    ]
    return sorted(
        visible_topic_ids,
        key=lambda topic_id: _feedback_rank(topic_id, feedback),
    )


def apply_topic_feedback(
    topics: pd.DataFrame,
    feedback: dict[int, TopicFeedback],
    *,
    include_hidden: bool = False,
    topic_column: str = "Topic",
) -> pd.DataFrame:
    """Return a copy filtered and ordered according to saved topic feedback."""
    if topics.empty:
        return topics.copy()
    if topic_column not in topics:
        raise KeyError(f"Missing topic column: {topic_column}")

    result = topics.copy()
    if not include_hidden:
        hidden_topic_ids = {
            topic_id for topic_id, status in feedback.items() if status == HIDDEN_TOPIC
        }
        result = result[~result[topic_column].isin(hidden_topic_ids)]

    ordered_positions = sorted(
        range(len(result)),
        key=lambda position: _feedback_rank(
            result.iloc[position][topic_column], feedback
        ),
    )
    return result.iloc[ordered_positions].copy()
