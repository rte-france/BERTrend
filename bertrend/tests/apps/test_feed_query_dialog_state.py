#  Copyright (c) 2024-2026, RTE (https://www.rte-france.com)
#  See AUTHORS.txt
#  SPDX-License-Identifier: MPL-2.0
#  This file is part of BERTrend.

"""Tests for feed dialog session-state reset (PR review: cancel must not leak)."""

from pathlib import Path
from unittest.mock import patch

from bertrend.bertrend_apps.prospective_demo import feeds_config as fc
from bertrend.bertrend_apps.prospective_demo.feeds_config import (
    _clear_feed_query_dialog_state,
    _feed_query_dialog_keys,
    open_feed_monitoring_dialog,
)


def test_feed_query_dialog_keys_for_new_and_existing_feeds():
    assert _feed_query_dialog_keys(None) == (
        "feed_query_new",
        "feed_monitoring_brief_new",
    )
    assert _feed_query_dialog_keys({"id": "feed_llm"}) == (
        "feed_query_feed_llm",
        "feed_monitoring_brief_feed_llm",
    )


def test_clear_feed_query_dialog_state_drops_stale_query_and_brief():
    config = {"id": "feed_llm", "query": "saved query"}
    query_key, brief_key = _feed_query_dialog_keys(config)
    state = {
        query_key: "unsaved edit from cancelled dialog",
        brief_key: "stale brief",
        "unrelated": True,
    }

    with patch.object(fc.st, "session_state", state):
        _clear_feed_query_dialog_state(config)

    assert query_key not in state
    assert brief_key not in state
    assert state["unrelated"] is True


def test_open_feed_monitoring_dialog_clears_before_opening_dialog():
    """Cancel-path fix: every open clears keys before edit_feed_monitoring runs."""
    config = {"id": "feed_llm", "query": "saved query"}
    query_key, brief_key = _feed_query_dialog_keys(config)
    state = {
        query_key: "stale generated query",
        brief_key: "stale brief",
    }
    seen = {}

    def capture_edit(cfg):
        # Keys must already be gone when the dialog body starts (re-seed from config).
        seen["query_present"] = query_key in state
        seen["brief_present"] = brief_key in state
        seen["config"] = cfg

    with (
        patch.object(fc.st, "session_state", state),
        patch.object(fc, "edit_feed_monitoring", side_effect=capture_edit) as mock_edit,
    ):
        open_feed_monitoring_dialog(config)

    mock_edit.assert_called_once_with(config)
    assert seen["query_present"] is False
    assert seen["brief_present"] is False
    assert seen["config"] is config


def test_generate_write_survives_when_clear_is_not_called_again():
    """Mid-dialog: generate updates the query key; only open/OK clear it."""
    query_key, brief_key = _feed_query_dialog_keys(None)
    state = {}

    with patch.object(fc.st, "session_state", state):
        _clear_feed_query_dialog_state(None)
        assert query_key not in state
        # Generate path in edit_feed_monitoring writes the same key helper produces.
        state[query_key] = '("generated" OR "query")'
        # A dialog re-run does not call _clear_feed_query_dialog_state again.
        assert state[query_key] == '("generated" OR "query")'
        assert brief_key not in state

    source = Path(fc.__file__).read_text(encoding="utf-8")
    assert "st.session_state[query_state_key] = generate_google_news_query(" in source
    assert "if query_state_key not in st.session_state:" in source


def test_configure_sources_wires_add_and_edit_through_open_helper():
    """Structural: ADD/EDIT must use open_feed_monitoring_dialog, not edit directly."""
    source = Path(fc.__file__).read_text(encoding="utf-8")
    assert "open_feed_monitoring_dialog()" in source
    assert '(EDIT_ICON, open_feed_monitoring_dialog, "secondary")' in source
    # Ensure the old direct edit call sites for open are gone from configure path.
    configure = source.split("def configure_information_sources")[1].split(
        "def toggle_icon"
    )[0]
    assert "edit_feed_monitoring()" not in configure
    assert "edit_feed_monitoring," not in configure
