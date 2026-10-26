#  Copyright (c) 2024-2026, RTE (https://www.rte-france.com)
#  See AUTHORS.txt
#  SPDX-License-Identifier: MPL-2.0
#  This file is part of BERTrend.
import jsonlines
import pandas as pd
import pytest

from bertrend.bertrend_apps.prospective_demo import (
    LLM_TOPIC_DESCRIPTION_COLUMN,
    LLM_TOPIC_TITLE_COLUMN,
    STRONG_SIGNALS,
    URLS_COLUMN,
    WEAK_SIGNALS,
    automated_report_generation,
)
from bertrend.bertrend_apps.prospective_demo.automated_report_generation import (
    create_detailed_newsletter_automated,
    load_signal_data,
)
from bertrend.bertrend_apps.prospective_demo.report_generation_utils import (
    render_html_report,
)
from bertrend.trend_analysis.data_structure import TopicSummary, TopicSummaryList

REFERENCE_TS = pd.Timestamp("2026-10-24")


def _signals(topics: list[int]) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "Topic": pd.Series(topics, dtype="int64"),
            LLM_TOPIC_TITLE_COLUMN: [f"title {t}" for t in topics],
            LLM_TOPIC_DESCRIPTION_COLUMN: [f"description {t}" for t in topics],
            URLS_COLUMN: [[f"https://example.com/{t}"] for t in topics],
        }
    )


@pytest.fixture
def interpretation_dir(tmp_path, monkeypatch):
    monkeypatch.setattr(
        automated_report_generation,
        "get_model_interpretation_path",
        lambda user, model_id, reference_ts: tmp_path,
    )
    return tmp_path


def _write(directory, name: str, signals: pd.DataFrame, interpretation: list | None):
    signals.to_parquet(directory / f"{name}.parquet")
    if interpretation is not None:
        with jsonlines.open(directory / f"{name}_interpretation.jsonl", "w") as w:
            w.write_all(interpretation)


def test_empty_interpretation_file_does_not_break_the_report(interpretation_dir):
    # Every LLM analysis of the weak signals failed: the interpretation file is empty
    _write(interpretation_dir, WEAK_SIGNALS, _signals([1, 2]), [])
    _write(
        interpretation_dir,
        STRONG_SIGNALS,
        _signals([5]),
        [{"topic": 5, "summary": "s5", "analysis": "a5"}],
    )

    weak, strong = load_signal_data("u", "m", REFERENCE_TS)

    assert list(weak["Topic"]) == [1, 2]
    assert weak["summary"].isna().all() and weak["analysis"].isna().all()
    assert strong.loc[0, "summary"] == "s5" and strong.loc[0, "analysis"] == "a5"


def test_missing_interpretation_file(interpretation_dir):
    _write(interpretation_dir, WEAK_SIGNALS, _signals([1]), None)

    weak, strong = load_signal_data("u", "m", REFERENCE_TS)

    assert weak["summary"].isna().all() and weak["analysis"].isna().all()
    assert strong is None


def test_partial_interpretation(interpretation_dir):
    # Some topics were analysed, others failed and are absent from the file
    _write(
        interpretation_dir,
        WEAK_SIGNALS,
        _signals([1, 2]),
        [{"topic": 2, "summary": "s2", "analysis": "a2"}],
    )

    weak, _ = load_signal_data("u", "m", REFERENCE_TS)

    summaries = weak.set_index("Topic")["summary"]
    assert pd.isna(summaries[1])
    assert summaries[2] == "s2"


def test_signals_without_interpretation_are_reported(interpretation_dir):
    _write(interpretation_dir, WEAK_SIGNALS, _signals([1, 2]), [])
    weak, strong = load_signal_data("u", "m", REFERENCE_TS)

    newsletter = create_detailed_newsletter_automated(weak, strong, "m", REFERENCE_TS)

    assert [t.title for t in newsletter.topics] == ["title 1", "title 2"]
    assert all(
        t.topic_evolution is None and t.topic_analysis is None
        for t in newsletter.topics
    )


WARNING = {
    "en": "The detailed analysis of the topics could not be generated",
    "fr": "L'analyse détaillée des sujets n'a pas pu être générée",
}


@pytest.mark.parametrize("language", ["en", "fr"])
def test_report_warns_when_no_topic_has_an_analysis(interpretation_dir, language):
    # Signals exist, but every LLM analysis failed
    _write(interpretation_dir, WEAK_SIGNALS, _signals([1, 2]), [])
    _write(interpretation_dir, STRONG_SIGNALS, _signals([5]), [])
    weak, strong = load_signal_data("u", "m", REFERENCE_TS)

    newsletter = create_detailed_newsletter_automated(weak, strong, "m", REFERENCE_TS)

    assert newsletter.analysis_unavailable
    assert WARNING[language] in render_html_report(newsletter, language=language)


def test_report_does_not_warn_when_some_topics_have_an_analysis(interpretation_dir):
    # Only the most popular topics are analysed: the others lack an analysis normally
    summary = TopicSummaryList(
        topic_summary_by_time_period=[
            TopicSummary(
                title="t",
                date="2026-10-24",
                key_developments=[],
                description="d",
                novelty="n",
            )
        ]
    ).model_dump_json()
    _write(
        interpretation_dir,
        WEAK_SIGNALS,
        _signals([1, 2]),
        [{"topic": 2, "summary": summary, "analysis": None}],
    )
    weak, strong = load_signal_data("u", "m", REFERENCE_TS)

    newsletter = create_detailed_newsletter_automated(
        weak, strong, "m", REFERENCE_TS, {"topic_evolution": False}
    )

    assert not newsletter.analysis_unavailable
    assert WARNING["en"] not in render_html_report(newsletter, language="en")
