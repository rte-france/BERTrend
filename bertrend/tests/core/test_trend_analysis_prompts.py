#  Copyright (c) 2024, RTE (https://www.rte-france.com)
#  See AUTHORS.txt
#  SPDX-License-Identifier: MPL-2.0
#  This file is part of BERTrend.

import pytest
from unittest.mock import patch, mock_open
import json

from bertrend.trend_analysis.prompts import (
    get_prompt,
    save_html_output,
    fill_html_template,
    SIGNAL_INTRO,
    SIGNAL_INSTRUCTIONS,
    TOPIC_SUMMARY_PROMPT,
)
from bertrend.trend_analysis.data_structure import (
    TopicSummary,
    TopicSummaryList,
    PotentialImplications,
    EvolutionScenario,
    TopicInterconnexions,
    Drivers,
    SignalAnalysis,
)


def test_get_prompt_weak_signal():
    """Test that get_prompt returns the correct weak signal prompt."""
    summary_from_first_prompt = json.dumps({"test": "data"})

    # Test English prompt
    prompt_en = get_prompt(
        language="English",
        prompt_type="weak_signal",
        summary_from_first_prompt=summary_from_first_prompt,
    )

    # Check that the prompt contains the expected parts
    assert summary_from_first_prompt in prompt_en
    assert (
        SIGNAL_INTRO["en"].format(summary_from_first_prompt=summary_from_first_prompt)
        in prompt_en
    )
    assert SIGNAL_INSTRUCTIONS["en"] in prompt_en

    # Test French prompt
    prompt_fr = get_prompt(
        language="French",
        prompt_type="weak_signal",
        summary_from_first_prompt=summary_from_first_prompt,
    )

    # Check that the prompt contains the expected parts
    assert summary_from_first_prompt in prompt_fr
    assert (
        SIGNAL_INTRO["fr"].format(summary_from_first_prompt=summary_from_first_prompt)
        in prompt_fr
    )
    assert SIGNAL_INSTRUCTIONS["fr"] in prompt_fr


def test_get_prompt_topic_summary():
    """Test that get_prompt returns the correct topic summary prompt."""
    topic_number = 42
    content_summary = "This is a test content summary"

    # Test English prompt
    prompt_en = get_prompt(
        language="English",
        prompt_type="topic_summary",
        topic_number=topic_number,
        content_summary=content_summary,
    )

    # Check that the prompt contains the expected parts
    assert str(topic_number) in prompt_en
    assert content_summary in prompt_en
    assert (
        TOPIC_SUMMARY_PROMPT["en"].format(
            topic_number=topic_number, content_summary=content_summary
        )
        == prompt_en
    )

    # Test French prompt
    prompt_fr = get_prompt(
        language="French",
        prompt_type="topic_summary",
        topic_number=topic_number,
        content_summary=content_summary,
    )

    # Check that the prompt contains the expected parts
    assert str(topic_number) in prompt_fr
    assert content_summary in prompt_fr
    assert (
        TOPIC_SUMMARY_PROMPT["fr"].format(
            topic_number=topic_number, content_summary=content_summary
        )
        == prompt_fr
    )


def test_get_prompt_invalid_type():
    """Test that get_prompt raises an error for invalid prompt types."""
    with pytest.raises(ValueError):
        get_prompt(
            language="English",
            prompt_type="invalid_type",
        )


@patch("bertrend.trend_analysis.prompts.OUTPUT_PATH")
@patch("builtins.open", new_callable=mock_open)
def test_save_html_output(mock_file, mock_output_path):
    """Test that save_html_output correctly saves HTML output."""
    # Setup
    mock_output_path.__truediv__.return_value = "mocked_path/signal_llm.html"
    html_output = "<html><body>Test HTML</body></html>"

    # Call the function
    save_html_output(html_output)

    # Check that the file was opened with the correct path
    mock_output_path.__truediv__.assert_called_once_with("signal_llm.html")

    # Check that the file was written with the correct content
    mock_file.assert_called_once_with(
        "mocked_path/signal_llm.html", "w", encoding="utf-8"
    )
    mock_file().write.assert_called_once_with(html_output)


@patch("bertrend.trend_analysis.prompts.Environment")
def test_fill_html_template(mock_environment):
    """Test that fill_html_template correctly fills the HTML template."""
    # Setup mock template
    mock_template = mock_environment.return_value.get_template.return_value
    mock_template.render.return_value = "<html>\n<body>\nTest HTML\n</body>\n</html>\n"

    # Create test data
    topic_summary = TopicSummary(
        title="Test Topic",
        date="2023-01-01",
        key_developments=["Development 1"],
        description="Description",
        novelty="Novelty",
    )

    topic_summary_list = TopicSummaryList(topic_summary_by_time_period=[topic_summary])

    implications = PotentialImplications(
        long_term_implications=["Long term"],
        short_term_implications=["Short term"],
    )

    scenario = EvolutionScenario(
        optimistic_scenario_description="Optimistic",
        optimistic_scenario_points=["Optimistic point"],
        pessimistic_scenario_description="Pessimistic",
        pessimistic_scenario_points=["Pessimistic point"],
    )

    interconnexions = TopicInterconnexions(
        interconnexions=["Interconnexion"],
        ripple_effects=["Ripple effect"],
    )

    drivers = Drivers(
        drivers=["Driver"],
        inhibitors=["Inhibitor"],
    )

    signal_analysis = SignalAnalysis(
        potential_implications=implications,
        evolution_scenario=scenario,
        topic_interconnexions=interconnexions,
        drivers_inhibitors=drivers,
    )

    # Call the function with English language
    result_en = fill_html_template(
        topic_summary_list=topic_summary_list,
        signal_analysis=signal_analysis,
        language="en",
    )

    # Check that the template was rendered with the correct data
    mock_environment.return_value.get_template.assert_called_with(
        "signal_llm_template_en.html"
    )
    mock_template.render.assert_called_with(
        topic_summary_list=topic_summary_list,
        signal_analysis=signal_analysis,
    )

    # Check that newlines were removed
    assert "\n" not in result_en
    assert result_en == "<html><body>Test HTML</body></html>"

    # Reset mocks
    mock_environment.reset_mock()
    mock_template.render.reset_mock()

    # Call the function with French language
    result_fr = fill_html_template(
        topic_summary_list=topic_summary_list,
        signal_analysis=signal_analysis,
        language="fr",
    )

    # Check that the template was rendered with the correct data
    mock_environment.return_value.get_template.assert_called_with(
        "signal_llm_template_fr.html"
    )
    mock_template.render.assert_called_with(
        topic_summary_list=topic_summary_list,
        signal_analysis=signal_analysis,
    )

    # Check that newlines were removed
    assert "\n" not in result_fr
    assert result_fr == "<html><body>Test HTML</body></html>"
