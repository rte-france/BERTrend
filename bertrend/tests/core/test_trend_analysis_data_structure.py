#  Copyright (c) 2024, RTE (https://www.rte-france.com)
#  See AUTHORS.txt
#  SPDX-License-Identifier: MPL-2.0
#  This file is part of BERTrend.

import pytest
from pydantic import ValidationError

from bertrend.trend_analysis.data_structure import (
    TopicSummary,
    TopicSummaryList,
    PotentialImplications,
    EvolutionScenario,
    TopicInterconnexions,
    Drivers,
    SignalAnalysis,
)


def test_topic_summary_creation():
    """Test that a TopicSummary can be created with valid data."""
    topic_summary = TopicSummary(
        title="Test Topic",
        date="2023-01-01",
        key_developments=["Development 1", "Development 2"],
        description="This is a test topic description",
        novelty="This is what's new",
    )

    assert topic_summary.title == "Test Topic"
    assert topic_summary.date == "2023-01-01"
    assert topic_summary.key_developments == ["Development 1", "Development 2"]
    assert topic_summary.description == "This is a test topic description"
    assert topic_summary.novelty == "This is what's new"


def test_topic_summary_validation():
    """Test that validation works correctly for TopicSummary."""
    # Test missing required fields
    with pytest.raises(ValidationError):
        TopicSummary(
            title="Test Topic",
            date="2023-01-01",
            key_developments=["Development 1"],
            description="Description",
            # Missing novelty
        )

    with pytest.raises(ValidationError):
        TopicSummary(
            title="Test Topic",
            date="2023-01-01",
            # Missing key_developments
            description="Description",
            novelty="Novelty",
        )


def test_topic_summary_list_creation():
    """Test that a TopicSummaryList can be created with valid data."""
    topic_summary1 = TopicSummary(
        title="Topic 1",
        date="2023-01-01",
        key_developments=["Development 1"],
        description="Description 1",
        novelty="Novelty 1",
    )

    topic_summary2 = TopicSummary(
        title="Topic 2",
        date="2023-02-01",
        key_developments=["Development 2"],
        description="Description 2",
        novelty="Novelty 2",
    )

    topic_summary_list = TopicSummaryList(
        topic_summary_by_time_period=[topic_summary1, topic_summary2]
    )

    assert len(topic_summary_list.topic_summary_by_time_period) == 2
    assert topic_summary_list.topic_summary_by_time_period[0].title == "Topic 1"
    assert topic_summary_list.topic_summary_by_time_period[1].title == "Topic 2"


def test_potential_implications_creation():
    """Test that a PotentialImplications can be created with valid data."""
    implications = PotentialImplications(
        long_term_implications=["Long term 1", "Long term 2"],
        short_term_implications=["Short term 1", "Short term 2"],
    )

    assert implications.long_term_implications == ["Long term 1", "Long term 2"]
    assert implications.short_term_implications == ["Short term 1", "Short term 2"]


def test_evolution_scenario_creation():
    """Test that an EvolutionScenario can be created with valid data."""
    scenario = EvolutionScenario(
        optimistic_scenario_description="Optimistic description",
        optimistic_scenario_points=["Optimistic point 1", "Optimistic point 2"],
        pessimistic_scenario_description="Pessimistic description",
        pessimistic_scenario_points=["Pessimistic point 1", "Pessimistic point 2"],
    )

    assert scenario.optimistic_scenario_description == "Optimistic description"
    assert scenario.optimistic_scenario_points == [
        "Optimistic point 1",
        "Optimistic point 2",
    ]
    assert scenario.pessimistic_scenario_description == "Pessimistic description"
    assert scenario.pessimistic_scenario_points == [
        "Pessimistic point 1",
        "Pessimistic point 2",
    ]


def test_topic_interconnexions_creation():
    """Test that a TopicInterconnexions can be created with valid data."""
    interconnexions = TopicInterconnexions(
        interconnexions=["Interconnexion 1", "Interconnexion 2"],
        ripple_effects=["Ripple effect 1", "Ripple effect 2"],
    )

    assert interconnexions.interconnexions == ["Interconnexion 1", "Interconnexion 2"]
    assert interconnexions.ripple_effects == ["Ripple effect 1", "Ripple effect 2"]


def test_drivers_creation():
    """Test that a Drivers can be created with valid data."""
    drivers = Drivers(
        drivers=["Driver 1", "Driver 2"],
        inhibitors=["Inhibitor 1", "Inhibitor 2"],
    )

    assert drivers.drivers == ["Driver 1", "Driver 2"]
    assert drivers.inhibitors == ["Inhibitor 1", "Inhibitor 2"]


def test_signal_analysis_creation():
    """Test that a SignalAnalysis can be created with valid data."""
    implications = PotentialImplications(
        long_term_implications=["Long term 1"],
        short_term_implications=["Short term 1"],
    )

    scenario = EvolutionScenario(
        optimistic_scenario_description="Optimistic description",
        optimistic_scenario_points=["Optimistic point 1"],
        pessimistic_scenario_description="Pessimistic description",
        pessimistic_scenario_points=["Pessimistic point 1"],
    )

    interconnexions = TopicInterconnexions(
        interconnexions=["Interconnexion 1"],
        ripple_effects=["Ripple effect 1"],
    )

    drivers = Drivers(
        drivers=["Driver 1"],
        inhibitors=["Inhibitor 1"],
    )

    signal_analysis = SignalAnalysis(
        potential_implications=implications,
        evolution_scenario=scenario,
        topic_interconnexions=interconnexions,
        drivers_inhibitors=drivers,
    )

    assert signal_analysis.potential_implications == implications
    assert signal_analysis.evolution_scenario == scenario
    assert signal_analysis.topic_interconnexions == interconnexions
    assert signal_analysis.drivers_inhibitors == drivers
