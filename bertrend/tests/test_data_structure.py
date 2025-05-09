#  Copyright (c) 2024, RTE (https://www.rte-france.com)
#  See AUTHORS.txt
#  SPDX-License-Identifier: MPL-2.0
#  This file is part of BERTrend.

import pytest
from pydantic import ValidationError

from bertrend.topic_analysis.data_structure import TopicDescription


def test_topic_description_creation():
    """Test that a TopicDescription can be created with valid data."""
    topic = TopicDescription(
        title="Test Topic", description="This is a test topic description"
    )

    assert topic.title == "Test Topic"
    assert topic.description == "This is a test topic description"


def test_topic_description_validation():
    """Test that validation works correctly for TopicDescription."""
    # Test missing title
    with pytest.raises(ValidationError):
        TopicDescription(description="Missing title")

    # Test missing description
    with pytest.raises(ValidationError):
        TopicDescription(title="Missing description")

    # Test empty strings (these should be valid according to the model definition)
    topic = TopicDescription(title="", description="")
    assert topic.title == ""
    assert topic.description == ""


def test_topic_description_dict_conversion():
    """Test that a TopicDescription can be converted to and from a dict."""
    original = TopicDescription(
        title="Test Topic", description="This is a test topic description"
    )

    # Convert to dict
    topic_dict = original.dict()
    assert isinstance(topic_dict, dict)
    assert topic_dict["title"] == "Test Topic"
    assert topic_dict["description"] == "This is a test topic description"

    # Create from dict
    recreated = TopicDescription(**topic_dict)
    assert recreated.title == original.title
    assert recreated.description == original.description
