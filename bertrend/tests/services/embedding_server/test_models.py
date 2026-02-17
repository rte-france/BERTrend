#  Copyright (c) 2024-2026, RTE (https://www.rte-france.com)
#  See AUTHORS.txt
#  SPDX-License-Identifier: MPL-2.0
#  This file is part of BERTrend.

import pytest
from pydantic import ValidationError

from bertrend.services.embedding_server.models import InputText


def test_input_text_with_string():
    """Test InputText accepts a single string."""
    input_text = InputText(text="hello world", show_progress_bar=False)
    assert input_text.text == "hello world"
    assert input_text.show_progress_bar is False


def test_input_text_with_list():
    """Test InputText accepts a list of strings."""
    input_text = InputText(text=["hello", "world"], show_progress_bar=True)
    assert input_text.text == ["hello", "world"]
    assert input_text.show_progress_bar is True


def test_input_text_missing_fields():
    """Test InputText raises ValidationError when required fields are missing."""
    with pytest.raises(ValidationError):
        InputText()


def test_input_text_missing_show_progress_bar():
    """Test InputText raises ValidationError when show_progress_bar is missing."""
    with pytest.raises(ValidationError):
        InputText(text="hello")
