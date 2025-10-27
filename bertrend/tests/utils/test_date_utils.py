#  Copyright (c) 2024, RTE (https://www.rte-france.com)
#  See AUTHORS.txt
#  SPDX-License-Identifier: MPL-2.0
#  This file is part of BERTrend.

from datetime import datetime

from bertrend_apps.common.date_utils import daterange


class TestDateRange:
    """Tests for the daterange function."""

    def test_daterange_basic(self):
        """Test basic daterange functionality with standard inputs."""
        start = datetime(2024, 1, 1)
        end = datetime(2024, 1, 10)
        ndays = 3

        result = list(daterange(start, end, ndays))

        assert len(result) == 3
        assert result[0] == (datetime(2024, 1, 1), datetime(2024, 1, 4))
        assert result[1] == (datetime(2024, 1, 4), datetime(2024, 1, 7))
        assert result[2] == (datetime(2024, 1, 7), datetime(2024, 1, 10))

    def test_daterange_exact_multiple(self):
        """Test when the date range is an exact multiple of ndays."""
        start = datetime(2024, 1, 1)
        end = datetime(2024, 1, 7)
        ndays = 3

        result = list(daterange(start, end, ndays))

        assert len(result) == 2
        assert result[0] == (datetime(2024, 1, 1), datetime(2024, 1, 4))
        assert result[1] == (datetime(2024, 1, 4), datetime(2024, 1, 7))

    def test_daterange_incomplete_last_period(self):
        """Test when the last period is incomplete."""
        start = datetime(2024, 1, 1)
        end = datetime(2024, 1, 11)
        ndays = 5

        result = list(daterange(start, end, ndays))

        assert len(result) == 2
        assert result[0] == (datetime(2024, 1, 1), datetime(2024, 1, 6))
        assert result[1] == (datetime(2024, 1, 6), datetime(2024, 1, 11))

    def test_daterange_single_period(self):
        """Test when the range is smaller than ndays."""
        start = datetime(2024, 1, 1)
        end = datetime(2024, 1, 3)
        ndays = 5

        result = list(daterange(start, end, ndays))

        assert len(result) == 1
        assert result[0] == (datetime(2024, 1, 1), datetime(2024, 1, 3))

    def test_daterange_one_day_period(self):
        """Test with ndays=1 (daily ranges)."""
        start = datetime(2024, 1, 1)
        end = datetime(2024, 1, 5)
        ndays = 1

        result = list(daterange(start, end, ndays))

        assert len(result) == 4
        assert result[0] == (datetime(2024, 1, 1), datetime(2024, 1, 2))
        assert result[1] == (datetime(2024, 1, 2), datetime(2024, 1, 3))
        assert result[2] == (datetime(2024, 1, 3), datetime(2024, 1, 4))
        assert result[3] == (datetime(2024, 1, 4), datetime(2024, 1, 5))

    def test_daterange_end_equals_start(self):
        """Test when end_date equals start_date (should return nothing)."""
        start = datetime(2024, 1, 1)
        end = datetime(2024, 1, 1)
        ndays = 5

        result = list(daterange(start, end, ndays))

        assert len(result) == 0

    def test_daterange_end_before_start(self):
        """Test when end_date is before start_date (should return nothing)."""
        start = datetime(2024, 1, 10)
        end = datetime(2024, 1, 5)
        ndays = 3

        result = list(daterange(start, end, ndays))

        assert len(result) == 0

    def test_daterange_large_period(self):
        """Test with a large period spanning multiple months."""
        start = datetime(2024, 1, 1)
        end = datetime(2024, 3, 1)
        ndays = 30

        result = list(daterange(start, end, ndays))

        assert len(result) == 2
        assert result[0] == (datetime(2024, 1, 1), datetime(2024, 1, 31))
        assert result[1] == (datetime(2024, 1, 31), datetime(2024, 3, 1))

    def test_daterange_with_time_component(self):
        """Test that time components are preserved."""
        start = datetime(2024, 1, 1, 12, 30, 0)
        end = datetime(2024, 1, 5, 14, 45, 0)
        ndays = 2

        result = list(daterange(start, end, ndays))

        assert len(result) == 3
        assert result[0] == (
            datetime(2024, 1, 1, 12, 30, 0),
            datetime(2024, 1, 3, 12, 30, 0),
        )
        assert result[1] == (
            datetime(2024, 1, 3, 12, 30, 0),
            datetime(2024, 1, 5, 12, 30, 0),
        )
        assert result[2] == (
            datetime(2024, 1, 5, 12, 30, 0),
            datetime(2024, 1, 5, 14, 45, 0),
        )

    def test_daterange_is_generator(self):
        """Test that daterange returns a generator (yields values)."""
        start = datetime(2024, 1, 1)
        end = datetime(2024, 1, 10)
        ndays = 3

        result = daterange(start, end, ndays)

        # Check it's a generator
        assert hasattr(result, "__iter__")
        assert hasattr(result, "__next__")

        # Get first value without consuming all
        first = next(result)
        assert first == (datetime(2024, 1, 1), datetime(2024, 1, 4))

    def test_daterange_leap_year(self):
        """Test daterange behavior around leap year dates."""
        start = datetime(2024, 2, 27)
        end = datetime(2024, 3, 2)
        ndays = 2

        result = list(daterange(start, end, ndays))

        # 2024 is a leap year, so Feb has 29 days
        assert len(result) == 2
        assert result[0] == (datetime(2024, 2, 27), datetime(2024, 2, 29))
        assert result[1] == (datetime(2024, 2, 29), datetime(2024, 3, 2))

    def test_daterange_year_boundary(self):
        """Test daterange crossing year boundaries."""
        start = datetime(2023, 12, 30)
        end = datetime(2024, 1, 3)
        ndays = 2

        result = list(daterange(start, end, ndays))

        assert len(result) == 2
        assert result[0] == (datetime(2023, 12, 30), datetime(2024, 1, 1))
        assert result[1] == (datetime(2024, 1, 1), datetime(2024, 1, 3))
