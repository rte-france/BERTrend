#  Copyright (c) 2024, RTE (https://www.rte-france.com)
#  See AUTHORS.txt
#  SPDX-License-Identifier: MPL-2.0
#  This file is part of BERTrend.

import locale
from unittest.mock import patch

from bertrend_apps.common.scheduler_utils import SchedulerUtils


class TestSchedulerUtils:
    """Tests for SchedulerUtils static methods."""

    def test_generate_crontab_expression_format(self):
        """Test that generate_crontab_expression returns a valid crontab format."""
        result = SchedulerUtils.generate_crontab_expression(5)

        # Should have 5 parts: minute hour day_of_month month day_of_week
        parts = result.split()
        assert len(parts) == 5

        # Verify format
        minute, hour, days, month, day_of_week = parts

        # Minute should be 0, 10, 20, 30, 40, or 50
        assert minute in ["0", "10", "20", "30", "40", "50"]

        # Hour should be between 0 and 6
        assert 0 <= int(hour) <= 6

        # Days should be comma-separated list
        day_list = days.split(",")
        assert len(day_list) > 0

        # Month should be *
        assert month == "*"

        # Day of week should be *
        assert day_of_week == "*"

    def test_generate_crontab_expression_days_interval_1(self):
        """Test crontab expression with 1-day interval."""
        result = SchedulerUtils.generate_crontab_expression(1)

        parts = result.split()
        days = parts[2].split(",")

        # Should have days from 1 to 30 (every day)
        assert len(days) == 30
        assert "1" in days
        assert "30" in days  # range(1, 31, 1) includes 30

    def test_generate_crontab_expression_days_interval_10(self):
        """Test crontab expression with 10-day interval."""
        result = SchedulerUtils.generate_crontab_expression(10)

        parts = result.split()
        days = parts[2].split(",")

        # Should have days 1, 11, 21
        expected_days = ["1", "11", "21"]
        assert days == expected_days

    def test_generate_crontab_expression_days_interval_15(self):
        """Test crontab expression with 15-day interval."""
        result = SchedulerUtils.generate_crontab_expression(15)

        parts = result.split()
        days = parts[2].split(",")

        # Should have days 1, 16
        expected_days = ["1", "16"]
        assert days == expected_days

    def test_generate_crontab_expression_randomness(self):
        """Test that generate_crontab_expression produces variation due to randomness."""
        results = set()
        for _ in range(20):
            result = SchedulerUtils.generate_crontab_expression(5)
            results.add(result)

        # With random hour (0-6) and minute (6 choices), we should get some variation
        # Not all 20 results should be identical
        assert len(results) > 1

    @patch(
        "bertrend_apps.common.scheduler_utils.get_current_internationalization_language"
    )
    def test_get_understandable_cron_description_english(self, mock_get_lang):
        """Test cron description generation in English."""
        mock_get_lang.return_value = "en"

        cron_expression = "30 2 * * *"
        description = SchedulerUtils.get_understandable_cron_description(
            cron_expression
        )

        # Should contain time information
        assert isinstance(description, str)
        assert len(description) > 0
        # The description should mention 2:30 AM in some form
        assert "2:30" in description or "02:30" in description

    @patch(
        "bertrend_apps.common.scheduler_utils.get_current_internationalization_language"
    )
    def test_get_understandable_cron_description_french(self, mock_get_lang):
        """Test cron description generation in French."""
        mock_get_lang.return_value = "fr"

        cron_expression = "0 12 * * *"
        description = SchedulerUtils.get_understandable_cron_description(
            cron_expression
        )

        # Should contain time information
        assert isinstance(description, str)
        assert len(description) > 0
        # Should mention 12:00
        assert "12:00" in description or "12h" in description.lower()

    @patch(
        "bertrend_apps.common.scheduler_utils.get_current_internationalization_language"
    )
    def test_get_understandable_cron_description_complex_expression(
        self, mock_get_lang
    ):
        """Test cron description with a complex expression."""
        mock_get_lang.return_value = "en"

        # Every Monday at 9:00
        cron_expression = "0 9 * * 1"
        description = SchedulerUtils.get_understandable_cron_description(
            cron_expression
        )

        assert isinstance(description, str)
        assert len(description) > 0
        # Should mention Monday and 9:00
        assert "monday" in description.lower() or "mon" in description.lower()

    @patch(
        "bertrend_apps.common.scheduler_utils.get_current_internationalization_language"
    )
    def test_get_understandable_cron_description_restores_locale(self, mock_get_lang):
        """Test that locale is restored after cron description generation."""
        mock_get_lang.return_value = "en"

        # Save original locale
        original_locale = locale.setlocale(locale.LC_ALL)

        cron_expression = "0 12 * * *"
        SchedulerUtils.get_understandable_cron_description(cron_expression)

        # Check locale is restored
        current_locale = locale.setlocale(locale.LC_ALL)
        assert current_locale == original_locale

    @patch(
        "bertrend_apps.common.scheduler_utils.get_current_internationalization_language"
    )
    def test_get_understandable_cron_description_daily(self, mock_get_lang):
        """Test description for daily execution."""
        mock_get_lang.return_value = "en"

        cron_expression = "15 14 * * *"
        description = SchedulerUtils.get_understandable_cron_description(
            cron_expression
        )

        assert isinstance(description, str)
        # Should mention the time (14:15)
        assert "14:15" in description

    @patch(
        "bertrend_apps.common.scheduler_utils.get_current_internationalization_language"
    )
    def test_get_understandable_cron_description_specific_days(self, mock_get_lang):
        """Test description for specific days of month."""
        mock_get_lang.return_value = "en"

        # On the 1st, 15th, and 30th of the month
        cron_expression = "0 10 1,15,30 * *"
        description = SchedulerUtils.get_understandable_cron_description(
            cron_expression
        )

        assert isinstance(description, str)
        assert len(description) > 0

    def test_generate_crontab_expression_large_interval(self):
        """Test crontab expression with large interval (30 days)."""
        result = SchedulerUtils.generate_crontab_expression(30)

        parts = result.split()
        days = parts[2].split(",")

        # Should only have day 1
        assert days == ["1"]

    def test_generate_crontab_expression_hour_in_night_range(self):
        """Test that generated hour is in night range (0-6)."""
        # Run multiple times to check randomness stays in range
        for _ in range(50):
            result = SchedulerUtils.generate_crontab_expression(5)
            parts = result.split()
            hour = int(parts[1])
            assert 0 <= hour <= 6, f"Hour {hour} is outside expected range 0-6"

    def test_generate_crontab_expression_minute_rounded_to_10(self):
        """Test that generated minute is rounded to nearest 10."""
        valid_minutes = {"0", "10", "20", "30", "40", "50"}

        # Run multiple times to verify minute is always rounded
        for _ in range(50):
            result = SchedulerUtils.generate_crontab_expression(5)
            parts = result.split()
            minute = parts[0]
            assert minute in valid_minutes, f"Minute {minute} is not in valid set"
