#  Copyright (c) 2024, RTE (https://www.rte-france.com)
#  See AUTHORS.txt
#  SPDX-License-Identifier: MPL-2.0
#  This file is part of BERTrend.

import pytest
from pathlib import Path
from unittest.mock import patch, MagicMock, Mock

from bertrend_apps.common.apscheduler_utils import (
    APSchedulerUtils,
    _request,
    _job_id_from_string,
    _list_jobs,
)


class TestHelperFunctions:
    """Tests for module-level helper functions."""

    @patch("bertrend_apps.common.apscheduler_utils._session")
    def test_request_success(self, mock_session):
        """Test successful HTTP request."""
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_session.request.return_value = mock_response

        result = _request("GET", "/test")

        assert result == mock_response
        mock_session.request.assert_called_once()

    @patch("bertrend_apps.common.apscheduler_utils._session")
    def test_request_with_json(self, mock_session):
        """Test HTTP request with JSON payload."""
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_session.request.return_value = mock_response

        data = {"key": "value"}
        result = _request("POST", "/test", json=data)

        mock_session.request.assert_called_once()
        call_kwargs = mock_session.request.call_args[1]
        assert call_kwargs["json"] == data

    def test_job_id_from_string_deterministic(self):
        """Test that job_id generation is deterministic."""
        input_str = "test_command"

        id1 = _job_id_from_string(input_str)
        id2 = _job_id_from_string(input_str)

        assert id1 == id2
        assert id1.startswith("job_")
        assert len(id1) == 14  # "job_" + 10 hex chars

    def test_job_id_from_string_different_inputs(self):
        """Test that different inputs produce different job IDs."""
        id1 = _job_id_from_string("command1")
        id2 = _job_id_from_string("command2")

        assert id1 != id2

    @patch("bertrend_apps.common.apscheduler_utils._request")
    def test_list_jobs_success(self, mock_request):
        """Test successfully listing jobs."""
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = [
            {"job_id": "job1", "name": "Test Job 1"},
            {"job_id": "job2", "name": "Test Job 2"},
        ]
        mock_request.return_value = mock_response

        result = _list_jobs()

        assert len(result) == 2
        assert result[0]["job_id"] == "job1"
        mock_request.assert_called_once_with("GET", "/jobs")

    @patch("bertrend_apps.common.apscheduler_utils._request")
    def test_list_jobs_failure(self, mock_request):
        """Test listing jobs when request fails."""
        mock_response = MagicMock()
        mock_response.status_code = 500
        mock_response.text = "Server error"
        mock_request.return_value = mock_response

        result = _list_jobs()

        assert result == []

    @patch("bertrend_apps.common.apscheduler_utils._request")
    def test_list_jobs_json_error(self, mock_request):
        """Test listing jobs when JSON parsing fails."""
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.side_effect = Exception("JSON error")
        mock_request.return_value = mock_response

        result = _list_jobs()

        assert result == []


class TestAPSchedulerUtils:
    """Tests for APSchedulerUtils class."""

    @patch("bertrend_apps.common.apscheduler_utils._request")
    def test_init_success(self, mock_request):
        """Test successful initialization."""
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_request.return_value = mock_response

        scheduler = APSchedulerUtils()

        mock_request.assert_called_once_with("GET", "/health")

    @patch("bertrend_apps.common.apscheduler_utils._request")
    def test_init_service_unavailable(self, mock_request):
        """Test initialization when service is unavailable."""
        mock_request.side_effect = Exception("Connection error")

        # Should not raise, just log error
        scheduler = APSchedulerUtils()

        assert scheduler is not None

    @patch("bertrend_apps.common.apscheduler_utils._request")
    def test_find_jobs_success(self, mock_request):
        """Test successfully finding jobs."""
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "matches_found": 2,
            "jobs": [
                {"job_id": "job1", "name": "Test 1"},
                {"job_id": "job2", "name": "Test 2"},
            ],
        }
        mock_request.return_value = mock_response

        patterns = {"user": "testuser", "model_id": "test_model"}
        result = APSchedulerUtils.find_jobs(patterns)

        assert len(result) == 2
        assert "job1" in result
        assert "job2" in result

    @patch("bertrend_apps.common.apscheduler_utils._request")
    def test_find_jobs_no_matches(self, mock_request):
        """Test finding jobs when no matches exist."""
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "matches_found": 0,
            "jobs": [],
        }
        mock_request.return_value = mock_response

        patterns = {"user": "testuser"}
        result = APSchedulerUtils.find_jobs(patterns)

        assert result == []

    @patch("bertrend_apps.common.apscheduler_utils._request")
    def test_find_jobs_error(self, mock_request):
        """Test finding jobs when request fails."""
        mock_response = MagicMock()
        mock_response.status_code = 500
        mock_response.text = "Server error"
        mock_request.return_value = mock_response

        patterns = {"user": "testuser"}

        with pytest.raises(Exception, match="Failed to find jobs"):
            APSchedulerUtils.find_jobs(patterns)

    @patch("bertrend_apps.common.apscheduler_utils._request")
    def test_find_jobs_match_all_false(self, mock_request):
        """Test finding jobs with match_all=False."""
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "matches_found": 1,
            "jobs": [{"job_id": "job1", "name": "Test"}],
        }
        mock_request.return_value = mock_response

        patterns = {"user": "testuser"}
        result = APSchedulerUtils.find_jobs(patterns, match_all=False)

        call_args = mock_request.call_args[1]["json"]
        assert call_args["match_all"] is False

    @patch("bertrend_apps.common.apscheduler_utils._request")
    def test_remove_jobs_success(self, mock_request):
        """Test successfully removing jobs."""
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_request.return_value = mock_response

        job_ids = ["job1", "job2"]
        APSchedulerUtils.remove_jobs(job_ids)

        assert mock_request.call_count == 2
        mock_request.assert_any_call("DELETE", "/jobs/job1")
        mock_request.assert_any_call("DELETE", "/jobs/job2")

    @patch("bertrend_apps.common.apscheduler_utils._request")
    def test_remove_jobs_partial_failure(self, mock_request):
        """Test removing jobs when some deletions fail."""

        def side_effect(method, path):
            mock_resp = MagicMock()
            if "job1" in path:
                mock_resp.status_code = 200
            else:
                mock_resp.status_code = 404
                mock_resp.text = "Not found"
            return mock_resp

        mock_request.side_effect = side_effect

        job_ids = ["job1", "job2"]
        # Should not raise, just log errors
        APSchedulerUtils.remove_jobs(job_ids)

        assert mock_request.call_count == 2

    @patch("bertrend_apps.common.apscheduler_utils._request")
    def test_add_job_to_crontab_success(self, mock_request):
        """Test successfully adding a job."""
        # Mock health check
        mock_health_response = MagicMock()
        mock_health_response.status_code = 200

        # Mock job creation
        mock_job_response = MagicMock()
        mock_job_response.status_code = 201

        mock_request.side_effect = [mock_health_response, mock_job_response]

        scheduler = APSchedulerUtils()
        result = scheduler.add_job_to_crontab(
            schedule="0 2 * * *",
            command="/test-command",
            env_vars="VAR=value",
        )

        assert result is True
        assert mock_request.call_count == 2

    @patch("bertrend_apps.common.apscheduler_utils._request")
    def test_add_job_to_crontab_with_kwargs(self, mock_request):
        """Test adding a job with command kwargs."""
        mock_health_response = MagicMock()
        mock_health_response.status_code = 200

        mock_job_response = MagicMock()
        mock_job_response.status_code = 200

        mock_request.side_effect = [mock_health_response, mock_job_response]

        scheduler = APSchedulerUtils()
        command_kwargs = {
            "method": "POST",
            "json_data": {"key": "value"},
        }
        result = scheduler.add_job_to_crontab(
            schedule="0 2 * * *",
            command="/test",
            command_kwargs=command_kwargs,
        )

        assert result is True

    @patch("bertrend_apps.common.apscheduler_utils._request")
    def test_add_job_to_crontab_failure(self, mock_request):
        """Test adding a job when request fails."""
        mock_health_response = MagicMock()
        mock_health_response.status_code = 200

        mock_job_response = MagicMock()
        mock_job_response.status_code = 400
        mock_job_response.text = "Bad request"
        mock_job_response.json.return_value = {"detail": "Invalid data"}

        mock_request.side_effect = [mock_health_response, mock_job_response]

        scheduler = APSchedulerUtils()
        result = scheduler.add_job_to_crontab(
            schedule="invalid",
            command="/test",
        )

        assert result is False

    @patch("bertrend_apps.common.apscheduler_utils.load_toml_config")
    @patch("bertrend_apps.common.apscheduler_utils._request")
    def test_schedule_scrapping(self, mock_request, mock_load_config):
        """Test scheduling a scraping job."""
        mock_config = {
            "data-feed": {
                "update_frequency": "0 2 * * *",
                "id": "test_feed",
            }
        }
        mock_load_config.return_value = mock_config

        mock_health_response = MagicMock()
        mock_health_response.status_code = 200

        mock_job_response = MagicMock()
        mock_job_response.status_code = 201

        mock_request.side_effect = [mock_health_response, mock_job_response]

        scheduler = APSchedulerUtils()
        feed_cfg = Path("/path/to/feed.toml")
        scheduler.schedule_scrapping(feed_cfg, user="testuser")

        # Verify job creation was called
        assert mock_request.call_count == 2

    @patch("bertrend_apps.common.apscheduler_utils.load_toml_config")
    @patch("bertrend_apps.common.apscheduler_utils._request")
    def test_schedule_newsletter(self, mock_request, mock_load_config):
        """Test scheduling a newsletter job."""
        mock_config = {
            "newsletter": {
                "update_frequency": "0 8 * * *",
                "id": "test_newsletter",
            }
        }
        mock_load_config.return_value = mock_config

        mock_health_response = MagicMock()
        mock_health_response.status_code = 200

        mock_job_response = MagicMock()
        mock_job_response.status_code = 200

        mock_request.side_effect = [mock_health_response, mock_job_response]

        scheduler = APSchedulerUtils()
        newsletter_cfg = Path("/path/to/newsletter.toml")
        feed_cfg = Path("/path/to/feed.toml")
        scheduler.schedule_newsletter(newsletter_cfg, feed_cfg, cuda_devices="1")

        assert mock_request.call_count == 2

    @patch("bertrend_apps.common.apscheduler_utils._request")
    def test_schedule_training_for_user(self, mock_request):
        """Test scheduling a training job."""
        mock_health_response = MagicMock()
        mock_health_response.status_code = 200

        mock_job_response = MagicMock()
        mock_job_response.status_code = 200

        mock_request.side_effect = [mock_health_response, mock_job_response]

        scheduler = APSchedulerUtils()
        result = scheduler.schedule_training_for_user(
            schedule="0 3 * * *",
            model_id="test_model",
            user="testuser",
        )

        assert result is True

    @patch("bertrend_apps.common.apscheduler_utils._request")
    def test_schedule_report_generation_with_auto_send(self, mock_request):
        """Test scheduling report generation when auto_send is enabled."""
        mock_health_response = MagicMock()
        mock_health_response.status_code = 200

        mock_job_response = MagicMock()
        mock_job_response.status_code = 200

        mock_request.side_effect = [mock_health_response, mock_job_response]

        scheduler = APSchedulerUtils()
        report_config = {
            "auto_send": True,
            "email_recipients": ["test@example.com"],
        }
        result = scheduler.schedule_report_generation_for_user(
            schedule="0 9 * * *",
            model_id="test_model",
            user="testuser",
            report_config=report_config,
        )

        assert result is True

    @patch("bertrend_apps.common.apscheduler_utils._request")
    def test_schedule_report_generation_auto_send_disabled(self, mock_request):
        """Test that report generation is not scheduled when auto_send is disabled."""
        mock_health_response = MagicMock()
        mock_health_response.status_code = 200
        mock_request.return_value = mock_health_response

        scheduler = APSchedulerUtils()
        report_config = {
            "auto_send": False,
            "email_recipients": ["test@example.com"],
        }
        result = scheduler.schedule_report_generation_for_user(
            schedule="0 9 * * *",
            model_id="test_model",
            user="testuser",
            report_config=report_config,
        )

        assert result is False
        # Only health check should be called
        assert mock_request.call_count == 1

    @patch("bertrend_apps.common.apscheduler_utils._request")
    def test_schedule_report_generation_no_recipients(self, mock_request):
        """Test that report generation is not scheduled when no recipients."""
        mock_health_response = MagicMock()
        mock_health_response.status_code = 200
        mock_request.return_value = mock_health_response

        scheduler = APSchedulerUtils()
        report_config = {
            "auto_send": True,
            "email_recipients": [],
        }
        result = scheduler.schedule_report_generation_for_user(
            schedule="0 9 * * *",
            model_id="test_model",
            user="testuser",
            report_config=report_config,
        )

        assert result is False

    @patch("bertrend_apps.common.apscheduler_utils.APSchedulerUtils.find_jobs")
    @patch("bertrend_apps.common.apscheduler_utils.APSchedulerUtils.remove_jobs")
    @patch("bertrend_apps.common.apscheduler_utils._request")
    def test_remove_scrapping_for_user(self, mock_request, mock_remove, mock_find):
        """Test removing scraping job for a user."""
        mock_health_response = MagicMock()
        mock_health_response.status_code = 200
        mock_request.return_value = mock_health_response

        mock_find.return_value = ["job1"]

        scheduler = APSchedulerUtils()
        result = scheduler.remove_scrapping_for_user("test_feed", user="testuser")

        assert result is True
        mock_find.assert_called_once()
        mock_remove.assert_called_once_with(["job1"])

    @patch("bertrend_apps.common.apscheduler_utils.APSchedulerUtils.find_jobs")
    @patch("bertrend_apps.common.apscheduler_utils._request")
    def test_remove_scrapping_no_jobs_found(self, mock_request, mock_find):
        """Test removing scraping when no jobs are found."""
        mock_health_response = MagicMock()
        mock_health_response.status_code = 200
        mock_request.return_value = mock_health_response

        mock_find.return_value = []

        scheduler = APSchedulerUtils()
        result = scheduler.remove_scrapping_for_user("test_feed", user="testuser")

        # Returns True even when no jobs found (successful operation, nothing to remove)
        assert result is True

    @patch("bertrend_apps.common.apscheduler_utils.APSchedulerUtils.find_jobs")
    @patch("bertrend_apps.common.apscheduler_utils.APSchedulerUtils.remove_jobs")
    @patch("bertrend_apps.common.apscheduler_utils._request")
    def test_remove_scheduled_training_for_user(
        self, mock_request, mock_remove, mock_find
    ):
        """Test removing training job for a user."""
        mock_health_response = MagicMock()
        mock_health_response.status_code = 200
        mock_request.return_value = mock_health_response

        mock_find.return_value = ["job1"]

        scheduler = APSchedulerUtils()
        result = scheduler.remove_scheduled_training_for_user("test_model", "testuser")

        assert result is True
        mock_remove.assert_called_once()

    @patch("bertrend_apps.common.apscheduler_utils.APSchedulerUtils.find_jobs")
    @patch("bertrend_apps.common.apscheduler_utils._request")
    def test_check_if_scrapping_active(self, mock_request, mock_find):
        """Test checking if scraping is active."""
        mock_health_response = MagicMock()
        mock_health_response.status_code = 200
        mock_request.return_value = mock_health_response

        mock_find.return_value = ["job1"]

        scheduler = APSchedulerUtils()
        result = scheduler.check_if_scrapping_active_for_user(
            "test_feed", user="testuser"
        )

        assert result is True

    @patch("bertrend_apps.common.apscheduler_utils.APSchedulerUtils.find_jobs")
    @patch("bertrend_apps.common.apscheduler_utils._request")
    def test_check_if_scrapping_not_active(self, mock_request, mock_find):
        """Test checking if scraping is not active."""
        mock_health_response = MagicMock()
        mock_health_response.status_code = 200
        mock_request.return_value = mock_health_response

        mock_find.return_value = []

        scheduler = APSchedulerUtils()
        result = scheduler.check_if_scrapping_active_for_user(
            "test_feed", user="testuser"
        )

        assert result is False

    @patch("bertrend_apps.common.apscheduler_utils.APSchedulerUtils.find_jobs")
    @patch("bertrend_apps.common.apscheduler_utils._request")
    def test_check_if_learning_active(self, mock_request, mock_find):
        """Test checking if learning is active."""
        mock_health_response = MagicMock()
        mock_health_response.status_code = 200
        mock_request.return_value = mock_health_response

        mock_find.return_value = ["job1"]

        scheduler = APSchedulerUtils()
        result = scheduler.check_if_learning_active_for_user("test_model", "testuser")

        assert result is True

    @patch("bertrend_apps.common.apscheduler_utils.APSchedulerUtils.find_jobs")
    @patch("bertrend_apps.common.apscheduler_utils._request")
    def test_check_if_report_generation_active(self, mock_request, mock_find):
        """Test checking if report generation is active."""
        mock_health_response = MagicMock()
        mock_health_response.status_code = 200
        mock_request.return_value = mock_health_response

        mock_find.return_value = ["job1"]

        scheduler = APSchedulerUtils()
        result = scheduler.check_if_report_generation_active_for_user(
            "test_model", "testuser"
        )

        assert result is True

    @patch("bertrend_apps.common.apscheduler_utils.APSchedulerUtils.find_jobs")
    @patch("bertrend_apps.common.apscheduler_utils._request")
    def test_check_if_report_generation_not_active(self, mock_request, mock_find):
        """Test checking if report generation is not active."""
        mock_health_response = MagicMock()
        mock_health_response.status_code = 200
        mock_request.return_value = mock_health_response

        mock_find.return_value = []

        scheduler = APSchedulerUtils()
        result = scheduler.check_if_report_generation_active_for_user(
            "test_model", "testuser"
        )

        assert result is False
