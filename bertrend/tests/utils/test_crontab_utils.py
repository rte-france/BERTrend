#  Copyright (c) 2024, RTE (https://www.rte-france.com)
#  See AUTHORS.txt
#  SPDX-License-Identifier: MPL-2.0
#  This file is part of BERTrend.

import pytest
import subprocess
from pathlib import Path
from unittest.mock import patch, MagicMock, call

from bertrend_apps.common.crontab_utils import CrontabSchedulerUtils


class TestCrontabSchedulerUtils:
    """Tests for CrontabSchedulerUtils class."""

    def setup_method(self):
        """Set up test fixtures."""
        self.scheduler = CrontabSchedulerUtils()

    @patch("subprocess.call")
    @patch("os.getenv")
    def test_add_job_to_crontab_success(self, mock_getenv, mock_subprocess_call):
        """Test successfully adding a job to crontab."""
        mock_getenv.return_value = "/home/user"
        mock_subprocess_call.return_value = 0

        result = self.scheduler.add_job_to_crontab(
            schedule="0 2 * * *",
            command="python test.py",
            env_vars="VAR=value",
        )

        assert result is True
        mock_subprocess_call.assert_called_once()
        call_args = mock_subprocess_call.call_args[0][0]
        assert "0 2 * * *" in call_args
        assert "python test.py" in call_args
        assert "VAR=value" in call_args

    @patch("subprocess.call")
    @patch("os.getenv")
    def test_add_job_to_crontab_failure(self, mock_getenv, mock_subprocess_call):
        """Test failure when adding a job to crontab."""
        mock_getenv.return_value = "/home/user"
        mock_subprocess_call.return_value = 1

        result = self.scheduler.add_job_to_crontab(
            schedule="0 2 * * *",
            command="python test.py",
        )

        assert result is False

    @patch("subprocess.call")
    @patch("os.getenv")
    def test_add_job_to_crontab_no_env_vars(self, mock_getenv, mock_subprocess_call):
        """Test adding a job without environment variables."""
        mock_getenv.return_value = "/home/user"
        mock_subprocess_call.return_value = 0

        result = self.scheduler.add_job_to_crontab(
            schedule="0 2 * * *",
            command="python test.py",
        )

        assert result is True
        call_args = mock_subprocess_call.call_args[0][0]
        # Should have empty env_vars
        assert "python test.py" in call_args

    @patch("subprocess.run")
    def test_check_cron_job_exists(self, mock_subprocess_run):
        """Test checking for an existing cron job."""
        mock_result = MagicMock()
        mock_result.stdout = "0 2 * * * python test.py\n30 3 * * * python other.py"
        mock_subprocess_run.return_value = mock_result

        result = self.scheduler._check_cron_job(r"python test\.py")

        assert result is True
        mock_subprocess_run.assert_called_once_with(
            ["crontab", "-l"], capture_output=True, text=True, check=True
        )

    @patch("subprocess.run")
    def test_check_cron_job_not_exists(self, mock_subprocess_run):
        """Test checking for a non-existent cron job."""
        mock_result = MagicMock()
        mock_result.stdout = "0 2 * * * python other.py"
        mock_subprocess_run.return_value = mock_result

        result = self.scheduler._check_cron_job(r"python test\.py")

        assert result is False

    @patch("subprocess.run")
    def test_check_cron_job_no_crontab(self, mock_subprocess_run):
        """Test checking when user has no crontab."""
        mock_subprocess_run.side_effect = subprocess.CalledProcessError(1, "crontab")

        result = self.scheduler._check_cron_job(r"python test\.py")

        assert result is False

    @patch("bertrend_apps.common.crontab_utils.CrontabSchedulerUtils._check_cron_job")
    @patch("subprocess.check_output")
    def test_remove_from_crontab_success(self, mock_check_output, mock_check_job):
        """Test successfully removing a job from crontab."""
        mock_check_job.return_value = True
        mock_check_output.return_value = 0

        result = self.scheduler._remove_from_crontab(r"python test\.py")

        assert result is True
        mock_check_output.assert_called_once()

    @patch("bertrend_apps.common.crontab_utils.CrontabSchedulerUtils._check_cron_job")
    def test_remove_from_crontab_not_found(self, mock_check_job):
        """Test removing a job that doesn't exist."""
        mock_check_job.return_value = False

        result = self.scheduler._remove_from_crontab(r"python test\.py")

        assert result is False

    @patch("bertrend_apps.common.crontab_utils.CrontabSchedulerUtils._check_cron_job")
    @patch("subprocess.check_output")
    def test_remove_from_crontab_error(self, mock_check_output, mock_check_job):
        """Test error handling when removing from crontab."""
        mock_check_job.return_value = True
        mock_check_output.side_effect = subprocess.CalledProcessError(1, "crontab")

        result = self.scheduler._remove_from_crontab(r"python test\.py")

        assert result is False

    @patch("bertrend_apps.common.crontab_utils.load_toml_config")
    @patch(
        "bertrend_apps.common.crontab_utils.CrontabSchedulerUtils.add_job_to_crontab"
    )
    @patch("bertrend_apps.common.crontab_utils.BERTREND_LOG_PATH", Path("/logs"))
    def test_schedule_scrapping(self, mock_add_job, mock_load_config):
        """Test scheduling a scraping job."""
        mock_config = {
            "data-feed": {
                "update_frequency": "0 2 * * *",
                "id": "test_feed",
            }
        }
        mock_load_config.return_value = mock_config
        mock_add_job.return_value = True

        feed_cfg = Path("/path/to/feed.toml")
        with patch.object(Path, "mkdir"):
            self.scheduler.schedule_scrapping(feed_cfg)

        mock_add_job.assert_called_once()
        call_args = mock_add_job.call_args
        assert call_args[0][0] == "0 2 * * *"
        assert "scrape-feed" in call_args[0][1]
        assert "test_feed" in call_args[0][1]

    @patch("bertrend_apps.common.crontab_utils.load_toml_config")
    @patch(
        "bertrend_apps.common.crontab_utils.CrontabSchedulerUtils.add_job_to_crontab"
    )
    @patch("bertrend_apps.common.crontab_utils.BERTREND_LOG_PATH", Path("/logs"))
    def test_schedule_scrapping_with_user(self, mock_add_job, mock_load_config):
        """Test scheduling a scraping job for a specific user."""
        mock_config = {
            "data-feed": {
                "update_frequency": "0 2 * * *",
                "id": "test_feed",
            }
        }
        mock_load_config.return_value = mock_config
        mock_add_job.return_value = True

        feed_cfg = Path("/path/to/feed.toml")
        with patch.object(Path, "mkdir"):
            self.scheduler.schedule_scrapping(feed_cfg, user="testuser")

        mock_add_job.assert_called_once()
        call_args = mock_add_job.call_args
        assert "users/testuser" in call_args[0][1]

    @patch("bertrend_apps.common.crontab_utils.load_toml_config")
    @patch(
        "bertrend_apps.common.crontab_utils.CrontabSchedulerUtils.add_job_to_crontab"
    )
    @patch("bertrend_apps.common.crontab_utils.BERTREND_LOG_PATH", Path("/logs"))
    @patch("bertrend_apps.common.crontab_utils.BEST_CUDA_DEVICE", "0")
    def test_schedule_newsletter(self, mock_add_job, mock_load_config):
        """Test scheduling a newsletter job."""
        mock_config = {
            "newsletter": {
                "update_frequency": "0 8 * * *",
                "id": "test_newsletter",
            }
        }
        mock_load_config.return_value = mock_config
        mock_add_job.return_value = True

        newsletter_cfg = Path("/path/to/newsletter.toml")
        feed_cfg = Path("/path/to/feed.toml")
        self.scheduler.schedule_newsletter(newsletter_cfg, feed_cfg, cuda_devices="0")

        mock_add_job.assert_called_once()
        call_args = mock_add_job.call_args
        assert call_args[0][0] == "0 8 * * *"
        assert "newsletters" in call_args[0][1]
        assert "CUDA_VISIBLE_DEVICES=0" in call_args[0][2]

    @patch(
        "bertrend_apps.common.crontab_utils.CrontabSchedulerUtils.add_job_to_crontab"
    )
    @patch("bertrend_apps.common.crontab_utils.BERTREND_LOG_PATH", Path("/logs"))
    @patch("bertrend_apps.common.crontab_utils.BEST_CUDA_DEVICE", "1")
    def test_schedule_training_for_user(self, mock_add_job):
        """Test scheduling a training job for a user."""
        mock_add_job.return_value = True

        with patch.object(Path, "mkdir"):
            result = self.scheduler.schedule_training_for_user(
                schedule="0 3 * * *", model_id="test_model", user="testuser"
            )

        assert result is True
        mock_add_job.assert_called_once()
        call_args = mock_add_job.call_args
        assert call_args[0][0] == "0 3 * * *"
        assert "train-new-model" in call_args[0][1]
        assert "testuser" in call_args[0][1]
        assert "test_model" in call_args[0][1]
        assert "CUDA_VISIBLE_DEVICES=1" in call_args[0][2]

    @patch(
        "bertrend_apps.common.crontab_utils.CrontabSchedulerUtils.add_job_to_crontab"
    )
    @patch("bertrend_apps.common.crontab_utils.BERTREND_LOG_PATH", Path("/logs"))
    def test_schedule_report_generation_with_auto_send(self, mock_add_job):
        """Test scheduling report generation when auto_send is enabled."""
        mock_add_job.return_value = True
        report_config = {
            "auto_send": True,
            "email_recipients": ["test@example.com"],
        }

        with patch.object(Path, "mkdir"):
            result = self.scheduler.schedule_report_generation_for_user(
                schedule="0 9 * * *",
                model_id="test_model",
                user="testuser",
                report_config=report_config,
            )

        assert result is True
        mock_add_job.assert_called_once()

    @patch(
        "bertrend_apps.common.crontab_utils.CrontabSchedulerUtils.add_job_to_crontab"
    )
    def test_schedule_report_generation_auto_send_disabled(self, mock_add_job):
        """Test that report generation is not scheduled when auto_send is disabled."""
        report_config = {
            "auto_send": False,
            "email_recipients": ["test@example.com"],
        }

        result = self.scheduler.schedule_report_generation_for_user(
            schedule="0 9 * * *",
            model_id="test_model",
            user="testuser",
            report_config=report_config,
        )

        assert result is False
        mock_add_job.assert_not_called()

    @patch(
        "bertrend_apps.common.crontab_utils.CrontabSchedulerUtils.add_job_to_crontab"
    )
    def test_schedule_report_generation_no_recipients(self, mock_add_job):
        """Test that report generation is not scheduled when no recipients are configured."""
        report_config = {
            "auto_send": True,
            "email_recipients": [],
        }

        result = self.scheduler.schedule_report_generation_for_user(
            schedule="0 9 * * *",
            model_id="test_model",
            user="testuser",
            report_config=report_config,
        )

        assert result is False
        mock_add_job.assert_not_called()

    @patch(
        "bertrend_apps.common.crontab_utils.CrontabSchedulerUtils._remove_from_crontab"
    )
    def test_remove_scrapping_for_user_with_user(self, mock_remove):
        """Test removing scraping job for a specific user."""
        mock_remove.return_value = True

        result = self.scheduler.remove_scrapping_for_user("test_feed", user="testuser")

        assert result is True
        mock_remove.assert_called_once()
        pattern = mock_remove.call_args[0][0]
        assert "users/testuser" in pattern
        assert "test_feed" in pattern

    @patch(
        "bertrend_apps.common.crontab_utils.CrontabSchedulerUtils._remove_from_crontab"
    )
    def test_remove_scrapping_for_user_without_user(self, mock_remove):
        """Test removing scraping job without user specification."""
        mock_remove.return_value = True

        result = self.scheduler.remove_scrapping_for_user("test_feed", user=None)

        assert result is True
        mock_remove.assert_called_once()
        pattern = mock_remove.call_args[0][0]
        assert "test_feed" in pattern
        assert "users" not in pattern

    @patch(
        "bertrend_apps.common.crontab_utils.CrontabSchedulerUtils._remove_from_crontab"
    )
    def test_remove_scheduled_training_for_user(self, mock_remove):
        """Test removing training job for a user."""
        mock_remove.return_value = True

        result = self.scheduler.remove_scheduled_training_for_user(
            "test_model", "testuser"
        )

        assert result is True
        mock_remove.assert_called_once()
        pattern = mock_remove.call_args[0][0]
        assert "train-new-model" in pattern
        assert "testuser" in pattern
        assert "test_model" in pattern

    @patch(
        "bertrend_apps.common.crontab_utils.CrontabSchedulerUtils._remove_from_crontab"
    )
    def test_remove_scheduled_training_no_user(self, mock_remove):
        """Test that removing training without user returns False."""
        result = self.scheduler.remove_scheduled_training_for_user("test_model", None)

        assert result is False
        mock_remove.assert_not_called()

    @patch(
        "bertrend_apps.common.crontab_utils.CrontabSchedulerUtils._remove_from_crontab"
    )
    def test_remove_scheduled_report_generation_for_user(self, mock_remove):
        """Test removing report generation job for a user."""
        mock_remove.return_value = True

        result = self.scheduler.remove_scheduled_report_generation_for_user(
            "test_model", "testuser"
        )

        assert result is True
        mock_remove.assert_called_once()
        pattern = mock_remove.call_args[0][0]
        assert "automated_report_generation" in pattern
        assert "testuser" in pattern
        assert "test_model" in pattern

    @patch("bertrend_apps.common.crontab_utils.CrontabSchedulerUtils._check_cron_job")
    def test_check_if_scrapping_active_with_user(self, mock_check):
        """Test checking if scraping is active for a specific user."""
        mock_check.return_value = True

        result = self.scheduler.check_if_scrapping_active_for_user(
            "test_feed", user="testuser"
        )

        assert result is True
        mock_check.assert_called_once()
        pattern = mock_check.call_args[0][0]
        assert "users/testuser" in pattern
        assert "test_feed" in pattern

    @patch("bertrend_apps.common.crontab_utils.CrontabSchedulerUtils._check_cron_job")
    def test_check_if_scrapping_active_without_user(self, mock_check):
        """Test checking if scraping is active without user specification."""
        mock_check.return_value = True

        result = self.scheduler.check_if_scrapping_active_for_user(
            "test_feed", user=None
        )

        assert result is True
        mock_check.assert_called_once()
        pattern = mock_check.call_args[0][0]
        assert "test_feed" in pattern

    @patch("bertrend_apps.common.crontab_utils.CrontabSchedulerUtils._check_cron_job")
    def test_check_if_learning_active_for_user(self, mock_check):
        """Test checking if learning is active for a user."""
        mock_check.return_value = True

        result = self.scheduler.check_if_learning_active_for_user(
            "test_model", "testuser"
        )

        assert result is True
        mock_check.assert_called_once()
        pattern = mock_check.call_args[0][0]
        assert "train-new-model" in pattern
        assert "testuser" in pattern
        assert "test_model" in pattern

    @patch("bertrend_apps.common.crontab_utils.CrontabSchedulerUtils._check_cron_job")
    def test_check_if_learning_active_no_user(self, mock_check):
        """Test that checking learning without user returns False."""
        result = self.scheduler.check_if_learning_active_for_user("test_model", None)

        assert result is False
        mock_check.assert_not_called()

    @patch("bertrend_apps.common.crontab_utils.CrontabSchedulerUtils._check_cron_job")
    def test_check_if_report_generation_active_for_user(self, mock_check):
        """Test checking if report generation is active for a user."""
        mock_check.return_value = True

        result = self.scheduler.check_if_report_generation_active_for_user(
            "test_model", "testuser"
        )

        assert result is True
        mock_check.assert_called_once()
        pattern = mock_check.call_args[0][0]
        assert "automated_report_generation" in pattern
        assert "testuser" in pattern
        assert "test_model" in pattern

    @patch("bertrend_apps.common.crontab_utils.CrontabSchedulerUtils._check_cron_job")
    def test_check_if_report_generation_active_no_user(self, mock_check):
        """Test that checking report generation without user returns False."""
        result = self.scheduler.check_if_report_generation_active_for_user(
            "test_model", None
        )

        assert result is False
        mock_check.assert_not_called()
