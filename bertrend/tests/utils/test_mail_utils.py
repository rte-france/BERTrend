#  Copyright (c) 2024, RTE (https://www.rte-france.com)
#  See AUTHORS.txt
#  SPDX-License-Identifier: MPL-2.0
#  This file is part of BERTrend.

from pathlib import Path
from unittest.mock import patch, MagicMock, mock_open
import tempfile

from bertrend_apps.common.mail_utils import (
    get_credentials,
    send_email,
    FROM,
    SCOPES,
)


class TestGetCredentials:
    """Tests for the get_credentials function."""

    @patch("bertrend_apps.common.mail_utils.TOKEN_PATH")
    @patch("bertrend_apps.common.mail_utils.Credentials")
    def test_get_credentials_with_valid_token(
        self, mock_credentials_class, mock_token_path
    ):
        """Test getting credentials when valid token file exists."""
        mock_token_path.exists.return_value = True
        mock_creds = MagicMock()
        mock_creds.valid = True
        mock_credentials_class.from_authorized_user_file.return_value = mock_creds

        result = get_credentials()

        assert result == mock_creds
        mock_credentials_class.from_authorized_user_file.assert_called_once_with(
            mock_token_path, SCOPES
        )

    @patch("bertrend_apps.common.mail_utils.TOKEN_PATH")
    @patch("bertrend_apps.common.mail_utils.Credentials")
    @patch("bertrend_apps.common.mail_utils.Request")
    def test_get_credentials_with_expired_token(
        self, mock_request, mock_credentials_class, mock_token_path
    ):
        """Test getting credentials when token is expired but has refresh token."""
        mock_token_path.exists.return_value = True
        mock_creds = MagicMock()
        mock_creds.valid = False
        mock_creds.expired = True
        mock_creds.refresh_token = "refresh_token"
        mock_credentials_class.from_authorized_user_file.return_value = mock_creds

        with patch("builtins.open", mock_open()) as mock_file:
            result = get_credentials()

        mock_creds.refresh.assert_called_once()
        assert mock_file.called

    @patch("bertrend_apps.common.mail_utils.TOKEN_PATH")
    @patch("bertrend_apps.common.mail_utils.Credentials")
    @patch("bertrend_apps.common.mail_utils.InstalledAppFlow")
    def test_get_credentials_no_token_file(
        self, mock_flow_class, mock_credentials_class, mock_token_path
    ):
        """Test getting credentials when no token file exists."""
        mock_token_path.exists.return_value = False
        mock_flow = MagicMock()
        mock_creds = MagicMock()
        mock_flow.run_local_server.return_value = mock_creds
        mock_flow_class.from_client_secrets_file.return_value = mock_flow

        with patch("builtins.open", mock_open()) as mock_file:
            result = get_credentials()

        mock_flow_class.from_client_secrets_file.assert_called_once()
        mock_flow.run_local_server.assert_called_once_with(port=0)
        assert mock_file.called
        assert result == mock_creds

    @patch("bertrend_apps.common.mail_utils.TOKEN_PATH")
    @patch("bertrend_apps.common.mail_utils.Credentials")
    @patch("bertrend_apps.common.mail_utils.InstalledAppFlow")
    def test_get_credentials_invalid_token_no_refresh(
        self, mock_flow_class, mock_credentials_class, mock_token_path
    ):
        """Test getting credentials when token is invalid and has no refresh token."""
        mock_token_path.exists.return_value = True
        mock_creds = MagicMock()
        mock_creds.valid = False
        mock_creds.expired = False
        mock_creds.refresh_token = None
        mock_credentials_class.from_authorized_user_file.return_value = mock_creds

        mock_flow = MagicMock()
        mock_new_creds = MagicMock()
        mock_flow.run_local_server.return_value = mock_new_creds
        mock_flow_class.from_client_secrets_file.return_value = mock_flow

        with patch("builtins.open", mock_open()) as mock_file:
            result = get_credentials()

        mock_flow.run_local_server.assert_called_once_with(port=0)
        assert result == mock_new_creds

    @patch("bertrend_apps.common.mail_utils.TOKEN_PATH")
    @patch("bertrend_apps.common.mail_utils.Credentials")
    @patch("bertrend_apps.common.mail_utils.InstalledAppFlow")
    def test_get_credentials_custom_path(
        self, mock_flow_class, mock_credentials_class, mock_token_path
    ):
        """Test getting credentials with custom credentials path."""
        mock_token_path.exists.return_value = False
        mock_flow = MagicMock()
        mock_creds = MagicMock()
        mock_flow.run_local_server.return_value = mock_creds
        mock_flow_class.from_client_secrets_file.return_value = mock_flow

        custom_path = Path("/custom/path/credentials.json")

        with patch("builtins.open", mock_open()) as mock_file:
            result = get_credentials(credentials_path=custom_path)

        mock_flow_class.from_client_secrets_file.assert_called_once_with(
            custom_path, SCOPES
        )


class TestSendEmail:
    """Tests for the send_email function."""

    @patch("bertrend_apps.common.mail_utils.build")
    def test_send_email_with_text_content(self, mock_build):
        """Test sending email with text content."""
        mock_service = MagicMock()
        mock_build.return_value = mock_service
        mock_creds = MagicMock()

        send_email(
            credentials=mock_creds,
            subject="Test Subject",
            recipients=["test@example.com"],
            content="Test content",
            content_type="text",
        )

        mock_build.assert_called_once_with("gmail", "v1", credentials=mock_creds)
        mock_service.users().messages().send.assert_called_once()

    @patch("bertrend_apps.common.mail_utils.build")
    def test_send_email_with_html_content(self, mock_build):
        """Test sending email with HTML content."""
        mock_service = MagicMock()
        mock_build.return_value = mock_service
        mock_creds = MagicMock()

        send_email(
            credentials=mock_creds,
            subject="HTML Email",
            recipients=["test@example.com"],
            content="<h1>Test</h1>",
            content_type="html",
        )

        mock_build.assert_called_once()
        mock_service.users().messages().send.assert_called_once()

    @patch("bertrend_apps.common.mail_utils.build")
    def test_send_email_multiple_recipients(self, mock_build):
        """Test sending email to multiple recipients."""
        mock_service = MagicMock()
        mock_build.return_value = mock_service
        mock_creds = MagicMock()

        recipients = ["test1@example.com", "test2@example.com", "test3@example.com"]

        send_email(
            credentials=mock_creds,
            subject="Multi Recipients",
            recipients=recipients,
            content="Test content",
        )

        mock_build.assert_called_once()
        mock_service.users().messages().send.assert_called_once()

    @patch("bertrend_apps.common.mail_utils.build")
    def test_send_email_with_file_attachment(self, mock_build):
        """Test sending email with file attachment."""
        mock_service = MagicMock()
        mock_build.return_value = mock_service
        mock_creds = MagicMock()

        # Create a temporary file
        with tempfile.NamedTemporaryFile(mode="w", delete=False, suffix=".txt") as f:
            f.write("Test file content")
            temp_file_path = Path(f.name)

        try:
            send_email(
                credentials=mock_creds,
                subject="Email with Attachment",
                recipients=["test@example.com"],
                content=temp_file_path,
            )

            mock_build.assert_called_once()
            mock_service.users().messages().send.assert_called_once()
        finally:
            # Clean up
            if temp_file_path.exists():
                temp_file_path.unlink()

    @patch("bertrend_apps.common.mail_utils.build")
    def test_send_email_with_custom_filename(self, mock_build):
        """Test sending email with file attachment and custom filename."""
        mock_service = MagicMock()
        mock_build.return_value = mock_service
        mock_creds = MagicMock()

        # Create a temporary file
        with tempfile.NamedTemporaryFile(mode="w", delete=False, suffix=".txt") as f:
            f.write("Test file content")
            temp_file_path = Path(f.name)

        try:
            send_email(
                credentials=mock_creds,
                subject="Email with Custom Filename",
                recipients=["test@example.com"],
                content=temp_file_path,
                file_name="custom_report.txt",
            )

            mock_build.assert_called_once()
            mock_service.users().messages().send.assert_called_once()
        finally:
            # Clean up
            if temp_file_path.exists():
                temp_file_path.unlink()

    @patch("bertrend_apps.common.mail_utils.build")
    @patch("bertrend_apps.common.mail_utils.logger")
    def test_send_email_http_error(self, mock_logger, mock_build):
        """Test send_email handles HttpError gracefully."""
        from googleapiclient.errors import HttpError

        mock_service = MagicMock()
        mock_build.return_value = mock_service
        mock_service.users().messages().send.side_effect = HttpError(
            resp=MagicMock(status=400), content=b"Error"
        )
        mock_creds = MagicMock()

        send_email(
            credentials=mock_creds,
            subject="Test",
            recipients=["test@example.com"],
            content="Test",
        )

        # Should log error
        mock_logger.error.assert_called()

    @patch("bertrend_apps.common.mail_utils.build")
    def test_send_email_file_not_found(self, mock_build):
        """Test send_email with non-existent Path (treated as string content)."""
        mock_service = MagicMock()
        mock_build.return_value = mock_service
        mock_creds = MagicMock()

        non_existent_path = Path("/non/existent/file.txt")

        send_email(
            credentials=mock_creds,
            subject="Test",
            recipients=["test@example.com"],
            content=non_existent_path,
        )

        # Should send successfully (Path is converted to string since is_file() returns False)
        mock_service.users().messages().send.assert_called_once()

    @patch("bertrend_apps.common.mail_utils.build")
    @patch("bertrend_apps.common.mail_utils.logger")
    def test_send_email_unexpected_error(self, mock_logger, mock_build):
        """Test send_email handles unexpected errors gracefully."""
        mock_service = MagicMock()
        mock_build.return_value = mock_service
        mock_service.users().messages().send.side_effect = Exception("Unexpected error")
        mock_creds = MagicMock()

        send_email(
            credentials=mock_creds,
            subject="Test",
            recipients=["test@example.com"],
            content="Test",
        )

        # Should log exception
        mock_logger.exception.assert_called()

    @patch("bertrend_apps.common.mail_utils.build")
    def test_send_email_from_address(self, mock_build):
        """Test that FROM address is correctly set."""
        mock_service = MagicMock()
        mock_build.return_value = mock_service
        mock_creds = MagicMock()

        send_email(
            credentials=mock_creds,
            subject="Test",
            recipients=["test@example.com"],
            content="Test",
        )

        # Verify the FROM constant is used
        assert FROM == "wattelse.ai@gmail.com"
        mock_build.assert_called_once()

    @patch("bertrend_apps.common.mail_utils.build")
    def test_send_email_content_type_markdown(self, mock_build):
        """Test sending email with markdown content type."""
        mock_service = MagicMock()
        mock_build.return_value = mock_service
        mock_creds = MagicMock()

        send_email(
            credentials=mock_creds,
            subject="Markdown Email",
            recipients=["test@example.com"],
            content="# Markdown Content",
            content_type="md",
        )

        mock_build.assert_called_once()
        mock_service.users().messages().send.assert_called_once()

    @patch("bertrend_apps.common.mail_utils.build")
    @patch("bertrend_apps.common.mail_utils.logger")
    def test_send_email_success_logging(self, mock_logger, mock_build):
        """Test that successful email send is logged."""
        mock_service = MagicMock()
        mock_build.return_value = mock_service
        mock_creds = MagicMock()

        send_email(
            credentials=mock_creds,
            subject="Test",
            recipients=["test@example.com"],
            content="Test",
        )

        # Should log success
        mock_logger.debug.assert_called()
