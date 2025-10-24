#  Copyright (c) 2024, RTE (https://www.rte-france.com)
#  See AUTHORS.txt
#  SPDX-License-Identifier: MPL-2.0
#  This file is part of BERTrend.

import tempfile
from pathlib import Path
from unittest.mock import Mock, patch, mock_open

from bertrend_apps.common.mail_utils import (
    get_credentials,
    send_email,
    SCOPES,
    FROM,
    TOKEN_PATH,
    DEFAULT_GMAIL_CREDENTIALS_PATH,
)


class TestConstants:
    """Test module constants"""

    def test_scopes_defined(self):
        """Test that SCOPES is properly defined"""
        assert SCOPES == ["https://mail.google.com/"]

    def test_from_email_defined(self):
        """Test that FROM email is defined"""
        assert FROM == "wattelse.ai@gmail.com"

    def test_paths_defined(self):
        """Test that paths are properly defined"""
        assert TOKEN_PATH is not None
        assert DEFAULT_GMAIL_CREDENTIALS_PATH is not None
        assert isinstance(TOKEN_PATH, Path)
        assert isinstance(DEFAULT_GMAIL_CREDENTIALS_PATH, Path)


class TestGetCredentials:
    """Test get_credentials function"""

    @patch("bertrend_apps.common.mail_utils.Credentials")
    @patch("bertrend_apps.common.mail_utils.TOKEN_PATH")
    def test_existing_valid_credentials(self, mock_token_path, mock_credentials_class):
        """Test loading existing valid credentials"""
        # Setup mocks
        mock_token_path.exists.return_value = True
        mock_creds = Mock()
        mock_creds.valid = True
        mock_credentials_class.from_authorized_user_file.return_value = mock_creds

        # Test
        result = get_credentials()

        # Assertions
        assert result == mock_creds
        mock_credentials_class.from_authorized_user_file.assert_called_once_with(
            mock_token_path, SCOPES
        )

    @patch("bertrend_apps.common.mail_utils.Credentials")
    @patch("bertrend_apps.common.mail_utils.TOKEN_PATH")
    @patch("bertrend_apps.common.mail_utils.Request")
    def test_expired_credentials_refresh(
        self, mock_request_class, mock_token_path, mock_credentials_class
    ):
        """Test refreshing expired credentials"""
        # Setup mocks
        mock_token_path.exists.return_value = True
        mock_creds = Mock()
        mock_creds.valid = False
        mock_creds.expired = True
        mock_creds.refresh_token = "refresh_token"
        mock_creds.to_json.return_value = '{"token": "data"}'
        mock_credentials_class.from_authorized_user_file.return_value = mock_creds
        mock_request = Mock()
        mock_request_class.return_value = mock_request

        with patch("builtins.open", mock_open()) as mock_file:
            result = get_credentials()

        # Assertions
        mock_creds.refresh.assert_called_once_with(mock_request)
        mock_file.assert_called_once_with(mock_token_path, "w")
        assert result == mock_creds

    @patch("bertrend_apps.common.mail_utils.Credentials")
    @patch("bertrend_apps.common.mail_utils.TOKEN_PATH")
    @patch("bertrend_apps.common.mail_utils.InstalledAppFlow")
    def test_no_valid_credentials_flow(
        self, mock_flow_class, mock_token_path, mock_credentials_class
    ):
        """Test OAuth flow when no valid credentials exist"""
        # Setup mocks
        mock_token_path.exists.return_value = False
        mock_flow = Mock()
        mock_creds = Mock()
        mock_creds.to_json.return_value = '{"token": "new_data"}'
        mock_flow.run_local_server.return_value = mock_creds
        mock_flow_class.from_client_secrets_file.return_value = mock_flow

        credentials_path = Path("/test/path/credentials.json")

        with patch("builtins.open", mock_open()) as mock_file:
            result = get_credentials(credentials_path)

        # Assertions
        mock_flow_class.from_client_secrets_file.assert_called_once_with(
            credentials_path, SCOPES
        )
        mock_flow.run_local_server.assert_called_once_with(port=0)
        mock_file.assert_called_once_with(mock_token_path, "w")
        assert result == mock_creds

    @patch("bertrend_apps.common.mail_utils.Credentials")
    @patch("bertrend_apps.common.mail_utils.TOKEN_PATH")
    def test_invalid_credentials_no_refresh_token(
        self, mock_token_path, mock_credentials_class
    ):
        """Test handling invalid credentials without refresh token"""
        # Setup mocks
        mock_token_path.exists.return_value = True
        mock_creds = Mock()
        mock_creds.valid = False
        mock_creds.expired = True
        mock_creds.refresh_token = None  # No refresh token
        mock_credentials_class.from_authorized_user_file.return_value = mock_creds

        with patch(
            "bertrend_apps.common.mail_utils.InstalledAppFlow"
        ) as mock_flow_class:
            mock_flow = Mock()
            mock_new_creds = Mock()
            mock_new_creds.to_json.return_value = '{"token": "new_data"}'
            mock_flow.run_local_server.return_value = mock_new_creds
            mock_flow_class.from_client_secrets_file.return_value = mock_flow

            with patch("builtins.open", mock_open()):
                result = get_credentials()

        # Should trigger OAuth flow since refresh failed
        assert result == mock_new_creds


class TestSendEmail:
    """Test send_email function"""

    def create_mock_credentials(self):
        """Helper to create mock credentials"""
        return Mock()

    @patch("bertrend_apps.common.mail_utils.build")
    def test_send_email_with_html_content(self, mock_build):
        """Test sending email with HTML content"""
        # Setup
        credentials = self.create_mock_credentials()
        mock_service = Mock()
        mock_build.return_value = mock_service
        mock_messages = Mock()
        mock_send = Mock()
        mock_execute = Mock()

        mock_service.users.return_value.messages.return_value = mock_messages
        mock_messages.send.return_value = mock_send
        mock_send.execute.return_value = mock_execute

        subject = "Test Subject"
        recipients = ["test@example.com"]
        content = "<h1>Test HTML Content</h1>"

        # Test
        send_email(credentials, subject, recipients, content, content_type="html")

        # Assertions
        mock_build.assert_called_once_with("gmail", "v1", credentials=credentials)
        mock_messages.send.assert_called_once()
        mock_send.execute.assert_called_once()

    @patch("bertrend_apps.common.mail_utils.build")
    def test_send_email_with_plain_content(self, mock_build):
        """Test sending email with plain text content"""
        # Setup
        credentials = self.create_mock_credentials()
        mock_service = Mock()
        mock_build.return_value = mock_service
        mock_messages = Mock()
        mock_send = Mock()
        mock_execute = Mock()

        mock_service.users.return_value.messages.return_value = mock_messages
        mock_messages.send.return_value = mock_send
        mock_send.execute.return_value = mock_execute

        subject = "Test Subject"
        recipients = ["test@example.com", "test2@example.com"]
        content = "Plain text content"

        # Test
        send_email(credentials, subject, recipients, content, content_type="text")

        # Assertions
        mock_build.assert_called_once_with("gmail", "v1", credentials=credentials)
        mock_messages.send.assert_called_once()
        mock_send.execute.assert_called_once()

    @patch("bertrend_apps.common.mail_utils.build")
    @patch("bertrend_apps.common.mail_utils.mimetypes")
    def test_send_email_with_file_attachment(self, mock_mimetypes, mock_build):
        """Test sending email with file attachment"""
        # Setup
        credentials = self.create_mock_credentials()
        mock_service = Mock()
        mock_build.return_value = mock_service
        mock_messages = Mock()
        mock_send = Mock()
        mock_execute = Mock()

        mock_service.users.return_value.messages.return_value = mock_messages
        mock_messages.send.return_value = mock_send
        mock_send.execute.return_value = mock_execute
        mock_mimetypes.guess_type.return_value = ("application/pdf", None)

        subject = "Test with Attachment"
        recipients = ["test@example.com"]

        # Create temporary file for testing
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".pdf", delete=False
        ) as tmp_file:
            tmp_file.write("PDF content")
            file_path = Path(tmp_file.name)

        try:
            # Test
            send_email(credentials, subject, recipients, file_path)

            # Assertions
            mock_build.assert_called_once_with("gmail", "v1", credentials=credentials)
            mock_messages.send.assert_called_once()
            mock_send.execute.assert_called_once()
            mock_mimetypes.guess_type.assert_called_once_with(file_path)

        finally:
            # Cleanup
            file_path.unlink()

    @patch("bertrend_apps.common.mail_utils.build")
    @patch("bertrend_apps.common.mail_utils.mimetypes")
    def test_send_email_with_custom_filename(self, mock_mimetypes, mock_build):
        """Test sending email with custom filename for attachment"""
        # Setup
        credentials = self.create_mock_credentials()
        mock_service = Mock()
        mock_build.return_value = mock_service
        mock_messages = Mock()
        mock_send = Mock()
        mock_execute = Mock()

        mock_service.users.return_value.messages.return_value = mock_messages
        mock_messages.send.return_value = mock_send
        mock_send.execute.return_value = mock_execute
        mock_mimetypes.guess_type.return_value = ("text/plain", None)

        subject = "Test with Custom Filename"
        recipients = ["test@example.com"]
        custom_filename = "custom_report.txt"

        # Create temporary file for testing
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".txt", delete=False
        ) as tmp_file:
            tmp_file.write("Text content")
            file_path = Path(tmp_file.name)

        try:
            # Test
            send_email(
                credentials, subject, recipients, file_path, file_name=custom_filename
            )

            # Assertions
            mock_build.assert_called_once()
            mock_messages.send.assert_called_once()
            mock_send.execute.assert_called_once()

        finally:
            # Cleanup
            file_path.unlink()

    @patch("bertrend_apps.common.mail_utils.build")
    @patch("bertrend_apps.common.mail_utils.logger")
    def test_send_email_http_error(self, mock_logger, mock_build):
        """Test handling of HttpError during email sending"""
        from googleapiclient.errors import HttpError

        # Setup
        credentials = self.create_mock_credentials()
        mock_service = Mock()
        mock_build.return_value = mock_service

        # Mock HttpError
        http_error = HttpError(Mock(), b'{"error": "test error"}')
        mock_service.users().messages().send().execute.side_effect = http_error

        subject = "Test Subject"
        recipients = ["test@example.com"]
        content = "Test content"

        # Test - should not raise exception
        send_email(credentials, subject, recipients, content)

        # Assertions
        mock_logger.error.assert_called_once_with("Gmail API error: ", http_error)

    @patch("bertrend_apps.common.mail_utils.build")
    @patch("bertrend_apps.common.mail_utils.logger")
    def test_send_email_file_not_found_error(self, mock_logger, mock_build):
        """Test handling of FileNotFoundError during email sending"""
        # Setup
        credentials = self.create_mock_credentials()
        mock_service = Mock()
        mock_build.return_value = mock_service

        subject = "Test Subject"
        recipients = ["test@example.com"]
        non_existent_file = Path("/non/existent/file.txt")

        # Mock the Path to appear as a file but raise FileNotFoundError when opened
        with (
            patch.object(Path, "is_file", return_value=True),
            patch.object(Path, "open", side_effect=FileNotFoundError()),
        ):
            # Test - should not raise exception
            send_email(credentials, subject, recipients, non_existent_file)

        # Assertions
        mock_logger.error.assert_called_once_with("File not found: ", non_existent_file)

    @patch("bertrend_apps.common.mail_utils.build")
    @patch("bertrend_apps.common.mail_utils.logger")
    def test_send_email_unexpected_error(self, mock_logger, mock_build):
        """Test handling of unexpected errors during email sending"""
        # Setup
        credentials = self.create_mock_credentials()
        mock_service = Mock()
        mock_build.return_value = mock_service

        # Mock unexpected error
        unexpected_error = ValueError("Unexpected error")
        mock_service.users().messages().send().execute.side_effect = unexpected_error

        subject = "Test Subject"
        recipients = ["test@example.com"]
        content = "Test content"

        # Test - should not raise exception
        send_email(credentials, subject, recipients, content)

        # Assertions
        mock_logger.exception.assert_called_once_with(
            "Unexpected error: ", unexpected_error
        )

    @patch("bertrend_apps.common.mail_utils.build")
    def test_send_email_multiple_recipients(self, mock_build):
        """Test sending email to multiple recipients"""
        # Setup
        credentials = self.create_mock_credentials()
        mock_service = Mock()
        mock_build.return_value = mock_service
        mock_messages = Mock()
        mock_send = Mock()
        mock_execute = Mock()

        mock_service.users.return_value.messages.return_value = mock_messages
        mock_messages.send.return_value = mock_send
        mock_send.execute.return_value = mock_execute

        subject = "Test Multiple Recipients"
        recipients = ["test1@example.com", "test2@example.com", "test3@example.com"]
        content = "Content for multiple recipients"

        # Test
        send_email(credentials, subject, recipients, content)

        # Assertions
        mock_build.assert_called_once_with("gmail", "v1", credentials=credentials)
        mock_messages.send.assert_called_once()
        mock_send.execute.assert_called_once()

    @patch("bertrend_apps.common.mail_utils.build")
    @patch("bertrend_apps.common.mail_utils.mimetypes")
    def test_send_email_unknown_mime_type(self, mock_mimetypes, mock_build):
        """Test sending email with file having unknown MIME type"""
        # Setup
        credentials = self.create_mock_credentials()
        mock_service = Mock()
        mock_build.return_value = mock_service
        mock_messages = Mock()
        mock_send = Mock()
        mock_execute = Mock()

        mock_service.users.return_value.messages.return_value = mock_messages
        mock_messages.send.return_value = mock_send
        mock_send.execute.return_value = mock_execute
        mock_mimetypes.guess_type.return_value = (None, None)  # Unknown MIME type

        subject = "Test Unknown MIME"
        recipients = ["test@example.com"]

        # Create temporary file with unknown extension
        with tempfile.NamedTemporaryFile(
            mode="wb", suffix=".unknown", delete=False
        ) as tmp_file:
            tmp_file.write(b"Unknown content")
            file_path = Path(tmp_file.name)

        try:
            # Test
            send_email(credentials, subject, recipients, file_path)

            # Assertions
            mock_build.assert_called_once()
            mock_messages.send.assert_called_once()
            mock_send.execute.assert_called_once()

        finally:
            # Cleanup
            file_path.unlink()


class TestIntegration:
    """Integration tests for mail utils"""

    @patch("bertrend_apps.common.mail_utils.build")
    @patch("bertrend_apps.common.mail_utils.Credentials")
    @patch("bertrend_apps.common.mail_utils.TOKEN_PATH")
    def test_full_email_workflow(
        self, mock_token_path, mock_credentials_class, mock_build
    ):
        """Test complete workflow from getting credentials to sending email"""
        # Setup credentials
        mock_token_path.exists.return_value = True
        mock_creds = Mock()
        mock_creds.valid = True
        mock_credentials_class.from_authorized_user_file.return_value = mock_creds

        # Setup email service
        mock_service = Mock()
        mock_build.return_value = mock_service
        mock_execute = Mock()
        mock_service.users().messages().send().execute = mock_execute

        # Test workflow
        credentials = get_credentials()
        send_email(
            credentials,
            "Integration Test",
            ["integration@example.com"],
            "Integration test content",
        )

        # Assertions
        assert credentials == mock_creds
        mock_build.assert_called_once_with("gmail", "v1", credentials=mock_creds)
        mock_execute.assert_called_once()
