#  Copyright (c) 2024, RTE (https://www.rte-france.com)
#  See AUTHORS.txt
#  SPDX-License-Identifier: MPL-2.0
#  This file is part of BERTrend.

import base64
import os
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from email.utils import COMMASPACE
from pathlib import Path

# Gmail API utils
from google.auth.transport.requests import Request
from google.oauth2.credentials import Credentials
from google_auth_oauthlib.flow import InstalledAppFlow
from googleapiclient.discovery import build
from googleapiclient.errors import HttpError
from loguru import logger

from bertrend import BASE_DATA_PATH

SCOPES = ["https://mail.google.com/"]  # full access to mail API
FROM = "wattelse.ai@gmail.com"
TOKEN_PATH = BASE_DATA_PATH / "gmail_token.json"
DEFAULT_GMAIL_CREDENTIALS_PATH = (
    Path(__file__).parent.parent / "config" / "gmail_credentials.json"
)

# Ensures to write with +rw for both user and groups
os.umask(0o002)


def get_credentials(
    credentials_path: Path = DEFAULT_GMAIL_CREDENTIALS_PATH,
) -> Credentials:
    """Returns credentials for the user"""
    creds = None
    # The file token.json stores the user's access and refresh tokens, and is
    # created automatically when the authorization flow completes for the first
    # time.
    logger.debug(f"Gmail credentials path: {credentials_path}")
    logger.debug(f"Gmail token path: {TOKEN_PATH}")
    if TOKEN_PATH.exists():
        creds = Credentials.from_authorized_user_file(TOKEN_PATH, SCOPES)
    # If there are no (valid) credentials available, let the user log in.
    if not creds or not creds.valid:
        if creds and creds.expired and creds.refresh_token:
            creds.refresh(Request())
        else:
            flow = InstalledAppFlow.from_client_secrets_file(credentials_path, SCOPES)
            creds = flow.run_local_server(port=0)
        # Save the credentials for the next run
        with open(TOKEN_PATH, "w") as token:
            token.write(creds.to_json())
    return creds


def send_email(
    credentials: Credentials,
    subject: str,
    recipients: list[str],
    content: str,
    content_type="html",
):
    try:
        # Call the Gmail API
        service = build("gmail", "v1", credentials=credentials)
        message = MIMEMultipart()
        message["To"] = COMMASPACE.join(recipients)
        message["From"] = FROM
        message["Subject"] = subject

        # Record the MIME types of both parts - text/plain and text/html.
        part1 = MIMEText(
            content, "plain" if content_type in ["md", "text", "txt"] else content_type
        )
        message.attach(part1)

        encoded_message = base64.urlsafe_b64encode(message.as_bytes()).decode()

        create_message = {"raw": encoded_message}

        # Send email using the Gmail API
        email = (
            service.users().messages().send(userId="me", body=create_message).execute()
        )

        logger.debug(f"Email sent: {email}")
    except HttpError as error:
        logger.error(f"An error occurred: {error}")
