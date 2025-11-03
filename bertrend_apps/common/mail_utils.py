#  Copyright (c) 2024, RTE (https://www.rte-france.com)
#  See AUTHORS.txt
#  SPDX-License-Identifier: MPL-2.0
#  This file is part of BERTrend.

import base64
import os
from email.message import EmailMessage
import mimetypes
from pathlib import Path

# Gmail API utils
from google.auth.transport.requests import Request
from google.oauth2.credentials import Credentials
from google_auth_oauthlib.flow import InstalledAppFlow
from googleapiclient.discovery import build
from googleapiclient.errors import HttpError
from loguru import logger

from bertrend import BASE_PATH

COMMASPACE = ", "


SCOPES = ["https://mail.google.com/"]  # full access to mail API
FROM = "wattelse.ai@gmail.com"
TOKEN_PATH = BASE_PATH / "gmail_token.json"
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
    credentials,
    subject: str,
    recipients: list[str],
    content: str | Path,
    content_type: str = "html",
    file_name: str = None,
) -> None:
    """
    Send an e-mail with Gmail API.

    If *content* is a valid path it is attached as a file (with an optional filename),
    otherwise it is used as the message body.
    """
    try:
        service = build("gmail", "v1", credentials=credentials)

        msg = EmailMessage()
        msg["Subject"] = subject
        msg["From"] = FROM
        msg["To"] = COMMASPACE.join(recipients)

        # ------------------------------------------------------------------ #
        # 1. Attach a file if *content* points to an existing file            #
        # ------------------------------------------------------------------ #
        if isinstance(content, Path) and content.is_file():
            file_path = Path(content)

            # Body text so the message is not “empty”
            msg.set_content(
                f"Please find the report {file_path.name if not file_name else file_name} attached.",
                subtype="plain",
            )

            mime_type, _ = mimetypes.guess_type(file_path)
            mime_type = mime_type or "application/octet-stream"
            maintype, subtype = mime_type.split("/", 1)

            with file_path.open("rb") as fp:
                msg.add_attachment(
                    fp.read(),
                    maintype=maintype,
                    subtype=subtype,
                    filename=file_path.name if not file_name else file_name,
                )

        # ------------------------------------------------------------------ #
        # 2. Otherwise use *content* as the body                             #
        # ------------------------------------------------------------------ #
        else:
            subtype = (
                "plain"
                if content_type.lower() in {"md", "text", "txt"}
                else content_type
            )
            # Convert Path to string if needed (e.g., for non-existent paths)
            content_str = str(content) if isinstance(content, Path) else content
            msg.set_content(content_str, subtype=subtype)

        # ------------------------------------------------------------------ #
        # 3. Send through Gmail API                                          #
        # ------------------------------------------------------------------ #
        raw_message = base64.urlsafe_b64encode(msg.as_bytes()).decode()
        service.users().messages().send(
            userId="me", body={"raw": raw_message}
        ).execute()

        logger.debug("E-mail successfully sent.")

    except HttpError as err:
        logger.error("Gmail API error: ", err)
    except FileNotFoundError:
        logger.error("File not found: ", content)
    except Exception as err:
        logger.exception("Unexpected error: ", err)
