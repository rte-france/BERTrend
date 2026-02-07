# Mail Configuration Guide

BERTrend includes email functionality for sending newsletters automatically. This guide explains how to configure email
sending using Gmail OAuth and provides alternatives for other email providers.

## Purpose

The mail functionality in BERTrend is primarily used for:

- **Newsletter Distribution**: Automatically sending generated newsletters to configured recipients
- **Report Delivery**: Sending topic analysis reports and summaries via email
- **Automated Notifications**: Delivering scheduled content based on data feeds and topic modeling results

## Gmail OAuth Configuration

BERTrend uses Gmail API with OAuth 2.0 authentication for secure email sending. This requires setting up Google Cloud
credentials.

### 1. Creating Gmail Credentials

#### Step 1: Google Cloud Console Setup

1. Go to the [Google Cloud Console](https://console.cloud.google.com/)
2. Create a new project or select an existing one
3. Enable the Gmail API:
    - Navigate to "APIs & Services" > "Library"
    - Search for "Gmail API" and enable it

#### Step 2: Create OAuth 2.0 Credentials

1. Go to "APIs & Services" > "Credentials"
2. Click "Create Credentials" > "OAuth client ID"
3. Choose "Desktop application" as the application type
4. Give it a name (e.g., "BERTrend Newsletter")
5. Download the JSON file

#### Step 3: Configure BERTrend

1. Place the downloaded JSON file at: `bertrend_apps/config/gmail_credentials.json`
2. Ensure the file has the following structure:
   ```json
   {
     "installed": {
       "client_id": "your-client-id",
       "project_id": "your-project-id",
       "auth_uri": "https://accounts.google.com/o/oauth2/auth",
       "token_uri": "https://oauth2.googleapis.com/token",
       "auth_provider_x509_cert_url": "https://www.googleapis.com/oauth2/v1/certs",
       "client_secret": "your-client-secret",
       "redirect_uris": ["http://localhost"]
     }
   }
   ```

### 2. First-Time Authentication

When running BERTrend's newsletter functionality for the first time:

1. The system will automatically open a browser window
2. Sign in with the Gmail account you want to use for sending
3. Grant permissions for BERTrend to send emails
4. The system will create a `gmail_token.json` file automatically
5. Subsequent runs will use this token (refreshing as needed)

### 3. Configuration in Newsletter Settings

In your newsletter TOML configuration file, specify recipients:

```toml
[newsletter]
recipients = "['recipient1@example.com', 'recipient2@example.com']"
title = "Your Newsletter Title"
# ... other newsletter settings
```

## Using Alternative Email Solutions

If you prefer not to use Gmail or need to use a different email provider, you can modify the mail functionality.

### Modifying mail_utils.py

The file `bertrend_apps/common/mail_utils.py` contains the mail implementation. Here's how to adapt it for other
providers:

#### Option 1: SMTP-based Email (e.g., Outlook, Corporate Email)

Replace the Gmail API implementation with SMTP:

```python
import smtplib
from email.message import EmailMessage
from email.utils import COMMASPACE
import mimetypes
from pathlib import Path

# SMTP Configuration
SMTP_SERVER = "smtp.your-provider.com"  # e.g., "smtp.office365.com" for Outlook
SMTP_PORT = 587
SMTP_USERNAME = "your-email@domain.com"
SMTP_PASSWORD = "your-password"  # Consider using environment variables

def send_email_smtp(
    subject: str,
    recipients: list[str],
    content: str | Path,
    content_type: str = "html",
    file_name: str = None,
) -> None:
    """Send email using SMTP"""
    try:
        msg = EmailMessage()
        msg["Subject"] = subject
        msg["From"] = SMTP_USERNAME
        msg["To"] = COMMASPACE.join(recipients)
        
        # Handle file attachment or content
        if isinstance(content, Path) and content.is_file():
            # File attachment logic (similar to original)
            # ... (adapt the file attachment code from original)
        else:
            # Content as body
            subtype = "plain" if content_type.lower() in {"md", "text", "txt"} else content_type
            msg.set_content(content, subtype=subtype)
        
        # Send via SMTP
        with smtplib.SMTP(SMTP_SERVER, SMTP_PORT) as server:
            server.starttls()
            server.login(SMTP_USERNAME, SMTP_PASSWORD)
            server.send_message(msg)
            
        logger.debug("Email successfully sent via SMTP.")
        
    except Exception as err:
        logger.exception("SMTP error: ", err)
```

#### Option 2: Other Email APIs (SendGrid, Mailgun, etc.)

For SendGrid example:

```python
import sendgrid
from sendgrid.helpers.mail import Mail

SENDGRID_API_KEY = "your-sendgrid-api-key"

def send_email_sendgrid(
    subject: str,
    recipients: list[str],
    content: str,
    content_type: str = "html"
) -> None:
    """Send email using SendGrid API"""
    try:
        sg = sendgrid.SendGridAPIClient(api_key=SENDGRID_API_KEY)
        
        message = Mail(
            from_email='sender@example.com',
            to_emails=recipients,
            subject=subject,
            html_content=content if content_type == "html" else None,
            plain_text_content=content if content_type != "html" else None
        )
        
        response = sg.send(message)
        logger.debug(f"SendGrid email sent, status: {response.status_code}")
        
    except Exception as err:
        logger.exception("SendGrid error: ", err)
```

#### Option 3: Environment-based Configuration

Make the email backend configurable via environment variables:

```python
import os

EMAIL_BACKEND = os.getenv("BERTREND_EMAIL_BACKEND", "gmail")  # gmail, smtp, sendgrid

def send_email(credentials, subject: str, recipients: list[str], content, **kwargs):
    """Send email using configured backend"""
    if EMAIL_BACKEND == "gmail":
        return send_email_gmail(credentials, subject, recipients, content, **kwargs)
    elif EMAIL_BACKEND == "smtp":
        return send_email_smtp(subject, recipients, content, **kwargs)
    elif EMAIL_BACKEND == "sendgrid":
        return send_email_sendgrid(subject, recipients, content, **kwargs)
    else:
        raise ValueError(f"Unsupported email backend: {EMAIL_BACKEND}")
```

### Required Changes in Newsletter Code

When modifying mail_utils.py, ensure you also update the import and function calls in:

- `bertrend_apps/newsletters/__main__.py` (lines 37, 204, 208)

For non-Gmail solutions, you might not need the `get_credentials()` function, so adapt the newsletter code accordingly:

```python
# Instead of:
credentials = get_credentials()
send_email(credentials=credentials, ...)

# Use:
send_email(...)  # for SMTP or API-based solutions
```

## Security Considerations

### Gmail OAuth

- Keep `gmail_credentials.json` and `gmail_token.json` secure
- Add these files to your `.gitignore`
- Use project-specific Google Cloud projects for better isolation

### SMTP/API Solutions

- Use environment variables for sensitive credentials
- Consider using application-specific passwords
- Enable two-factor authentication on email accounts
- For production, use dedicated email service accounts

## Troubleshooting

### Common Gmail OAuth Issues

- **Token expired**: Delete `gmail_token.json` and re-authenticate
- **Insufficient permissions**: Ensure Gmail API is enabled and OAuth consent screen is configured
- **Rate limiting**: Gmail API has sending limits; consider batching for large recipient lists

### General Email Issues

- **Firewall/Network**: Ensure outbound connections are allowed (port 587 for SMTP, 443 for APIs)
- **Spam filters**: Configure SPF/DKIM records for better deliverability
- **Large attachments**: Consider file size limits (Gmail: 25MB, others vary)

## Testing Email Configuration

To test your email setup:

1. Create a simple test script:

```python
from bertrend.bertrend_apps.common import get_credentials, send_email

# For Gmail OAuth
credentials = get_credentials()
send_email(
    credentials=credentials,
    subject="BERTrend Test Email",
    recipients=["your-test-email@example.com"],
    content="<h1>Test successful!</h1>",
    content_type="html"
)
```

2. Run the newsletter functionality with a small test configuration to verify end-to-end functionality.