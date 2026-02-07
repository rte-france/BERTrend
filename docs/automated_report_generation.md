# Automated Report Generation

This document explains how to configure and use the automated report generation feature in the prospective_demo application.

## Overview

The automated report generation feature allows you to automatically generate and send reports via email each time new data is processed. Reports are generated based on weak and strong signal analysis and sent to configured recipients.

## Features

- **Automatic Scheduling**: Reports are automatically scheduled when model training is activated
- **Email Delivery**: Reports are sent via email to configured recipients
- **Configurable Options**: Control what information is included in reports
- **Cron-based Execution**: Uses cron jobs for reliable scheduled execution

## Configuration

### 1. Model Configuration File

Each model has a configuration file located at:
```
~/.bertrend/config/users/{username}/{model_id}_analysis.toml
```

Add or modify the `report_config` section:

```toml
[report_config]
auto_send = true
email_recipients = ["user1@example.com", "user2@example.com"]
report_title = "Weekly AI Trends Report"
max_emerging_topics = 3
max_strong_topics = 5
```

**Configuration Options:**

- `auto_send` (boolean): Enable/disable automated report generation
  - `true`: Reports will be automatically generated and sent
  - `false`: No automatic reports (default)

- `email_recipients` (list of strings): Email addresses to receive reports
  - Must be valid email addresses
  - Can include multiple recipients

- `report_title` (string): Subject line for the email
  - If empty, defaults to "Automated Report - {model_id}"

- `max_emerging_topics` (integer): Maximum number of emerging (weak signal) topics to include in the report
  - Default: 3
  - Range: 1-20

- `max_strong_topics` (integer): Maximum number of strong signal topics to include in the report
  - Default: 5
  - Range: 1-20

### 2. Analysis Configuration

Control what information is included in the report:

```toml
[analysis_config]
topic_evolution = true
evolution_scenarios = true
multifactorial_analysis = true
```

**Options:**
- `topic_evolution`: Include topic evolution over time
- `evolution_scenarios`: Include potential evolution scenarios
- `multifactorial_analysis`: Include implications, interconnections, and drivers/inhibitors

### 3. GUI Configuration (Streamlit)

You can also configure report settings directly through the Streamlit interface:

1. **Navigate to Models Monitoring** page in the application
2. **Click the Edit button** (pencil icon) for your model
3. **Scroll to the "Report Parameters" section** which includes:
   - **Auto Send Reports**: Enable/disable automatic report generation
   - **Report Title**: Custom title for the email subject
   - **Email Recipients**: Comma-separated list of email addresses
   - **Max Emerging Topics**: Number of weak signal topics to include (1-20)
   - **Max Strong Topics**: Number of strong signal topics to include (1-20)
4. **Click OK** to save changes
5. **Toggle learning on** to activate scheduling (if not already active)

When you save changes:
- Configuration is updated in the TOML file
- If learning is active, the report generation schedule is automatically updated
- Changes take effect immediately for the next scheduled run

## Usage

### Automatic Mode (Recommended)

1. **Configure your model** with report settings in the `{model_id}_analysis.toml` file
2. **Activate training** in the Streamlit UI by toggling the model's learning switch
3. **Reports are automatically scheduled** to run 1 hour after each training session

When you activate training for a model:
- If `auto_send = true` and `email_recipients` are configured
- Report generation is automatically scheduled in crontab
- Reports will be generated and sent after each training cycle

### Manual Mode

You can also generate reports manually using the CLI:

```bash
python -m bertrend_apps.prospective_demo.automated_report_generation \
    <username> \
    <model_id> \
    [--reference-date YYYY-MM-DD]
```

**Arguments:**
- `username`: Your user identifier
- `model_id`: The model identifier
- `--reference-date`: (Optional) Specific date for report generation. If not provided, uses the most recent data.

**Example:**
```bash
python -m bertrend_apps.prospective_demo.automated_report_generation \
    john_doe \
    ai_trends \
    --reference-date 2025-10-13
```

## How It Works

### Workflow

1. **Training Job Runs**: Scheduled training processes new data (e.g., every 7 days)
2. **Analysis Generated**: Weak and strong signals are analyzed with LLM descriptions
3. **Report Scheduled**: Report generation runs 1 hour after training completes
4. **Report Created**: The system loads the latest analysis and generates an HTML report
5. **Email Sent**: Report is automatically sent to configured recipients

### Scheduling Details

- Reports are scheduled to run **1 hour after** the training schedule
- This ensures that training and analysis are complete before report generation
- Schedule is based on the model's `granularity` setting (e.g., every 7 days)
- Random time between 0-6 AM is chosen to distribute load

### Cron Jobs

When you activate training, two cron jobs are created:
1. **Training job**: `process_new_data train-new-model {user} {model_id}`
2. **Report job**: `automated_report_generation {user} {model_id}`

View active cron jobs:
```bash
crontab -l
```

## Email Setup

### Prerequisites

The system uses Gmail API for sending emails. Ensure you have:

1. **Gmail credentials file**: `bertrend_apps/config/gmail_credentials.json`
2. **Token file**: Created automatically on first use at `~/.bertrend/gmail_token.json`

### First Time Setup

When the system first tries to send an email:
1. It will prompt you to authorize the application
2. Follow the browser-based authorization flow
3. The token is saved for future automated use

## Troubleshooting

### Reports Not Being Sent

**Check configuration:**
```bash
# View your model configuration
cat ~/.bertrend/config/users/{username}/{model_id}_analysis.toml
```

Ensure:
- `auto_send = true`
- `email_recipients` is not empty
- Email addresses are valid

**Check cron jobs:**
```bash
crontab -l | grep automated_report_generation
```

Should show a line like:
```
0 1 1,8,15,22,29 * * umask 002; source ~/.bashrc; python -m bertrend_apps.prospective_demo.automated_report_generation ...
```

**Check logs:**
```bash
# Training logs
tail -f ~/.bertrend/logs/users/{username}/learning_{model_id}.log

# Report generation logs
tail -f ~/.bertrend/logs/users/{username}/report_{model_id}.log
```

### Email Authentication Issues

If you see authentication errors:
```bash
# Remove old token
rm ~/.bertrend/gmail_token.json

# Run manually to re-authenticate
python -m bertrend_apps.prospective_demo.automated_report_generation {username} {model_id}
```

### No Recent Data Available

The system automatically uses the most recent interpretation data. If no data is found:
1. Ensure training has run at least once
2. Check that analysis was generated (check interpretation directory)
3. Manually specify a date: `--reference-date YYYY-MM-DD`

## Example Configuration

Complete example configuration file:

```toml
[model_config]
granularity = 7
window_size = 7
language = "en"
split_by_paragraph = true

[analysis_config]
topic_evolution = true
evolution_scenarios = true
multifactorial_analysis = true

[report_config]
auto_send = true
email_recipients = [
    "team-lead@company.com",
    "analyst@company.com"
]
report_title = "Weekly AI & Technology Trends Report"
max_emerging_topics = 3
max_strong_topics = 5
```

## Advanced Usage

### Disable Report Generation

To stop automatic reports while keeping training active:
1. Set `auto_send = false` in the configuration
2. Re-toggle training in the UI to update the schedule

Or manually remove the cron job:
```bash
# List all jobs
crontab -l

# Remove specific job
crontab -l | grep -v "automated_report_generation "  | crontab -
```

### Custom Schedule

The report schedule is automatically derived from the training schedule. To customize:
1. Modify `granularity` in `model_config` (affects both training and reports)
2. Reports run 1 hour after training by design

### Multiple Models

You can configure automated reports for multiple models. Each model:
- Has its own configuration file
- Has independent scheduling
- Can have different recipients and settings

## Support

For issues or questions:
1. Check logs in `~/.bertrend/logs/users/{username}/`
2. Verify configuration in `~/.bertrend/config/users/{username}/`
3. Test manual report generation first
4. Check cron jobs with `crontab -l`
