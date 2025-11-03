#  Copyright (c) 2024, RTE (https://www.rte-france.com)
#  See AUTHORS.txt
#  SPDX-License-Identifier: MPL-2.0
#  This file is part of BERTrend.
from datetime import datetime
from typing import Optional, List, Any, Dict, Union

from pydantic import BaseModel, Field


class JobCreate(BaseModel):
    job_id: str = Field(..., description="Unique identifier for the job")
    job_name: Optional[str] = Field(
        None, description="Human-readable job name (defaults to job_id)"
    )
    job_type: str = Field(..., description="Type: 'interval', 'cron', or 'date'")
    function_name: str = Field(..., description="Name of the function to execute")
    args: Optional[List[Any]] = Field(
        default=[], description="Arguments for the function"
    )
    kwargs: Optional[Dict[str, Any]] = Field(
        default={}, description="Keyword arguments"
    )

    # Overwrite existing job
    replace_existing: Optional[bool] = Field(
        default=True, description="Overwrite existing job if it exists"
    )

    # Interval trigger fields
    seconds: Optional[int] = None
    minutes: Optional[int] = None
    hours: Optional[int] = None
    days: Optional[int] = None

    # Cron trigger fields (two ways to specify)
    cron_expression: Optional[str] = Field(
        None,
        description="Cron expression (e.g., '0 12 * * *' for daily at noon) or use named fields below",
    )
    cron_minute: Optional[str] = Field(None, description="Minute (0-59 or */5 or 0,30)")
    cron_hour: Optional[str] = Field(None, description="Hour (0-23)")
    cron_day: Optional[str] = Field(None, description="Day of month (1-31)")
    cron_month: Optional[str] = Field(None, description="Month (1-12)")
    cron_day_of_week: Optional[str] = Field(
        None, description="Day of week (0-6, 0=Monday)"
    )

    # Date trigger fields
    run_date: Optional[datetime] = Field(
        None, description="Specific datetime to run once"
    )

    # Execution options
    max_instances: Optional[int] = Field(
        default=3, description="Max concurrent instances"
    )
    coalesce: Optional[bool] = Field(
        default=False, description="Coalesce missed executions"
    )


class JobUpdate(BaseModel):
    job_type: Optional[str] = None
    function_name: Optional[str] = None
    args: Optional[List[Any]] = None
    kwargs: Optional[Dict[str, Any]] = None
    seconds: Optional[int] = None
    minutes: Optional[int] = None
    hours: Optional[int] = None
    days: Optional[int] = None
    cron_expression: Optional[str] = None
    cron_minute: Optional[str] = None
    cron_hour: Optional[str] = None
    cron_day: Optional[str] = None
    cron_month: Optional[str] = None
    cron_day_of_week: Optional[str] = None
    run_date: Optional[datetime] = None
    max_instances: Optional[int] = None
    coalesce: Optional[bool] = None


class JobResponse(BaseModel):
    job_id: str
    name: str
    next_run_time: Optional[datetime]
    trigger: str
    kwargs: dict
    args: list
    func: str
    executor: str
    max_instances: int


class JobExecutionResponse(BaseModel):
    message: str
    timestamp: datetime
    job_id: str


class CronExpressionRequest(BaseModel):
    expression: Optional[str] = Field(None, description="Standard cron expression")
    minute: Optional[str] = Field(None, description="Minute field")
    hour: Optional[str] = Field(None, description="Hour field")
    day: Optional[str] = Field(None, description="Day field")
    month: Optional[str] = Field(None, description="Month field")
    day_of_week: Optional[str] = Field(None, description="Day of week field")


class CronExpressionResponse(BaseModel):
    expression: str
    description: str
    next_runs: List[datetime]
    is_valid: bool
    timezone: str


class JobFindRequest(BaseModel):
    """Request model for finding jobs based on kwargs patterns"""

    kwargs_patterns: Dict[str, Union[str, Dict[str, Any]]] = Field(
        ...,
        description="Dictionary where keys are kwargs field names and values are either regex patterns (strings) or nested dictionaries for deep matching",
    )
    match_all: Optional[bool] = Field(
        default=True,
        description="If True, all patterns must match (AND logic). If False, any pattern match is sufficient (OR logic)",
    )


class JobFindResponse(BaseModel):
    """Response model for job search results"""

    matches_found: int = Field(description="Number of jobs found matching the criteria")
    jobs: List[JobResponse] = Field(
        description="List of jobs that matched the search criteria"
    )
