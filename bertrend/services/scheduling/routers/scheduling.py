#  Copyright (c) 2024, RTE (https://www.rte-france.com)
#  See AUTHORS.txt
#  SPDX-License-Identifier: MPL-2.0
#  This file is part of BERTrend.
import re
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import HTTPException, APIRouter, FastAPI
from apscheduler.schedulers.background import BackgroundScheduler
from apscheduler.jobstores.sqlalchemy import SQLAlchemyJobStore
from apscheduler.executors.pool import ProcessPoolExecutor
from apscheduler.triggers.cron import CronTrigger
from apscheduler.triggers.interval import IntervalTrigger
from apscheduler.triggers.date import DateTrigger
from datetime import datetime
from typing import List, Union, Any

from loguru import logger
from bertrend.services.scheduling.job_utils.job_functions import JOB_FUNCTIONS
from bertrend.services.scheduling.models.scheduling_models import (
    JobCreate,
    JobUpdate,
    JobResponse,
    JobExecutionResponse,
    CronExpressionRequest,
    CronExpressionResponse,
    JobFindResponse,
    JobFindRequest,
)

router = APIRouter()

DB_PATH = Path.home() / ".bertrend" / "db"
DB_NAME = "bertrend_jobs.sqlite"
DB_PATH.mkdir(parents=True, exist_ok=True)

# Scheduler will be initialized on first use or set by tests
scheduler = None


def _init_scheduler():
    """Initialize the scheduler if not already set (e.g., by tests)"""
    global scheduler
    if scheduler is None:
        # Configure job stores and executors
        jobstores = {
            "default": SQLAlchemyJobStore(url=f"sqlite:///{DB_PATH}/{DB_NAME}")
        }
        executors = {"default": ProcessPoolExecutor(max_workers=5)}
        job_defaults = {
            "coalesce": False,  # Run all missed executions
            "max_instances": 3,  # Maximum instances of the job running concurrently
        }
        # Initialize APScheduler with persistence (Paris timezone)
        scheduler = BackgroundScheduler(
            jobstores=jobstores,
            executors=executors,
            job_defaults=job_defaults,
            timezone="Europe/Paris",
        )
        scheduler.start()


# Initialize scheduler at module load if not in test environment
# Tests will monkeypatch this before it's used
try:
    _init_scheduler()
except Exception:
    # If initialization fails (e.g., in test environment), scheduler will be set by tests
    pass


def get_trigger(job_data: JobCreate):
    """Create appropriate trigger based on job type"""
    if job_data.job_type == "interval":
        return IntervalTrigger(
            seconds=job_data.seconds or 0,
            minutes=job_data.minutes or 0,
            hours=job_data.hours or 0,
            days=job_data.days or 0,
        )
    elif job_data.job_type == "cron":
        # Support both cron_expression string and individual fields
        if job_data.cron_expression:
            parts = job_data.cron_expression.split()
            if len(parts) != 5:
                raise ValueError(
                    "Cron expression must have 5 parts: minute hour day month day_of_week"
                )
            return CronTrigger(
                minute=parts[0],
                hour=parts[1],
                day=parts[2],
                month=parts[3],
                day_of_week=parts[4],
                timezone="Europe/Paris",
            )
        elif any(
            [
                job_data.cron_minute,
                job_data.cron_hour,
                job_data.cron_day,
                job_data.cron_month,
                job_data.cron_day_of_week,
            ]
        ):
            return CronTrigger(
                minute=job_data.cron_minute or "*",
                hour=job_data.cron_hour or "*",
                day=job_data.cron_day or "*",
                month=job_data.cron_month or "*",
                day_of_week=job_data.cron_day_of_week or "*",
                timezone="Europe/Paris",
            )
        else:
            raise ValueError(
                "Either cron_expression or individual cron fields must be provided"
            )
    elif job_data.job_type == "date":
        if not job_data.run_date:
            raise ValueError("run_date is required for date jobs")
        return DateTrigger(run_date=job_data.run_date, timezone="Europe/Paris")
    else:
        raise ValueError(f"Invalid job_type: {job_data.job_type}")


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup
    logger.info("FastAPI Job Scheduler started")
    logger.info(f"Job store: SQLite (jobs.sqlite)")
    logger.info(f"Executor: ProcessPoolExecutor (max_workers=5)")
    logger.info(f"Timezone: Europe/Paris")

    # Print existing jobs
    existing_jobs = scheduler.get_jobs()
    if existing_jobs:
        logger.info(f"Loaded {len(existing_jobs)} existing jobs from database")
        for job in existing_jobs:
            logger.info(f"  - {job.id}: next run at {job.next_run_time}")

    yield  # Application runs here

    # Shutdown
    logger.info("Shutting down scheduler...")
    scheduler.shutdown(wait=True)
    logger.info("Scheduler shutdown complete")


@router.get("/functions", summary="List available job functions")
def list_functions():
    """List available job functions"""
    return {
        "available_functions": list(JOB_FUNCTIONS.keys()),
        "details": {
            name: {
                "description": (
                    func.__doc__.strip() if func.__doc__ else "No description"
                ),
                "signature": str(
                    func.__code__.co_varnames[: func.__code__.co_argcount]
                ),
            }
            for name, func in JOB_FUNCTIONS.items()
        },
    }


@router.post("/jobs", response_model=JobResponse, status_code=201)
def create_job(job: JobCreate, summary="Create a new scheduled job"):
    """Create a new scheduled job"""
    try:
        # Check if job_id already exists
        if scheduler.get_job(job.job_id):
            raise HTTPException(
                status_code=400, detail=f"Job with id '{job.job_id}' already exists"
            )

        # Get the function to execute
        if job.function_name not in JOB_FUNCTIONS:
            raise HTTPException(
                status_code=400,
                detail=f"Function '{job.function_name}' not found. Available: {list(JOB_FUNCTIONS.keys())}",
            )

        func = JOB_FUNCTIONS[job.function_name]
        trigger = get_trigger(job)

        # Add job to scheduler with persistence
        scheduler.add_job(
            func,
            trigger=trigger,
            id=job.job_id,
            name=job.job_name or job.job_id,
            args=job.args,
            kwargs=job.kwargs,
            max_instances=job.max_instances,
            coalesce=job.coalesce,
            replace_existing=job.replace_existing,
        )

        added_job = scheduler.get_job(job.job_id)

        logger.info(f"Job '{job.job_id}' created successfully")

        return JobResponse(
            job_id=added_job.id,
            name=added_job.name,
            next_run_time=added_job.next_run_time,
            trigger=str(added_job.trigger),
            kwargs=added_job.kwargs,
            args=added_job.args,
            func=added_job.func_ref,
            executor=added_job.executor,
            max_instances=added_job.max_instances,
        )

    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"Error creating job: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error creating job: {str(e)}")


@router.get(
    "/jobs", response_model=List[JobResponse], summary="List all scheduled jobs"
)
def list_jobs():
    """List all scheduled jobs"""
    jobs = scheduler.get_jobs()
    return [
        JobResponse(
            job_id=job.id,
            name=job.name,
            next_run_time=job.next_run_time,
            trigger=str(job.trigger),
            kwargs=job.kwargs,
            args=job.args,
            func=job.func_ref,
            executor=job.executor,
            max_instances=job.max_instances,
        )
        for job in jobs
    ]


@router.get(
    "/jobs/{job_id}", response_model=JobResponse, summary="Get details of a job"
)
def get_job(job_id: str):
    """Get details of a specific job"""
    job = scheduler.get_job(job_id)
    if not job:
        raise HTTPException(status_code=404, detail=f"Job '{job_id}' not found")

    return JobResponse(
        job_id=job.id,
        name=job.name,
        next_run_time=job.next_run_time,
        trigger=str(job.trigger),
        kwargs=job.kwargs,
        args=job.args,
        func=job.func_ref,
        executor=job.executor,
        max_instances=job.max_instances,
    )


@router.put(
    "/jobs/{job_id}", response_model=JobResponse, summary="Update an existing job"
)
def update_job(job_id: str, job_update: JobUpdate):
    """Update an existing job"""
    job = scheduler.get_job(job_id)
    if not job:
        raise HTTPException(status_code=404, detail=f"Job '{job_id}' not found")

    try:
        # Prepare update data
        update_data = job_update.dict(exclude_unset=True)

        # Update trigger if job_type or trigger parameters provided
        if any(
            k in update_data
            for k in [
                "job_type",
                "seconds",
                "minutes",
                "hours",
                "days",
                "cron_expression",
                "run_date",
            ]
        ):
            # Create a temporary JobCreate object for trigger generation
            job_type = update_data.get("job_type", "interval")
            temp_job = JobCreate(
                job_id=job_id,
                job_type=job_type,
                function_name="sample_job",
                seconds=update_data.get("seconds"),
                minutes=update_data.get("minutes"),
                hours=update_data.get("hours"),
                days=update_data.get("days"),
                cron_expression=update_data.get("cron_expression"),
                run_date=update_data.get("run_date"),
            )
            trigger = get_trigger(temp_job)
            scheduler.reschedule_job(job_id, trigger=trigger)

        # Update function if provided
        if "function_name" in update_data:
            if update_data["function_name"] not in JOB_FUNCTIONS:
                raise HTTPException(
                    status_code=400,
                    detail=f"Function '{update_data['function_name']}' not found",
                )
            scheduler.modify_job(
                job_id, func=JOB_FUNCTIONS[update_data["function_name"]]
            )

        # Update args/kwargs if provided
        if "args" in update_data:
            scheduler.modify_job(job_id, args=update_data["args"])
        if "kwargs" in update_data:
            scheduler.modify_job(job_id, kwargs=update_data["kwargs"])

        # Update execution options
        if "max_instances" in update_data:
            scheduler.modify_job(job_id, max_instances=update_data["max_instances"])
        if "coalesce" in update_data:
            scheduler.modify_job(job_id, coalesce=update_data["coalesce"])

        updated_job = scheduler.get_job(job_id)

        logger.info(f"Job '{job_id}' updated successfully")

        return JobResponse(
            job_id=updated_job.id,
            name=updated_job.name,
            next_run_time=updated_job.next_run_time,
            trigger=str(updated_job.trigger),
            executor=updated_job.executor,
            max_instances=updated_job.max_instances,
        )

    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"Error updating job: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error updating job: {str(e)}")


@router.delete("/jobs/{job_id}", summary="Remove a scheduled job")
def delete_job(job_id: str):
    """Remove a scheduled job"""
    job = scheduler.get_job(job_id)
    if not job:
        raise HTTPException(status_code=404, detail=f"Job '{job_id}' not found")

    scheduler.remove_job(job_id)
    logger.info(f"Job '{job_id}' removed successfully")
    return {"message": f"Job '{job_id}' removed successfully"}


def _match_pattern(value: Any, pattern: Union[str, dict[str, Any]]) -> bool:
    """
    Recursively match a value against a pattern.

    Args:
        value: The value to check (can be string, dict, list, etc.)
        pattern: Either a regex string or a dict with nested patterns

    Returns:
        True if the pattern matches, False otherwise
    """
    # If pattern is a string (regex), convert value to string and match
    if isinstance(pattern, str):
        try:
            regex = re.compile(pattern)
            return regex.search(str(value)) is not None
        except re.error:
            return False

    # If pattern is a dict, value must also be a dict for deep matching
    elif isinstance(pattern, dict):
        if not isinstance(value, dict):
            return False

        # Check all pattern keys against value
        for key, sub_pattern in pattern.items():
            if key not in value:
                return False
            if not _match_pattern(value[key], sub_pattern):
                return False
        return True

    # For other types, do string comparison
    else:
        return str(value) == str(pattern)


def _validate_patterns(patterns: Union[str, dict[str, Any]], path: str = ""):
    """Recursively validate regex patterns"""
    if isinstance(patterns, str):
        try:
            re.compile(patterns)
        except re.error as e:
            raise HTTPException(
                status_code=400,
                detail=f"Invalid regex pattern at '{path}': {str(e)}",
            )
    elif isinstance(patterns, dict):
        for key, value in patterns.items():
            _validate_patterns(value, f"{path}.{key}" if path else key)


@router.post("/jobs/find", response_model=JobFindResponse)
def find_job(request: JobFindRequest):
    """Find scheduled jobs based on regex patterns on their kwargs"""
    try:
        # Validate all regex patterns in the request
        _validate_patterns(request.kwargs_patterns)

        # Get all jobs from the scheduler
        all_jobs = scheduler.get_jobs()

        # Filter jobs based on pattern matching in kwargs
        matching_jobs = []
        for job in all_jobs:
            if not job.kwargs:
                # Skip jobs without kwargs if we're searching for kwargs patterns
                continue

            # Check each pattern against the job's kwargs
            matches = []
            for key, pattern in request.kwargs_patterns.items():
                # Check if the key exists in kwargs and if the pattern matches
                if key in job.kwargs:
                    matches.append(_match_pattern(job.kwargs[key], pattern))
                else:
                    matches.append(False)

            # Determine if this job should be included based on match_all flag
            should_include = all(matches) if request.match_all else any(matches)

            if should_include:
                matching_jobs.append(
                    JobResponse(
                        job_id=job.id,
                        name=job.name,
                        next_run_time=job.next_run_time,
                        trigger=str(job.trigger),
                        kwargs=job.kwargs,
                        args=job.args,
                        func=job.func_ref,
                        executor=job.executor,
                        max_instances=job.max_instances,
                    )
                )

        logger.info(
            f"Job search completed: {len(matching_jobs)} matches found "
            f"for patterns {request.kwargs_patterns} (match_all={request.match_all})"
        )

        return JobFindResponse(
            matches_found=len(matching_jobs),
            jobs=matching_jobs,
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error searching for jobs: {str(e)}")
        raise HTTPException(
            status_code=500, detail=f"Error searching for jobs: {str(e)}"
        )


@router.post("/jobs/{job_id}/pause")
def pause_job(job_id: str):
    """Pause a job"""
    job = scheduler.get_job(job_id)
    if not job:
        raise HTTPException(status_code=404, detail=f"Job '{job_id}' not found")

    scheduler.pause_job(job_id)
    logger.info(f"Job '{job_id}' paused")
    return {"message": f"Job '{job_id}' paused"}


@router.post("/jobs/{job_id}/resume")
def resume_job(job_id: str):
    """Resume a paused job"""
    job = scheduler.get_job(job_id)
    if not job:
        raise HTTPException(status_code=404, detail=f"Job '{job_id}' not found")

    scheduler.resume_job(job_id)
    logger.info(f"Job '{job_id}' resumed")
    return {"message": f"Job '{job_id}' resumed"}


@router.post("/jobs/{job_id}/run", response_model=JobExecutionResponse)
def run_job_now(job_id: str):
    """Execute a job immediately (outside of its schedule)"""
    job = scheduler.get_job(job_id)
    if not job:
        raise HTTPException(status_code=404, detail=f"Job '{job_id}' not found")

    try:
        # Run the job immediately
        job.func(*job.args, **job.kwargs)
        logger.info(f"Job '{job_id}' executed manually")

        return JobExecutionResponse(
            message=f"Job '{job_id}' executed successfully",
            timestamp=datetime.now(),
            job_id=job_id,
        )
    except Exception as e:
        logger.error(f"Error executing job '{job_id}': {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error executing job: {str(e)}")


@router.post("/cron/validate", response_model=CronExpressionResponse)
def validate_cron(cron_req: CronExpressionRequest):
    """Validate a cron expression and show next execution times"""
    try:
        # Build cron expression
        if cron_req.expression:
            parts = cron_req.expression.split()
            if len(parts) != 5:
                raise ValueError(
                    "Cron expression must have 5 parts: minute hour day month day_of_week"
                )
            minute, hour, day, month, day_of_week = parts
            expression = cron_req.expression
        else:
            minute = cron_req.minute or "*"
            hour = cron_req.hour or "*"
            day = cron_req.day or "*"
            month = cron_req.month or "*"
            day_of_week = cron_req.day_of_week or "*"
            expression = f"{minute} {hour} {day} {month} {day_of_week}"

        # Create trigger and get next run times
        trigger = CronTrigger(
            minute=minute,
            hour=hour,
            day=day,
            month=month,
            day_of_week=day_of_week,
            timezone="Europe/Paris",
        )

        # Get next 5 execution times
        now = datetime.now()
        next_runs = []
        current = now
        for _ in range(5):
            next_run = trigger.get_next_fire_time(None, current)
            if next_run:
                next_runs.append(next_run)
                current = next_run
            else:
                break

        # Generate human-readable description
        description = _describe_cron(minute, hour, day, month, day_of_week)

        return CronExpressionResponse(
            expression=expression,
            description=description,
            next_runs=next_runs,
            is_valid=True,
            timezone="Europe/Paris",
        )

    except Exception as e:
        raise HTTPException(
            status_code=400, detail=f"Invalid cron expression: {str(e)}"
        )


def _describe_cron(
    minute: str, hour: str, day: str, month: str, day_of_week: str
) -> str:
    """Generate human-readable description of cron expression"""
    parts = []

    # Minute
    if minute == "*":
        parts.append("every minute")
    elif "/" in minute:
        interval = minute.split("/")[1]
        parts.append(f"every {interval} minutes")
    else:
        parts.append(f"at minute {minute}")

    # Hour
    if hour != "*":
        if "/" in hour:
            interval = hour.split("/")[1]
            parts.append(f"every {interval} hours")
        else:
            parts.append(f"at {hour}:00")

    # Day
    if day != "*":
        parts.append(f"on day {day}")

    # Month
    if month != "*":
        months = [
            "Jan",
            "Feb",
            "Mar",
            "Apr",
            "May",
            "Jun",
            "Jul",
            "Aug",
            "Sep",
            "Oct",
            "Nov",
            "Dec",
        ]
        if month.isdigit():
            parts.append(f"in {months[int(month)-1]}")
        else:
            parts.append(f"in month {month}")

    # Day of week
    if day_of_week != "*":
        days = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]
        if day_of_week.isdigit():
            parts.append(f"on {days[int(day_of_week)]}")
        else:
            parts.append(f"on day-of-week {day_of_week}")

    return ", ".join(parts)
