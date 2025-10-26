#  Copyright (c) 2024, RTE (https://www.rte-france.com)
#  See AUTHORS.txt
#  SPDX-License-Identifier: MPL-2.0
#  This file is part of BERTrend.

from fastapi import APIRouter


router = APIRouter()


@router.get("/health")
def health():
    return {"status": "ok"}


@router.get("/")
def root():
    """Root endpoint"""
    return {
        "message": "Job Scheduler API with Persistent Storage",
        "storage": "SQLite (jobs.sqlite)",
        "executor": "ProcessPoolExecutor (max_workers=5)",
        "timezone": "Europe/Paris",
        "endpoints": {
            "POST /jobs": "Create a new job",
            "GET /jobs": "List all jobs",
            "GET /jobs/{job_id}": "Get job details",
            "PUT /jobs/{job_id}": "Update a job",
            "DELETE /jobs/{job_id}": "Remove a job",
            "POST /jobs/{job_id}/pause": "Pause a job",
            "POST /jobs/{job_id}/resume": "Resume a job",
            "POST /jobs/{job_id}/run": "Run a job immediately",
            "GET /functions": "List available job functions",
            "POST /cron/validate": "Validate and preview cron expression",
            "GET /cron/examples": "Get cron expression examples",
        },
    }
