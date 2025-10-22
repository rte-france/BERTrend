#  Copyright (c) 2024, RTE (https://www.rte-france.com)
#  See AUTHORS.txt
#  SPDX-License-Identifier: MPL-2.0
#  This file is part of BERTrend.
from typing import Optional
from pydantic import BaseModel, Field


class TrainNewModelRequest(BaseModel):
    """Request model for training a new BERTrend model incrementally"""
    user_name: str = Field(..., description="Identifier of the user")
    model_id: str = Field(..., description="ID of the model/data to train")


class TrainNewModelResponse(BaseModel):
    """Response model for train new model endpoint"""
    status: str
    message: str


class RegenerateRequest(BaseModel):
    """Request model for regenerating models from scratch"""
    user: str = Field(..., description="Identifier of the user")
    model_id: str = Field(..., description="ID of the model to be regenerated from scratch")
    with_analysis: bool = Field(
        default=True, 
        description="Regenerate LLM analysis (may take time)"
    )
    since: Optional[str] = Field(
        default=None, 
        description="Date to be considered as the beginning of the analysis (format: YYYY-MM-dd)"
    )


class RegenerateResponse(BaseModel):
    """Response model for regenerate endpoint"""
    status: str
    message: str
