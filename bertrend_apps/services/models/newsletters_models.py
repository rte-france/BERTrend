#  Copyright (c) 2024, RTE (https://www.rte-france.com)
#  See AUTHORS.txt
#  SPDX-License-Identifier: MPL-2.0
#  This file is part of BERTrend.
from pathlib import Path
from typing import Optional

from pydantic import BaseModel, Field


class NewsletterRequest(BaseModel):
    newsletter_toml_path: Path = Field(
        ..., description="Path to newsletters toml config file"
    )
    data_feed_toml_path: Path = Field(
        ..., description="Path to data feed toml config file"
    )


class NewsletterResponse(BaseModel):
    output_path: Path
    status: str


class ScheduleNewsletterRequest(BaseModel):
    newsletter_toml_cfg_path: Path = Field(
        ..., description="Path to newsletters toml config file"
    )
    data_feed_toml_cfg_path: Path = Field(
        ..., description="Path to data feed toml config file"
    )
    cuda_devices: Optional[str] = Field(
        default=None, description="CUDA_VISIBLE_DEVICES parameters"
    )
