#  Copyright (c) 2024-2026, RTE (https://www.rte-france.com)
#  See AUTHORS.txt
#  SPDX-License-Identifier: MPL-2.0
#  This file is part of BERTrend.

# Pydantic class for FastAPI typing control
from pydantic import BaseModel


class InputText(BaseModel):
    text: str | list[str]
    show_progress_bar: bool
