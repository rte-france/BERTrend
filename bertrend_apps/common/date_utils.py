#  Copyright (c) 2024, RTE (https://www.rte-france.com)
#  See AUTHORS.txt
#  SPDX-License-Identifier: MPL-2.0
#  This file is part of BERTrend.
from datetime import timedelta


def daterange(start_date, end_date, ndays):
    """Create a range of dates with periods of ndays, including the last incomplete period if needed."""

    # Ensure that end_date is after start_date, if not return nothing
    if end_date <= start_date:
        return

    # Iterate through the periods
    current_start = start_date
    while current_start < end_date:
        # Calculate the end of the current period
        current_end = min(current_start + timedelta(days=ndays), end_date)

        # Yield the start and end of the current period
        yield current_start, current_end

        # Update current_start to be the next period's start
        current_start = current_end
