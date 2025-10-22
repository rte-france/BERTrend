#  Copyright (c) 2024, RTE (https://www.rte-france.com)
#  See AUTHORS.txt
#  SPDX-License-Identifier: MPL-2.0
#  This file is part of BERTrend.
from datetime import timedelta


def daterange(start_date, end_date, ndays):
    """Create a range of dates."""
    for n in range(int((end_date - start_date).days / ndays)):
        yield (
            start_date + timedelta(ndays * n),
            start_date + timedelta(ndays * (n + 1)),
        )
