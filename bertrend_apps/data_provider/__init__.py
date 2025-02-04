#  Copyright (c) 2024, RTE (https://www.rte-france.com)
#  See AUTHORS.txt
#  SPDX-License-Identifier: MPL-2.0
#  This file is part of BERTrend.
# Define a pattern for a basic URL validation
URL_PATTERN = (
    r"^(https?://)?([a-z0-9-]+\.)+[a-z]{2,6}(:\d+)?(/[\w.-]*)*$|"
    r"^(https?://)?(localhost|(\d{1,3}\.){3}\d{1,3})(:\d+)?(/[\w.-]*)*$"
)
