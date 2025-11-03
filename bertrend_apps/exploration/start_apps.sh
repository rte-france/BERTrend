#!/bin/bash

#
# Copyright (c) 2024, RTE (https://www.rte-france.com)
# See AUTHORS.txt
# SPDX-License-Identifier: MPL-2.0
# This file is part of BERTrend.
#
eval $(python3 << 'EOF'
from dotenv import dotenv_values
from dotenv import find_dotenv

# Find .env file starting from current directory
dotenv_path = find_dotenv(usecwd=True)

if dotenv_path:
    # Load only the variables from .env file (not all environment)
    env_vars = dotenv_values(dotenv_path)

    # Export only the variables from .env
    for key, value in env_vars.items():
        if value is not None:  # Skip empty values
            # Escape special characters in the value
            value_escaped = value.replace('"', '\\"')
            print(f'export {key}="{value_escaped}"')
else:
    import sys
    print('echo "Warning: .env file not found"', file=sys.stderr)
EOF
)
# Set logs directory and create if not exists
export BERTREND_LOGS_DIR=$BERTREND_BASE_DIR/logs/bertrend
mkdir -p $BERTREND_LOGS_DIR

echo "Starting Wattelse Veille & Analyse"
screen -dmS curebot bash -c 'cd `pwd`/curebot && ./start_newsletter_generator.sh 2>&1 | tee -a $BERTREND_LOGS_DIR/curebot.log; bash'
