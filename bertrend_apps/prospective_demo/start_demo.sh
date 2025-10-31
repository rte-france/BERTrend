#
# Copyright (c) 2024, RTE (https://www.rte-france.com)
# See AUTHORS.txt
# SPDX-License-Identifier: MPL-2.0
# This file is part of BERTrend.
#
# Use Python to load .env file and export variables
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

echo "Starting BERTrend Prospective Analysis demo"
CUDA_VISIBLE_DEVICES=0 streamlit run app.py 2>&1 | tee -a $BERTREND_LOGS_DIR/prospective_analysis_demo.log