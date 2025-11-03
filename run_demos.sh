#!/bin/bash
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

export BERTREND_HOME=$(python -c "import os; import bertrend; print(os.path.dirname(os.path.dirname(bertrend.__file__)))")
supervisord -c supervisord.conf
