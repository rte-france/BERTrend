#  Copyright (c) 2024, RTE (https://www.rte-france.com)
#  See AUTHORS.txt
#  SPDX-License-Identifier: MPL-2.0
#  This file is part of BERTrend.
import subprocess


def get_least_used_gpu():
    try:
        # Execute the command
        result = subprocess.run(
            "nvidia-smi --query-gpu=index,memory.used --format=csv,nounits | tail -n +2 | sort -t',' -k2 -n | head -n 1 | cut -d',' -f1",
            shell=True,
            check=True,
            text=True,
            capture_output=True,
        )
        # Get the output
        least_used_gpu = result.stdout.strip()
        return least_used_gpu
    except subprocess.CalledProcessError as e:
        print(f"An error occurred: {e}")
        return None
