#  Copyright (c) 2024-2026, RTE (https://www.rte-france.com)
#  See AUTHORS.txt
#  SPDX-License-Identifier: MPL-2.0
#  This file is part of BERTrend.

import os

import uvicorn

# Load the configuration BEFORE any other imports that might use CUDA
from bertrend.services.embedding_server.config.settings import get_config
from bertrend.services.embedding_server.security import (
    get_secret_key,
    load_client_registry,
)

CONFIG = get_config()
# Set the CUDA_VISIBLE_DEVICES environment variable BEFORE importing uvicorn
# This is critical because uvicorn will import the app module, which imports torch
# We override the value with the content of the config
os.environ["CUDA_VISIBLE_DEVICES"] = CONFIG.cuda_visible_devices


def main():
    """Validate security configuration and start the embedding API."""
    get_secret_key()
    load_client_registry()
    uvicorn.run(
        "bertrend.services.embedding_server.main:app",
        host=CONFIG.host,
        port=CONFIG.port,
        workers=CONFIG.number_workers,
        ssl_keyfile="../key.pem",
        ssl_certfile="../cert.pem",
    )


# Start the FastAPI application
if __name__ == "__main__":
    main()
