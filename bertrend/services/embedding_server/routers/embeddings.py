#  Copyright (c) 2024-2026, RTE (https://www.rte-france.com)
#  See AUTHORS.txt
#  SPDX-License-Identifier: MPL-2.0
#  This file is part of BERTrend.

from fastapi import APIRouter, Security
from loguru import logger
from sentence_transformers import SentenceTransformer

from bertrend.services.embedding_server.security import (
    TokenData,
    get_current_client,
    FULL_ACCESS,
)

from bertrend.services.embedding_server.config.settings import get_config
from bertrend.services.embedding_server.models import InputText

# Load the configuration
CONFIG = get_config()

# Load embedding model
logger.debug(f"Loading embedding model : {CONFIG.model_name}")
EMBEDDING_MODEL = SentenceTransformer(CONFIG.model_name, trust_remote_code=True)

# Fix max model length error
if EMBEDDING_MODEL.max_seq_length == 514:
    EMBEDDING_MODEL.max_seq_length = 512
router = APIRouter()


@router.post("/encode", summary="Embed data (requires full_access scope)")
def embed(
    input: InputText,
    current_client: TokenData = Security(get_current_client, scopes=[FULL_ACCESS]),
):
    logger.debug(f"Request by: {current_client.client_id}")
    emb = EMBEDDING_MODEL.encode(input.text, show_progress_bar=input.show_progress_bar)
    return {"embeddings": emb.tolist()}
