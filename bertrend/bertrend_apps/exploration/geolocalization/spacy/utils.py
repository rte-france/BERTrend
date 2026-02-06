#  Copyright (c) 2024, RTE (https://www.rte-france.com)
#  See AUTHORS.txt
#  SPDX-License-Identifier: MPL-2.0
#  This file is part of Wattelse, a NLP application suite.

import os
import time
from pathlib import Path
from typing import Union

from dotenv import load_dotenv
from loguru import logger
from spacy.lang.fr import French
from thinc.config import Config

from bertrend.bertrend_apps.exploration.geolocalization.spacy import common_factory

load_dotenv(override=True)

# Those imports are not explicitly used, but mandatory to have a working nlp pipeline... - DO NOT REMOVE!

ASSETS_PATH = Path(__file__).parent.absolute()
RTE_NLP_CONFIG_PATH = ASSETS_PATH / "nlp.config"
DBPEDIA_REST_API_URL = os.getenv("DBPEDIA_REST_API_URL", "http://localhost:2222/rest")

# These logs are just there to force the imports of the factories - DO NOT REMOVE!
logger.debug(f"{common_factory.__name__} loaded.")


def load_nlp(config_path: Union[str, Path] = RTE_NLP_CONFIG_PATH, **kwargs):
    t1 = time.time()
    logger.info("Building nlp...")
    config = Config().from_disk(
        config_path,
        overrides={"components.dbpedia.dbpedia_rest_endpoint": DBPEDIA_REST_API_URL},
    )

    nlp = French.from_config(config, **kwargs)
    t2 = time.time()

    logger.success(f"NLP pipeline built in {t2 - t1:3f} seconds!")
    return nlp
