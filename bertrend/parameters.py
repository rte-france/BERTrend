#  Copyright (c) 2024, RTE (https://www.rte-france.com)
#  See AUTHORS.txt
#  SPDX-License-Identifier: MPL-2.0
#  This file is part of BERTrend.

import json
from pathlib import Path

import torch

from bertrend import BERTOPIC_PARAMETERS, BERTREND_PARAMETERS

stopwords_en_file = Path(__file__).parent / "resources" / "stopwords-en.json"
stopwords_fr_file = Path(__file__).parent / "resources" / "stopwords-fr.json"
stopwords_rte_file = Path(__file__).parent / "resources" / "stopwords-rte.json"
common_ngrams_file = Path(__file__).parent / "resources" / "common_ngrams.json"
with open(stopwords_en_file, "r", encoding="utf-8") as file:
    ENGLISH_STOPWORDS = json.load(file)
with open(stopwords_fr_file, "r", encoding="utf-8") as file:
    FRENCH_STOPWORDS = json.load(file)
with open(stopwords_rte_file, "r", encoding="utf-8") as file:
    STOP_WORDS_RTE = json.load(file)
with open(common_ngrams_file, "r", encoding="utf-8") as file:
    COMMON_NGRAMS = json.load(file)

STOPWORDS = STOP_WORDS_RTE + COMMON_NGRAMS + FRENCH_STOPWORDS

# File names
STATE_FILE = "app_state.pkl"
EMBEDDINGS_FILE = "embeddings.npy"
DOC_GROUPS_FILE = "doc_groups.pkl"
EMB_GROUPS_FILE = "emb_groups.pkl"
GRANULARITY_FILE = "granularity.pkl"
HYPERPARAMS_FILE = "hyperparams.pkl"
DOC_INFO_DF_FILE = "doc_info_df.pkl"
TOPIC_INFO_DF_FILE = "topic_info_df.pkl"
MODELS_TRAINED_FILE = "models_trained_flag.pkl"

# Model file names
ZEROSHOT_TOPICS_DATA_FILE = "zeroshot_topics_data.json"
INDIVIDUAL_MODEL_TOPIC_COUNTS_FILE = "individual_topic_counts.json"
CUMULATIVE_MERGED_TOPIC_COUNTS_FILE = "cumulative_topic_counts.json"

# Embedding models
ENGLISH_EMBEDDING_MODELS = [
    "all-mpnet-base-v2",
    "Alibaba-NLP/gte-base-en-v1.5",
    "all-MiniLM-L12-v2",
]
FRENCH_EMBEDDING_MODELS = [
    "OrdalieTech/Solon-embeddings-base-0.1",
    "OrdalieTech/Solon-embeddings-large-0.1",
    "dangvantuan/sentence-camembert-large",
    "antoinelouis/biencoder-distilcamembert-mmarcoFR",
]

# MODEL REPRESENTATIONS
MMR_REPRESENTATION_MODEL = "MaximalMarginalRelevance"
KEYBERTINSPIRED_REPRESENTATION_MODEL = "KeyBERTInspired"
OPENAI_REPRESENTATION_MODEL = "OpenAI"

# BERTopic Hyperparameters
DEFAULT_BERTOPIC_CONFIG_FILE = (
    Path(__file__).parent / "topic_model" / "topic_model_default_config.toml"
)
DEFAULT_GRANULARITY = BERTOPIC_PARAMETERS["granularity"]
DEFAULT_MIN_SIMILARITY = BERTOPIC_PARAMETERS["min_similarity"]
DEFAULT_ZEROSHOT_MIN_SIMILARITY = BERTOPIC_PARAMETERS["zeroshot_min_similarity"]
BERTOPIC_SERIALIZATION = BERTOPIC_PARAMETERS["bertopic_serialization"]
LANGUAGES = ["French", "English"]
REPRESENTATION_MODELS = [
    MMR_REPRESENTATION_MODEL,
    KEYBERTINSPIRED_REPRESENTATION_MODEL,
    OPENAI_REPRESENTATION_MODEL,
]

# BERTrend parameters
# Signal classification Settings
SIGNAL_CLASSIF_LOWER_BOUND = BERTREND_PARAMETERS["signal_classif_lower_bound"]
SIGNAL_CLASSIF_UPPER_BOUND = BERTREND_PARAMETERS["signal_classif_upper_bound"]

# Embedding Settings
EMBEDDING_DTYPES = ["float32", "float16", "bfloat16"]
EMBEDDING_BATCH_SIZE = 5000
EMBEDDING_MAX_SEQ_LENGTH = 512
EMBEDDING_DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Representation models parameters
KEYBERT_NR_REPR_DOCS = (5,)
KEYBERT_NR_CANDIDATE_WORDS = (40,)
KEYBERT_TOP_N_WORDS = 20
OPENAI_NR_DOCS = 5

# Data Processing
MIN_CHARS_DEFAULT = 1
SAMPLE_SIZE_DEFAULT = 1.0  # Or whatever default you want, None means all documents

# Time Settings
DEFAULT_WINDOW_SIZE = 7  # days
MAX_WINDOW_SIZE = 365  # days

# Data Analysis Settings
POPULARITY_THRESHOLD = 0.1  # for weak signal detection, if applicable
