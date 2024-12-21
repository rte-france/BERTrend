#  Copyright (c) 2024, RTE (https://www.rte-france.com)
#  See AUTHORS.txt
#  SPDX-License-Identifier: MPL-2.0
#  This file is part of BERTrend.

import json
from pathlib import Path

import torch

from bertrend import PARAMETERS_CONFIG

stopwords_fr_file = Path(__file__).parent / "resources" / "stopwords-fr.json"
stopwords_rte_file = Path(__file__).parent / "resources" / "stopwords-rte.json"
common_ngrams_file = Path(__file__).parent / "resources" / "common_ngrams.json"
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

# BERTopic Hyperparameters
DEFAULT_UMAP_N_COMPONENTS = PARAMETERS_CONFIG["default_umap_n_components"]
DEFAULT_UMAP_N_NEIGHBORS = PARAMETERS_CONFIG["default_umap_n_neighbors"]
DEFAULT_HDBSCAN_MIN_CLUSTER_SIZE = PARAMETERS_CONFIG["default_hdbscan_min_cluster_size"]
DEFAULT_HDBSCAN_MIN_SAMPLES = PARAMETERS_CONFIG["default_hdbscan_min_samples"]
DEFAULT_TOP_N_WORDS = PARAMETERS_CONFIG["default_top_n_words"]
DEFAULT_MIN_DF = PARAMETERS_CONFIG["default_min_df"]
DEFAULT_GRANULARITY = PARAMETERS_CONFIG["default_granularity"]
DEFAULT_MIN_SIMILARITY = PARAMETERS_CONFIG["default_min_similarity"]
DEFAULT_ZEROSHOT_MIN_SIMILARITY = PARAMETERS_CONFIG["default_zeroshot_min_similarity"]
BERTOPIC_SERIALIZATION = PARAMETERS_CONFIG["bertopic_serialization"]
DEFAULT_MMR_DIVERSITY = PARAMETERS_CONFIG["default_mmr_diversity"]
DEFAULT_UMAP_MIN_DIST = PARAMETERS_CONFIG["default_umap_min_dist"]
OUTLIER_REDUCTION_STRATEGY = PARAMETERS_CONFIG["outlier_reduction_strategy"]

# Signal classification Settings
SIGNAL_CLASSIF_LOWER_BOUND = PARAMETERS_CONFIG["signal_classif_lower_bound"]
SIGNAL_CLASSIF_UPPER_BOUND = PARAMETERS_CONFIG["signal_classif_upper_bound"]

# Other Constants
DEFAULT_ZEROSHOT_TOPICS = PARAMETERS_CONFIG["default_zeroshot_topics"]


# Embedding Settings
EMBEDDING_DTYPES = ["float32", "float16", "bfloat16"]
EMBEDDING_BATCH_SIZE = 5000
EMBEDDING_MAX_SEQ_LENGTH = 512
EMBEDDING_DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Other constants
LANGUAGES = ["French", "English"]
HDBSCAN_CLUSTER_SELECTION_METHODS = ["eom", "leaf"]
VECTORIZER_NGRAM_RANGES = [(1, 2), (1, 1), (2, 2)]


# Data Processing
MIN_CHARS_DEFAULT = 100
SAMPLE_SIZE_DEFAULT = None  # Or whatever default you want, None means all documents

# Time Settings
DEFAULT_WINDOW_SIZE = 7  # days
MAX_WINDOW_SIZE = 365  # days

# Data Analysis Settings
POPULARITY_THRESHOLD = 0.1  # for weak signal detection, if applicable
