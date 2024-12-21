#  Copyright (c) 2024, RTE (https://www.rte-france.com)
#  See AUTHORS.txt
#  SPDX-License-Identifier: MPL-2.0
#  This file is part of BERTrend.

import os
from typing import List, Tuple, Union

import numpy as np
import pandas as pd
import torch
from bertopic import BERTopic
from bertopic.backend import BaseEmbedder
from bertopic.representation import KeyBERTInspired, MaximalMarginalRelevance, OpenAI
from bertopic.vectorizers import ClassTfidfTransformer
from hdbscan import HDBSCAN
from loguru import logger
from nltk.corpus import stopwords
from numpy import ndarray
from sentence_transformers import SentenceTransformer
from sklearn.feature_extraction.text import CountVectorizer
from tqdm import tqdm
from umap import UMAP

from bertrend import BASE_CACHE_PATH, LLM_CONFIG
from bertrend.parameters import STOPWORDS
from bertrend.llm_utils.openai_client import OpenAI_Client
from bertrend.utils.data_loading import TEXT_COLUMN
from bertrend_apps.newsletters.prompts import FRENCH_TOPIC_REPRESENTATION_PROMPT

from bertrend.utils.cache_utils import load_embeddings, save_embeddings, get_hash

# Parameters:
DEFAULT_EMBEDDING_MODEL_NAME = "paraphrase-multilingual-MiniLM-L12-v2"
DEFAULT_NR_TOPICS = 10
DEFAULT_NGRAM_RANGE = (1, 1)
DEFAULT_MIN_DF = 2

DEFAULT_UMAP_MODEL = UMAP(n_neighbors=15, n_components=5, min_dist=0.0, metric="cosine")
DEFAULT_HDBSCAN_MODEL = HDBSCAN(
    min_cluster_size=15,
    metric="euclidean",
    cluster_selection_method="eom",
    prediction_data=True,
)

DEFAULT_VECTORIZER_MODEL = CountVectorizer(
    stop_words=STOPWORDS,
    ngram_range=DEFAULT_NGRAM_RANGE,
    min_df=DEFAULT_MIN_DF,
)

DEFAULT_CTFIDF_MODEL = ClassTfidfTransformer(reduce_frequent_words=True)

RepresentationModelType = Union[KeyBERTInspired, MaximalMarginalRelevance, OpenAI]
DEFAULT_REPRESENTATION_MODEL: List[RepresentationModelType] = [
    MaximalMarginalRelevance(diversity=0.3)
]

# TODO - a lot of duplicate code with weak_signals - to be unified


class EmbeddingModel(BaseEmbedder):
    """
    Custom class for the embedding model. Currently, supports SentenceBert models (model_name should refer to a SentenceBert model).
    Implements batch processing for efficient memory usage and handles different input types.
    """

    def __init__(self, model_name, batch_size=5000):
        super().__init__()

        logger.info(f"Loading embedding model: {model_name}...")
        self.embedding_model = SentenceTransformer(model_name)

        # Handle the particular scenario of when max seq length in original model is abnormal (not a power of 2)
        if self.embedding_model.max_seq_length == 514:
            self.embedding_model.max_seq_length = 512

        self.name = model_name
        self.batch_size = batch_size
        logger.debug("Embedding model loaded")

    def embed(self, documents: Union[List[str], pd.Series], verbose=True) -> np.ndarray:
        # Convert to list if input is a pandas Series
        if isinstance(documents, pd.Series):
            documents = documents.tolist()

        num_documents = len(documents)
        num_batches = (num_documents + self.batch_size - 1) // self.batch_size
        embeddings = []

        # Embed by batches instead of everything at once to not quickly saturate GPU
        for i in tqdm(
            range(num_batches), desc="Embedding batches", disable=not verbose
        ):
            start_idx = i * self.batch_size
            end_idx = min(start_idx + self.batch_size, num_documents)
            batch_documents = documents[start_idx:end_idx]

            batch_embeddings = self.embedding_model.encode(
                batch_documents, show_progress_bar=False, output_value=None
            )
            embeddings.append(batch_embeddings)

        # Concatenate all batch embeddings
        all_embeddings = np.concatenate(embeddings, axis=0)

        logger.success(f"Embedded {num_documents} documents in {num_batches} batches")
        return all_embeddings


def convert_to_numpy(obj, type=None):
    """
    Convert a torch.Tensor or list of torch.Tensors to numpy arrays.
    Args:
        obj: The object to convert (torch.Tensor or list).
        type: The type of conversion (optional, used for token ids).
    Returns:
        np.ndarray or list of np.ndarray.
    """
    if isinstance(obj, torch.Tensor):
        return (
            obj.numpy().astype(np.int64)
            if type == "token_id"
            else obj.numpy().astype(np.float32)
        )
    elif isinstance(obj, list):
        return [convert_to_numpy(item) for item in obj]
    else:
        raise TypeError("Object must be a list or torch.Tensor")


def group_tokens(tokenizer, token_ids, token_embeddings, language="french"):
    """
    Group split tokens into whole words and average their embeddings.
    Args:
        tokenizer: The tokenizer to use for converting ids to tokens.
        token_ids: List of token ids.
        token_embeddings: List of token embeddings.
        language: The language of the tokens (default is "french").
    Returns:
        List of grouped tokens and their corresponding embeddings.
    """
    grouped_token_lists = []
    grouped_embedding_lists = []

    special_tokens = {
        "english": ["[CLS]", "[SEP]", "[PAD]"],
        "french": ["<s>", "</s>", "<pad>"],
    }
    subword_prefix = {"english": "##", "french": "▁"}

    for token_id, token_embedding in tqdm(
        zip(token_ids, token_embeddings), desc="Grouping split tokens into whole words"
    ):
        tokens = tokenizer.convert_ids_to_tokens(token_id)

        grouped_tokens = []
        grouped_embeddings = []
        current_word = ""
        current_embedding = []

        for token, embedding in zip(tokens, token_embedding):
            if token in special_tokens[language]:
                continue

            if language == "french" and token.startswith(subword_prefix[language]):
                if current_word:
                    grouped_tokens.append(current_word)
                    grouped_embeddings.append(np.mean(current_embedding, axis=0))
                current_word = token[1:]
                current_embedding = [embedding]
            elif language == "english" and not token.startswith(
                subword_prefix[language]
            ):
                if current_word:
                    grouped_tokens.append(current_word)
                    grouped_embeddings.append(np.mean(current_embedding, axis=0))
                current_word = token
                current_embedding = [embedding]
            else:
                current_word += token.lstrip(subword_prefix[language])
                current_embedding.append(embedding)

        if current_word:
            grouped_tokens.append(current_word)
            grouped_embeddings.append(np.mean(current_embedding, axis=0))

        grouped_token_lists.append(grouped_tokens)
        grouped_embedding_lists.append(np.array(grouped_embeddings))

    return grouped_token_lists, grouped_embedding_lists


def remove_special_tokens(tokenizer, token_id, token_embedding, special_tokens):
    """
    Remove special tokens from the token ids and embeddings.
    Args:
        tokenizer: The tokenizer to use for converting ids to tokens.
        token_id: List of token ids.
        token_embedding: List of token embeddings.
        special_tokens: List of special tokens to remove.
    Returns:
        List of filtered tokens and their corresponding embeddings.
    """
    tokens = tokenizer.convert_ids_to_tokens(token_id)

    filtered_tokens = []
    filtered_embeddings = []
    for token, embedding in zip(tokens, token_embedding):
        if token not in special_tokens:
            filtered_tokens.append(token)
            filtered_embeddings.append(embedding)

    return filtered_tokens, filtered_embeddings


def train_BERTopic(
    full_dataset: pd.DataFrame,
    indices: pd.Series = None,
    column: str = TEXT_COLUMN,
    embedding_model_name: str = DEFAULT_EMBEDDING_MODEL_NAME,
    umap_model: UMAP = DEFAULT_UMAP_MODEL,
    hdbscan_model: HDBSCAN = DEFAULT_HDBSCAN_MODEL,
    vectorizer_model: CountVectorizer = DEFAULT_VECTORIZER_MODEL,
    ctfidf_model: ClassTfidfTransformer = DEFAULT_CTFIDF_MODEL,
    representation_model: List[RepresentationModelType] = DEFAULT_REPRESENTATION_MODEL,
    top_n_words: int = STOPWORDS,
    nr_topics: Union[str, int] = DEFAULT_NR_TOPICS,
    use_cache: bool = True,
    cache_base_name: str = None,
    form_parameters: dict = None,
    **kwargs,
) -> Tuple[BERTopic, List[int], ndarray, ndarray, List[ndarray], List[List[str]]]:
    """
    Train a BERTopic model with customizable representation models.

    Parameters:
    ----------
    full_dataset: pd.DataFrame
        The full dataset to train
    indices: pd.Series
        Indices of the full_dataset to be used for partial training
    column: str
        Column name containing the text data
    embedding_model_name: str
        Name of the embedding model to use
    umap_model: UMAP
        UMAP model to be used in BERTopic
    hdbscan_model: HDBSCAN
        HDBSCAN model to be used in BERTopic
    vectorizer_model: CountVectorizer
        CountVectorizer model to be used in BERTopic
    ctfidf_model: ClassTfidfTransformer
        ClassTfidfTransformer model to be used in BERTopic
    representation_model: List[RepresentationModelType]
        List of representation models to use
    top_n_words: int
        Number of descriptive words per topic
    nr_topics: int
        Number of topics
    use_cache: bool
        Parameter to decide to store embeddings of the full dataset in cache
    cache_base_name: str
        Base name of the cache
    form_parameters: dict
        Additional parameters passed from the Streamlit form

    Returns:
    -------
    A tuple containing:
        - a topic model
        - a list of topics indices corresponding to the documents
        - an array of probabilities
        - the document embeddings
        - the token embeddings of each document
        - the tokens (str) of each documents
    """
    if form_parameters:
        # Update parameters based on form_parameters
        embedding_model_name = form_parameters["embedding_model_name"]
        umap_model = UMAP(
            n_neighbors=form_parameters["umap_n_neighbors"],
            n_components=form_parameters["umap_n_components"],
            min_dist=form_parameters["umap_min_dist"],
            metric=form_parameters["umap_metric"],
        )
        hdbscan_model = HDBSCAN(
            min_cluster_size=form_parameters["hdbscan_min_cluster_size"],
            min_samples=form_parameters["hdbscan_min_samples"],
            metric=form_parameters["hdbscan_metric"],
            cluster_selection_method=form_parameters[
                "hdbscan_cluster_selection_method"
            ],
            cluster_selection_epsilon=form_parameters[
                "hdbscan_cluster_selection_epsilon"
            ],
            max_cluster_size=form_parameters["hdbscan_max_cluster_size"],
            allow_single_cluster=form_parameters["hdbscan_allow_single_cluster"],
            prediction_data=True,
        )
        stop_words = (
            stopwords.words("english")
            if form_parameters["countvectorizer_stop_words"] == "english"
            else STOPWORDS
        )
        vectorizer_model = CountVectorizer(
            stop_words=stop_words,
            ngram_range=form_parameters["countvectorizer_ngram_range"],
            min_df=form_parameters["countvectorizer_min_df"],
        )
        ctfidf_model = ClassTfidfTransformer(
            reduce_frequent_words=form_parameters["ctfidf_reduce_frequent_words"],
            bm25_weighting=form_parameters["ctfidf_bm25_weighting"],
        )
        representation_model = []
        for model in form_parameters["representation_model"]:
            if model == "MaximalMarginalRelevance":
                representation_model.append(
                    MaximalMarginalRelevance(
                        diversity=form_parameters["MaximalMarginalRelevance_diversity"],
                        top_n_words=form_parameters[
                            "MaximalMarginalRelevance_top_n_words"
                        ],
                    )
                )
            elif model == "KeyBERTInspired":
                representation_model.append(
                    KeyBERTInspired(
                        top_n_words=form_parameters["KeyBERTInspired_top_n_words"],
                        nr_repr_docs=form_parameters["KeyBERTInspired_nr_repr_docs"],
                        nr_candidate_words=form_parameters[
                            "KeyBERTInspired_nr_candidate_words"
                        ],
                    )
                )
            elif model == "OpenAI":
                representation_model.append(
                    OpenAI(
                        client=OpenAI_Client(
                            api_key=LLM_CONFIG["api_key"],
                            endpoint=LLM_CONFIG["endpoint"],
                            model=LLM_CONFIG["model"],
                        ).llm_client,
                        model=os.environ["OPENAI_DEFAULT_MODEL_NAME"],
                        nr_docs=form_parameters["OpenAI_nr_docs"],
                        prompt=(
                            FRENCH_TOPIC_REPRESENTATION_PROMPT
                            if form_parameters.get("OpenAI_language", "Français")
                            == "Français"
                            else None
                        ),
                        chat=True,
                    )
                )
        top_n_words = form_parameters["bertopic_top_n_words"]
        nr_topics = (
            form_parameters["bertopic_nr_topics"]
            if form_parameters["bertopic_nr_topics"] > 0
            else None
        )
        use_cache = form_parameters["use_cached_embeddings"]

    if use_cache and cache_base_name is None:
        cache_base_name = get_hash(full_dataset[column])

    cache_path = BASE_CACHE_PATH / f"{embedding_model_name}_{cache_base_name}.pkl"
    cache_path.parent.mkdir(parents=True, exist_ok=True)

    logger.debug(f"Using cache: {use_cache}")
    embedding_model = EmbeddingModel(embedding_model_name)

    if indices is not None:
        filtered_dataset = full_dataset[full_dataset.index.isin(indices)].reset_index(
            drop=True
        )
    else:
        filtered_dataset = full_dataset

    if cache_path.exists() and use_cache:
        embeddings = load_embeddings(cache_path)
        logger.info(f"Embeddings loaded from cache file: {cache_path}")
        token_embeddings = None
        token_strings = None
    else:
        logger.info("Computing embeddings")
        output = embedding_model.embed(filtered_dataset[column])

        embeddings = [item["sentence_embedding"].detach().cpu() for item in output]
        embeddings = torch.stack(embeddings)
        embeddings = embeddings.numpy()

        token_embeddings = [item["token_embeddings"].detach().cpu() for item in output]
        token_ids = [item["input_ids"].detach().cpu() for item in output]

        token_embeddings = convert_to_numpy(token_embeddings)
        token_ids = convert_to_numpy(token_ids, type="token_id")

        tokenizer = embedding_model.embedding_model._first_module().tokenizer

        token_strings, token_embeddings = group_tokens(
            tokenizer, token_ids, token_embeddings, language="french"
        )

        if use_cache:
            save_embeddings(embeddings, cache_path)
            logger.info(f"Embeddings stored to cache file: {cache_path}")

    if nr_topics == 0:
        nr_topics = None

    # Separate OpenAI model if present
    openai_model = None
    other_models = []
    for model in representation_model:
        if isinstance(model, OpenAI):
            openai_model = model
        else:
            other_models.append(model)

    logger.debug(f"Representation models used: {other_models}")
    logger.debug(f"Using OpenAI to finetune representations: {(openai_model != None)}")

    # Build BERTopic model
    topic_model = BERTopic(
        embedding_model=embedding_model.embedding_model,
        umap_model=umap_model,
        hdbscan_model=hdbscan_model,
        vectorizer_model=vectorizer_model,
        ctfidf_model=ctfidf_model,
        representation_model=other_models,
        top_n_words=top_n_words,
        nr_topics=nr_topics,
        calculate_probabilities=True,
        verbose=True,
    )

    logger.info("Fitting BERTopic...")

    topics, probs = topic_model.fit_transform(filtered_dataset[column], embeddings)

    logger.info("Reducing outliers via embeddings strategy...")
    if not topic_model._outliers:
        logger.warning("No outliers to reduce.")
        new_topics = topics
    else:
        new_topics = topic_model.reduce_outliers(
            documents=filtered_dataset[column],
            topics=topics,
            embeddings=embeddings,
            strategy="embeddings",
        )

        # BUG: here with bertopic > 0.16.2
        topic_model.update_topics(
            filtered_dataset[column],
            topics=new_topics,
            vectorizer_model=vectorizer_model,
            representation_model=other_models,
        )

    # If OpenAI model is present, apply it after reducing outliers
    if openai_model:
        logger.info("Applying OpenAI representation model...")
        backup_representation_model = topic_model.representation_model
        topic_model.update_topics(
            filtered_dataset[column],
            topics=new_topics,
            representation_model=openai_model,
        )
        topic_model.representation_model = backup_representation_model

    return topic_model, new_topics, probs, embeddings, token_embeddings, token_strings
