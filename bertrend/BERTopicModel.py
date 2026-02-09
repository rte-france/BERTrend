#  Copyright (c) 2024-2026, RTE (https://www.rte-france.com)
#  See AUTHORS.txt
#  SPDX-License-Identifier: MPL-2.0
#  This file is part of BERTrend.
from pathlib import Path

import numpy as np
from bertopic import BERTopic
from bertopic.representation import (
    BaseRepresentation,
    KeyBERTInspired,
    MaximalMarginalRelevance,
    OpenAI,
)
from bertopic.vectorizers import ClassTfidfTransformer
from hdbscan import HDBSCAN
from loguru import logger
from sentence_transformers import SentenceTransformer
from sklearn.feature_extraction.text import CountVectorizer
from umap import UMAP

from bertrend import BERTOPIC_DEFAULT_CONFIG_PATH, LLM_CONFIG, load_toml_config
from bertrend.config.parameters import (
    ENGLISH_STOPWORDS,
    KEYBERT_NR_CANDIDATE_WORDS,
    KEYBERT_NR_REPR_DOCS,
    KEYBERT_TOP_N_WORDS,
    KEYBERTINSPIRED_REPRESENTATION_MODEL,
    MMR_REPRESENTATION_MODEL,
    OPENAI_NR_DOCS,
    OPENAI_REPRESENTATION_MODEL,
    STOPWORDS,
)
from bertrend.llm_utils.openai_client import OpenAI_Client
from bertrend.llm_utils.prompts import BERTOPIC_FRENCH_TOPIC_REPRESENTATION_PROMPT


class BERTopicModelOutput:
    """Wrapper to encapsulate all results related to topic model output"""

    def __init__(self, topic_model: BERTopic):
        """
        - a BERTopic model
        - a list of topics indices corresponding to the documents
        - an array of probabilities
        - the document embeddings
        - the token embeddings of each document
        - the tokens (str) of each documents
        """
        # Topic model
        self.topic_model = topic_model
        # List of topics indices corresponding to the documents
        self.topics = None
        # Array of probabilities
        self.probs = None
        # Document embeddings
        self.embeddings = None
        # Token embeddings of each document
        self.token_embeddings = None
        # Tokens (str) of each document
        self.token_strings = None


class BERTopicModel:
    """
    Utility class to manage and configure BERTopic instances with custom parameters.
    """

    def __init__(self, config: str | Path | dict = BERTOPIC_DEFAULT_CONFIG_PATH):
        """
        Initialize a class from a TOML config file.

        Parameters
        ----------
        config : str or Path or dict, default=BERTOPIC_DEFAULT_CONFIG_PATH
            Configuration source, which can be:
            - a `str` representing the TOML file
            - a `Path` to a TOML file
            - a `dict` (with the same structure of the default config) containing values
              to be overridden compared to the default configuration

        Notes
        -----
        To see file format and list of parameters: bertrend/config/topic_model_default_config.toml

        Raises
        ------
        Exception
            If the TOML config file cannot be loaded.
        TypeError
            If the config parameter is not a string, Path, or dict.
        """
        if isinstance(config, str | Path):
            try:
                self.config = load_toml_config(config)
            except Exception as e:
                raise Exception(f"Failed to load TOML config: {e}")
        elif isinstance(config, dict):
            # load default config
            self.config = load_toml_config(BERTOPIC_DEFAULT_CONFIG_PATH)
            # overrides keys with provided dict
            for section, settings in config.items():
                if section in config:
                    self.config[section].update(
                        settings
                    )  # Update the settings in that section
                else:
                    self.config[section] = settings  # If section doesn't exist, add it
        else:
            raise TypeError(
                f"Config must be a string, Path or dict object, got: {type(config)}"
            )

        # Update config file (depending on language, etc.)
        self._update_config()

        # Initialize models based on those parameters
        self._initialize_models()

        # Get representation model
        self.config["bertopic_model"]["representation_model"] = (
            self._get_representation_models(
                self.config["bertopic_model"]["representation_model"]
            )
        )

    @classmethod
    def get_default_config(cls) -> dict:
        """
        Helper function to get default configuration.

        Returns
        -------
        dict
            The default configuration dictionary loaded from the default config file.
        """
        return load_toml_config(BERTOPIC_DEFAULT_CONFIG_PATH)

    def _update_config(self):
        """
        Update the config file depending on initially loaded parameters.
        """
        # Handle specific parameters
        # Transform ngram_range into tuple
        if self.config["vectorizer_model"].get("ngram_range"):
            self.config["vectorizer_model"]["ngram_range"] = tuple(
                self.config["vectorizer_model"]["ngram_range"]
            )

        # Load stop words list
        if self.config["vectorizer_model"].get("stop_words"):
            stop_words = (
                STOPWORDS
                if self.config["global"]["language"] == "French"
                else ENGLISH_STOPWORDS
            )
            self.config["vectorizer_model"]["stop_words"] = stop_words

        # BERTopic needs a "None" instead of an empty list, otherwise it'll attempt zeroshot topic modeling on an empty list
        if not self.config["bertopic_model"].get("zeroshot_topic_list"):  # empty list
            self.config["bertopic_model"]["zeroshot_topic_list"] = None

    def _initialize_models(self):
        self.umap_model = UMAP(**self.config["umap_model"])

        self.hdbscan_model = HDBSCAN(**self.config["hdbscan_model"])

        self.vectorizer_model = CountVectorizer(**self.config["vectorizer_model"])

        self.ctfidf_model = ClassTfidfTransformer(**self.config["ctfidf_model"])

        self.mmr_model = MaximalMarginalRelevance(**self.config["mmr_model"])

    def _initialize_openai_representation(self):
        return OpenAI(
            client=OpenAI_Client(
                api_key=LLM_CONFIG["api_key"],
                base_url=LLM_CONFIG["base_url"],
                model=LLM_CONFIG["model"],
            ).llm_client,
            model=LLM_CONFIG["model"],
            nr_docs=OPENAI_NR_DOCS,
            prompt=(
                BERTOPIC_FRENCH_TOPIC_REPRESENTATION_PROMPT
                if self.config["global"]["language"] == "French"
                else None
            ),
            chat=True,
        )

    @classmethod
    def _initialize_keybert_representation(cls):
        return KeyBERTInspired(
            top_n_words=KEYBERT_TOP_N_WORDS,
            nr_repr_docs=KEYBERT_NR_REPR_DOCS,
            nr_candidate_words=KEYBERT_NR_CANDIDATE_WORDS,
        )

    def _get_representation_models(
        self,
        representation_model: list[str],
    ) -> BaseRepresentation | list[BaseRepresentation]:
        # NB. If OpenAI representation model is present, it will be used in separate step
        model_map = {
            MMR_REPRESENTATION_MODEL: self.mmr_model,
            KEYBERTINSPIRED_REPRESENTATION_MODEL: self._initialize_keybert_representation(),
        }
        models = [
            model_map[rep]
            for rep in representation_model
            if rep != OPENAI_REPRESENTATION_MODEL and rep in model_map
        ]
        self.use_openai_representation = (
            OPENAI_REPRESENTATION_MODEL in representation_model
        )

        return models[0] if len(models) == 1 else models

    def fit(
        self,
        docs: list[str],
        embeddings: np.ndarray,
        embedding_model: SentenceTransformer | str | None = None,
        zeroshot_topic_list: list[str] | None = None,
        zeroshot_min_similarity: float | None = None,
    ) -> BERTopicModelOutput:
        """
        Create a TopicModelOutput model.

        Parameters
        ----------
        docs : list[str]
            List of documents.
        embeddings : np.ndarray
            Precomputed document embeddings.
        embedding_model : SentenceTransformer or str or None, optional
            Sentence transformer (or associated model name) model for embeddings.
            NB. Important: if not set, the keywords describing the topics may not be correct (e.g. accents strip); therefore, it is recommended to set this parameter.
        zeroshot_topic_list : list[str] or None, optional
            List of topics used for zeroshot classification.
        zeroshot_min_similarity : float or None, optional
            Parameter used for zeroshot classification.

        Returns
        -------
        BERTopicModelOutput
            A fitted BERTopic model output object containing the model and related data.

        Raises
        ------
        Exception
            If there's an error during the model creation or fitting process.
        """
        # Override zeroshot parameters if provided in method argument
        if zeroshot_topic_list:
            self.config["bertopic_model"]["zeroshot_topic_list"] = zeroshot_topic_list
        if zeroshot_min_similarity is not None:
            self.config["bertopic_model"]["zeroshot_min_similarity"] = (
                zeroshot_min_similarity
            )
        # Load and fit model
        try:
            logger.debug("\tInitializing BERTopic model")

            topic_model = BERTopic(
                embedding_model=embedding_model,
                umap_model=self.umap_model,
                hdbscan_model=self.hdbscan_model,
                vectorizer_model=self.vectorizer_model,
                ctfidf_model=self.ctfidf_model,
                **self.config["bertopic_model"],
            )
            logger.success("\tBERTopic model instance created successfully")

            logger.debug("\tFitting BERTopic model")
            topics, probs = topic_model.fit_transform(docs, embeddings)

            if not topic_model._outliers:
                logger.warning("\tNo outliers to reduce.")
                new_topics = topics
            else:
                logger.debug("\tReducing outliers")
                new_topics = topic_model.reduce_outliers(
                    documents=docs,
                    topics=topics,
                    embeddings=embeddings,
                    strategy=self.config["reduce_outliers"]["strategy"],
                )
            logger.debug("\tUpdating topics")
            topic_model.update_topics(
                docs=docs,
                topics=new_topics,
                vectorizer_model=self.vectorizer_model,
                ctfidf_model=self.ctfidf_model,
                representation_model=self.config["bertopic_model"][
                    "representation_model"
                ],
            )

            # If OpenAI model is present, apply it after reducing outliers
            if self.use_openai_representation:
                logger.info("\tApplying OpenAI representation model...")
                backup_representation_model = topic_model.representation_model
                topic_model.update_topics(
                    docs=docs,
                    topics=new_topics,
                    representation_model=self._initialize_openai_representation(),
                )
                topic_model.representation_model = backup_representation_model

            logger.success("\tBERTopic model fitted successfully")
            output = BERTopicModelOutput(topic_model)
            output.topics = new_topics
            output.probs = probs
            return output
        except Exception as e:
            logger.error(f"\tError in create_topic_model: {str(e)}")
            logger.exception("\tTraceback:")
            raise
