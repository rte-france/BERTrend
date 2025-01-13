#  Copyright (c) 2024, RTE (https://www.rte-france.com)
#  See AUTHORS.txt
#  SPDX-License-Identifier: MPL-2.0
#  This file is part of BERTrend.
import numpy as np

from pathlib import Path
from loguru import logger

from bertopic import BERTopic
from bertopic.representation import (
    MaximalMarginalRelevance,
    OpenAI,
    KeyBERTInspired,
    BaseRepresentation,
)
from bertopic.vectorizers import ClassTfidfTransformer

from hdbscan import HDBSCAN
from sklearn.feature_extraction.text import CountVectorizer
from umap import UMAP
from sentence_transformers import SentenceTransformer

from bertrend import load_toml_config, LLM_CONFIG
from bertrend.llm_utils.openai_client import OpenAI_Client
from bertrend.llm_utils.prompts import BERTOPIC_FRENCH_TOPIC_REPRESENTATION_PROMPT
from bertrend.parameters import (
    STOPWORDS,
    ENGLISH_STOPWORDS,
    KEYBERT_TOP_N_WORDS,
    KEYBERT_NR_REPR_DOCS,
    KEYBERT_NR_CANDIDATE_WORDS,
    OPENAI_NR_DOCS,
    MMR_REPRESENTATION_MODEL,
    OPENAI_REPRESENTATION_MODEL,
    KEYBERTINSPIRED_REPRESENTATION_MODEL,
    DEFAULT_BERTOPIC_CONFIG_FILE,
)


class TopicModelOutput:
    """Wrapper to encapsulate all results related to topic model output"""

    def __init__(self, topic_model: BERTopic):
        """
        - a topic model
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


class TopicModel:
    """
    Utility class to manage and configure BERTopic instances with custom parameters.
    """

    def __init__(self, config_file: str | Path = DEFAULT_BERTOPIC_CONFIG_FILE):
        """
        Instanciate a class from a TOML config file.
        `config_file` can be:
            - a `str` representing the TOML file
            - a `Path` to a TOML file

        To see file format and list of parameters: bertrend/topic_model/topic_model_default_config.toml
        """
        self.config_file = config_file

        # Load config file
        self.config = self.load_config()

        # Initialize models based on those parameters
        self._initialize_models()

        # Get representation model
        self.config["bertopic_model"]["representation_model"] = (
            self._get_representation_models(
                self.config["bertopic_model"]["representation_model"]
            )
        )

    def load_config(self) -> dict:
        config = load_toml_config(self.config_file)

        # Handle specific parameters

        # Transform ngram_range into tuple
        config["vectorizer_model"]["ngram_range"] = tuple(
            config["vectorizer_model"]["ngram_range"]
        )

        # Load stop words list
        if config["vectorizer_model"]["stop_words"]:
            stop_words = (
                STOPWORDS
                if config["global"]["language"] == "French"
                else ENGLISH_STOPWORDS
            )
            config["vectorizer_model"]["stop_words"] = stop_words

        # BERTopic needs a "None" instead of an empty list, otherwise it'll attempt zeroshot topic modeling on an empty list
        if len(config["bertopic_model"]["zeroshot_topic_list"]) == 0:
            config["bertopic_model"]["zeroshot_topic_list"] = None

        return config

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
                endpoint=LLM_CONFIG["endpoint"],
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
        embedding_model: SentenceTransformer | str,
        embeddings: np.ndarray,
    ) -> TopicModelOutput:
        """
        Create a TopicModelOutput model.

        Args:
            docs (List[str]): List of documents.
            embedding_model (SentenceTransformer | str): Sentence transformer (or associated model name) model for embeddings.
            embeddings (np.ndarray): Precomputed document embeddings.
            umap_model (UMAP): UMAP model for dimensionality reduction.
            hdbscan_model (HDBSCAN): HDBSCAN model for clustering.
            vectorizer_model (CountVectorizer): CountVectorizer model for creating the document-term matrix.
            mmr_model (MaximalMarginalRelevance): MMR model for diverse topic representation.

        Returns:
            BERTopic: A fitted BERTopic model.
        """
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
                logger.info("Applying OpenAI representation model...")
                backup_representation_model = topic_model.representation_model
                topic_model.update_topics(
                    docs=docs,
                    topics=new_topics,
                    representation_model=self._initialize_openai_representation(),
                )
                topic_model.representation_model = backup_representation_model

            logger.success("\tBERTopic model fitted successfully")
            output = TopicModelOutput(topic_model)
            output.topics = new_topics
            output.probs = probs
            return output
        except Exception as e:
            logger.error(f"\tError in create_topic_model: {str(e)}")
            logger.exception("\tTraceback:")
            raise
