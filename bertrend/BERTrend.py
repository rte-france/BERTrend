#  Copyright (c) 2024, RTE (https://www.rte-france.com)
#  See AUTHORS.txt
#  SPDX-License-Identifier: MPL-2.0
#  This file is part of BERTrend.
import os
import pickle

import dill  # improvement to pickle
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from bertopic import BERTopic
from loguru import logger
from pandas import Timestamp
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from tqdm import tqdm

from bertrend import (
    MODELS_DIR,
    BERTREND_DEFAULT_CONFIG_PATH,
    load_toml_config,
    SIGNAL_EVOLUTION_DATA_DIR,
)

from bertrend.BERTopicModel import BERTopicModel
from bertrend.config.parameters import (
    DOC_INFO_DF_FILE,
    TOPIC_INFO_DF_FILE,
    BERTOPIC_SERIALIZATION,
    SIGNAL_CLASSIF_LOWER_BOUND,
    SIGNAL_CLASSIF_UPPER_BOUND,
    BERTREND_FILE,
    LANGUAGES,
)
from bertrend.services.embedding_service import EmbeddingService
from bertrend.trend_analysis.weak_signals import (
    _initialize_new_topic,
    update_existing_topic,
    _apply_decay_to_inactive_topics,
    _filter_data,
    _is_rising_popularity,
    _create_dataframes,
)
from bertrend.utils.data_loading import TEXT_COLUMN


class BERTrend:
    """
    A comprehensive trend analysis and weak signal detection tool using BERTopic.

    This class provides functionality for analyzing trends and detecting weak signals
    in text data over time using BERTopic models.

    Parameters
    ----------
    embedding_model_name : str
        Name of the embedding model to use
    granularity : int
        Number of days to group documents
    min_chars : int
        Minimum character length for documents
    split_by_paragraph : bool
        Whether to split documents by paragraph
    sample_size : int
        Number of documents to sample
    """

    def __init__(
        self,
        config_file: str | Path = BERTREND_DEFAULT_CONFIG_PATH,
        topic_model: BERTopicModel = None,
    ):
        """
        Initialize a class from a TOML config file.

        Parameters
        ----------
        config_file : str or Path, default=BERTREND_DEFAULT_CONFIG_PATH
            Configuration file path, which can be:
            - a `str` representing the TOML file
            - a `Path` to a TOML file
        topic_model : BERTopicModel, optional
            Pre-configured BERTopicModel instance. If None, a default instance will be created.

        Notes
        -----
        To see file format and list of parameters: bertrend/config/bertrend_default_config.toml
        """
        # Load configuration file
        self.config_file = config_file
        self.config = self._load_config()

        # Initialize topic model
        self.topic_model = BERTopicModel() if topic_model is None else topic_model

        # State variables of BERTrend
        self._is_fitted = False

        # Variables related to time-based topic models
        # - last_topic_model: last trained BERTopic model (used for last timestamp)
        self.last_topic_model: BERTopic = None
        # - last_timestamp: timestamp associated to the last trained BERTopic model
        self.last_topic_model_timestamp: pd.Timestamp = None
        # - doc_groups: Dictionary of document groups for each timestamp.
        self.doc_groups: dict[pd.Timestamp, list[str]] = {}
        # - emb_groups: Dictionary of document embeddings for each timestamp.
        self.emb_groups: dict[pd.Timestamp, np.ndarray] = {}
        self.merge_df_size_over_time = []

        # Variables containing info about merged topics
        self.all_new_topics_df = None
        self.all_merge_histories_df = None
        self.merged_df = None

        # Variables containing info about topic popularity
        # - topic_sizes: Dictionary storing topic sizes and related information over time.
        self.topic_sizes: dict[int, dict[str, Any]] = defaultdict(
            lambda: defaultdict(list)
        )
        # - topic_last_popularity: Dictionary storing the last known popularity of each topic.
        self.topic_last_popularity: dict[int, float] = {}
        # - topic_last_update: Dictionary storing the last update timestamp of each topic.
        self.topic_last_update: dict[int, pd.Timestamp] = {}

    def _load_config(self) -> dict:
        """
        Load the TOML config file as a dict when initializing the class.

        Returns
        -------
        dict
            The configuration dictionary loaded from the config file.
        """
        config = load_toml_config(self.config_file)
        return config

    def get_periods(self):
        """
        Return the time periods used by the model.

        Returns
        -------
        list
            List of timestamps representing the time periods used by the model.
        """
        return list(self.doc_groups.keys())

    def _train_by_period(
        self,
        period: pd.Timestamp,
        group: pd.DataFrame,
        embedding_model: SentenceTransformer,
        embeddings: np.ndarray,
    ) -> tuple[
        BERTopic,
        list[str],
        np.ndarray,
    ]:
        """
        Train BERTopic models for a given time period from the grouped data.

        Parameters
        ----------
        period : pd.Timestamp
            Timestamp of the time period.
        group : pd.DataFrame
            Group of data associated to that timestamp.
        embedding_model : SentenceTransformer
            Sentence transformer model for embeddings.
        embeddings : np.ndarray
            Precomputed document embeddings.

        Returns
        -------
        tuple
            A tuple containing:
            - BERTopic : Trained BERTopic model for this period.
            - list[str] : Document groups for this period.
            - np.ndarray : Document embeddings for this period.
        """
        docs = group[TEXT_COLUMN].tolist()
        embeddings_subset = embeddings[group.index]

        logger.debug(f"Processing period: {period}")
        logger.debug(f"Number of documents: {len(docs)}")

        logger.debug("Creating topic model...")
        topic_model = self.topic_model.fit(
            docs=docs,
            embeddings=embeddings_subset,
        ).topic_model

        logger.debug("Topic model created successfully")

        doc_info_df = topic_model.get_document_info(docs=docs).rename(
            columns={"Document": "Paragraph"}
        )
        doc_info_df = doc_info_df.merge(
            group[[TEXT_COLUMN, "document_id", "source", "url"]],
            left_on="Paragraph",
            right_on=TEXT_COLUMN,
            how="left",
        )
        doc_info_df = doc_info_df.drop(columns=[TEXT_COLUMN])

        topic_info_df = topic_model.get_topic_info()
        topic_doc_count_df = (
            doc_info_df.groupby("Topic")["document_id"]
            .nunique()
            .reset_index(name="Document_Count")
        )
        topic_sources_df = (
            doc_info_df.groupby("Topic")["source"]
            .apply(list)
            .reset_index(name="Sources")
        )
        topic_urls_df = (
            doc_info_df.groupby("Topic")["url"].apply(list).reset_index(name="URLs")
        )

        topic_info_df = topic_info_df.merge(topic_doc_count_df, on="Topic", how="left")
        topic_info_df = topic_info_df.merge(topic_sources_df, on="Topic", how="left")
        topic_info_df = topic_info_df.merge(topic_urls_df, on="Topic", how="left")

        topic_info_df = topic_info_df[
            [
                "Topic",
                "Count",
                "Document_Count",
                "Representation",
                "Name",
                "Representative_Docs",
                "Sources",
                "URLs",
            ]
        ]

        topic_model.doc_info_df = doc_info_df
        topic_model.topic_info_df = topic_info_df
        return topic_model, docs, embeddings_subset

    def train_topic_models(
        self,
        grouped_data: dict[pd.Timestamp, pd.DataFrame],
        embedding_model: SentenceTransformer | str,
        embeddings: np.ndarray,
        bertrend_models_path: Path = MODELS_DIR,
        save_topic_models: bool = True,
    ):
        """
        Train BERTopic models for each timestamp in the grouped data.

        This method trains BERTopic models for each timestamp in the grouped data and
        stores the results in instance variables.

        Parameters
        ----------
        grouped_data : dict[pd.Timestamp, pd.DataFrame]
            Dictionary of grouped data by timestamp.
        embedding_model : SentenceTransformer or str
            Sentence transformer model for embeddings.
        embeddings : np.ndarray
            Precomputed document embeddings.
        bertrend_models_path : Path, default=MODELS_DIR
            Path to BERTrend models folder.
        save_topic_models : bool, default=True
            Boolean flag to save topic models.

        Notes
        -----
        This method updates the following instance variables:
        - doc_groups : Dictionary of document groups for each timestamp.
        - emb_groups : Dictionary of document embeddings for each timestamp.
        - last_topic_model : The most recently trained BERTopic model.
        - last_topic_model_timestamp : The timestamp of the most recently trained model.
        """
        non_empty_groups = [
            (period, group) for period, group in grouped_data.items() if not group.empty
        ]

        failed_periods = []
        successful_periods = []

        for i, (period, group) in enumerate(non_empty_groups):
            try:
                logger.info(f"Training topic model {i + 1}/{len(non_empty_groups)}...")
                (
                    new_topic_model,
                    self.doc_groups[period],
                    self.emb_groups[period],
                ) = self._train_by_period(
                    period, group, embedding_model, embeddings
                )  # TODO: parallelize?

                if self.last_topic_model is not None:
                    self.merge_models_with(new_topic_model, period)

                if save_topic_models:
                    logger.info(f"Saving topic model for period {period}...")
                    # save new model to disk for potential reuse
                    BERTrend.save_topic_model(
                        period, new_topic_model, bertrend_models_path
                    )

                # Update last topic model
                self.last_topic_model = new_topic_model
                self.last_topic_model_timestamp = period

                logger.debug(f"Successfully processed period: {period}")
                successful_periods.append(period)

            except Exception as e:
                logger.error(f"Error processing period {period}: {e}")
                logger.exception("Traceback:")
                failed_periods.append(period)
                continue  # TODO: better error handling

        self._is_fitted = True

        if failed_periods:
            logger.error(
                f"Training completed with failures for {len(failed_periods)} periods: {failed_periods}"
            )
            if successful_periods:
                logger.warning(
                    f"Successfully trained topic models for {len(successful_periods)} periods: {successful_periods}"
                )
        else:
            logger.success("Finished training all topic models")

    def merge_models_with(
        self,
        new_model: BERTopic,
        new_model_timestamp: pd.Timestamp,
        min_similarity: int | None = None,
    ):
        """
        Merge two specific topic models.

        Parameters
        ----------
        new_model : BERTopic
            The new topic model to merge with the last trained model.
        new_model_timestamp : pd.Timestamp
            Timestamp associated with the new model.
        min_similarity : int or None, default=None
            Minimum similarity threshold for merging topics. If None, uses the value from config.

        Raises
        ------
        ValueError
            If either the new model or the last topic model is not valid.

        Notes
        -----
        This method updates the merged_df, all_merge_histories_df, and all_new_topics_df instance variables.
        """
        logger.debug(
            f"Merging topic models for timestamps: {self.last_topic_model_timestamp} and {new_model_timestamp}"
        )

        # Check if both timestamps exist in topic models
        if new_model is None or self.last_topic_model is None:
            raise ValueError(
                f"At least one topic model is not valid. You must fit the BERTrend model before merging models."
            )

        # Get default BERTrend config if argument is not provided
        if min_similarity is None:
            min_similarity = self.config["min_similarity"]

        merged_df_without_outliers = self.merged_df

        if merged_df_without_outliers is None:
            # Preprocess the two specific models
            topic_df1 = _preprocess_model(
                self.last_topic_model,
                self.doc_groups[self.last_topic_model_timestamp],
                self.emb_groups[self.last_topic_model_timestamp],
            )
            # Remove outliers (Topic == -1)
            df1 = topic_df1[topic_df1["Topic"] != -1]
            merged_df_without_outliers = df1

        topic_df2 = _preprocess_model(
            new_model,
            self.doc_groups[new_model_timestamp],
            self.emb_groups[new_model_timestamp],
        )
        # Remove outliers (Topic == -1)
        df2 = topic_df2[topic_df2["Topic"] != -1]

        # Check if either dataframe is empty
        if merged_df_without_outliers.empty or df2.empty:
            logger.warning(
                f"One of the dataframes is empty. Skipping merge for {new_model_timestamp}"
            )
            return None

        # Merge the two models
        (
            merged_df_without_outliers,
            merge_history,
            new_topics,
        ) = _merge_models(
            merged_df_without_outliers,
            df2,
            min_similarity=min_similarity,
            timestamp=self.last_topic_model_timestamp,
        )

        # Store results
        self.merged_df = merged_df_without_outliers
        self.all_merge_histories_df = pd.concat(
            [self.all_merge_histories_df, merge_history], ignore_index=True
        )
        self.all_new_topics_df = pd.concat(
            [self.all_new_topics_df, new_topics], ignore_index=True
        )
        self.merge_df_size_over_time.append(
            (
                new_model_timestamp,
                merged_df_without_outliers["Topic"].max() + 1,
            )
        )
        logger.success(f"Models {new_model_timestamp} merged successfully with others")

    def calculate_signal_popularity(
        self,
        decay_factor: float | None = None,
        decay_power: float | None = None,
    ):
        """
        Compute the popularity of signals (topics) over time, accounting for merges and applying decay.

        Parameters
        ----------
        decay_factor : float or None, default=None
            Factor for exponential decay calculation. If None, uses the value from config.
        decay_power : float or None, default=None
            Power for exponential decay calculation. If None, uses the value from config.

        Notes
        -----
        This method updates the following instance variables:
        - topic_sizes : Dict[int, Dict[str, Any]]
            Dictionary storing topic sizes and related information over time.
        - topic_last_popularity : Dict[int, float]
            Dictionary storing the last known popularity of each topic.
        - topic_last_update : Dict[int, pd.Timestamp]
            Dictionary storing the last update timestamp of each topic.

        Raises
        ------
        ValueError
            If less than two topic models have been trained.
        """
        # Get default BERTrend config if argument is not provided
        if decay_factor is None:
            decay_factor = self.config["decay_factor"]
        if decay_power is None:
            decay_power = self.config["decay_power"]

        # Check if models are merged
        if len(self.doc_groups) < 2:
            logger.error(
                "You must train at least two topic models first before computing signal popularity."
            )
            return

        topic_sizes = defaultdict(lambda: defaultdict(list))
        topic_last_popularity = {}
        topic_last_update = {}

        min_timestamp = self.all_merge_histories_df["Timestamp"].min()
        max_timestamp = self.all_merge_histories_df["Timestamp"].max()
        granularity_timedelta = pd.Timedelta(days=self.config["granularity"])
        time_range = pd.date_range(
            start=min_timestamp.to_pydatetime(),
            end=(max_timestamp + granularity_timedelta).to_pydatetime(),
            freq=granularity_timedelta,
        )

        for current_timestamp in time_range:
            current_df = self.all_merge_histories_df[
                self.all_merge_histories_df["Timestamp"] == current_timestamp
            ]
            updated_topics = set()

            # Process active topics (those appearing in the current timestamp)
            for _, row in current_df.iterrows():
                current_topic = row["Topic1"]
                updated_topics.add(current_topic)

                if current_topic not in topic_sizes:
                    # Initialize new topic
                    _initialize_new_topic(
                        topic_sizes,
                        topic_last_popularity,
                        topic_last_update,
                        current_topic,
                        current_timestamp,
                        row,
                    )

                # Update existing topic
                update_existing_topic(
                    topic_sizes,
                    topic_last_popularity,
                    topic_last_update,
                    current_topic,
                    current_timestamp,
                    granularity_timedelta,
                    row,
                )

                # Mark the topic as updated for the next timestamp
                updated_topics.add(current_topic)

            # Apply decay to topics that weren't updated in this timestamp or the next
            next_timestamp = current_timestamp + granularity_timedelta
            next_df = self.all_merge_histories_df[
                self.all_merge_histories_df["Timestamp"] == next_timestamp
            ]
            topics_updated_next = set(next_df["Topic1"])

            _apply_decay_to_inactive_topics(
                topic_sizes,
                topic_last_popularity,
                topic_last_update,
                updated_topics,
                topics_updated_next,
                current_timestamp,
                granularity_timedelta,
                decay_factor,
                decay_power,
            )

        self.topic_sizes = topic_sizes
        self.topic_last_popularity = topic_last_popularity
        self.topic_last_update = topic_last_update

    def _compute_popularity_values_and_thresholds(
        self, window_size: int, current_date: Timestamp
    ) -> tuple[Timestamp, Timestamp, list, float, float]:
        """
        Compute the popularity values and thresholds for the considered time window.

        Parameters
        ----------
        window_size : int
            The retrospective window size in days.
        current_date : Timestamp
            The current date selected by the user.

        Returns
        -------
        tuple
            A tuple containing:
            - window_start : Timestamp
                Start timestamp of the time window.
            - window_end : Timestamp
                End timestamp of the time window.
            - all_popularity_values : list
                List of all popularity values within the time window.
            - q1 : float
                The lower threshold value (10th percentile of popularity values).
            - q3 : float
                The upper threshold value (90th percentile of popularity values).
        """

        window_size_timedelta = pd.Timedelta(days=window_size)
        granularity_timedelta = pd.Timedelta(days=self.config["granularity"])

        current_date = pd.to_datetime(current_date).floor("D")  # Floor to start of day
        window_start = current_date - window_size_timedelta
        window_end = current_date + granularity_timedelta

        # Calculate q1 and q3 values (we remove very low values of disappearing signals to not skew the thresholds)
        all_popularity_values = [
            popularity
            for topic, data in self.topic_sizes.items()
            for timestamp, popularity in zip(
                pd.to_datetime(data["Timestamps"]), data["Popularity"]
            )
            if window_start <= timestamp <= current_date and popularity > 1e-5
        ]

        if all_popularity_values:
            q1 = np.percentile(all_popularity_values, SIGNAL_CLASSIF_LOWER_BOUND)
            q3 = np.percentile(all_popularity_values, SIGNAL_CLASSIF_UPPER_BOUND)
        else:
            q1, q3 = 0, 0

        return window_start, window_end, all_popularity_values, q1, q3

    def _classify_signals(
        self,
        window_start: pd.Timestamp,
        window_end: pd.Timestamp,
        q1: float,
        q3: float,
        rising_popularity_only: bool = True,
        keep_documents: bool = True,
    ) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """
        Classify signals into weak signal and strong signal dataframes.

        Parameters
        ----------
        window_start : pd.Timestamp
            The start timestamp of the window.
        window_end : pd.Timestamp
            The end timestamp of the window.
        q1 : float
            The lower threshold value (10th percentile of popularity values).
        q3 : float
            The upper threshold value (90th percentile of popularity values).
        rising_popularity_only : bool, default=True
            Whether to consider only rising popularity topics as weak signals.
        keep_documents : bool, default=True
            Whether to keep track of the documents or not.

        Returns
        -------
        tuple
            A tuple containing:
            - noise_topics_df : pd.DataFrame
                DataFrame containing noise topics.
            - weak_signal_topics_df : pd.DataFrame
                DataFrame containing weak signal topics.
            - strong_signal_topics_df : pd.DataFrame
                DataFrame containing strong signal topics.
        """
        noise_topics = []
        weak_signal_topics = []
        strong_signal_topics = []

        sorted_topics = sorted(self.topic_sizes.items(), key=lambda x: x[0])

        for topic, data in sorted_topics:
            filtered_data = _filter_data(data, window_end, keep_documents)
            if not filtered_data["Timestamps"]:
                continue

            window_popularities = [
                (timestamp, popularity)
                for timestamp, popularity in zip(
                    filtered_data["Timestamps"], filtered_data["Popularity"]
                )
                if window_start <= timestamp <= window_end
            ]

            if window_popularities:
                latest_timestamp, latest_popularity = window_popularities[-1]
                docs_count = (
                    filtered_data["Docs_Count"][-1]
                    if filtered_data["Docs_Count"]
                    else 0
                )
                paragraphs_count = (
                    filtered_data["Paragraphs_Count"][-1]
                    if filtered_data["Paragraphs_Count"]
                    else 0
                )
                source_diversity = (
                    filtered_data["Source_Diversity"][-1]
                    if filtered_data["Source_Diversity"]
                    else 0
                )

                topic_data = (
                    topic,
                    latest_popularity,
                    latest_timestamp,
                    docs_count,
                    paragraphs_count,
                    source_diversity,
                    filtered_data,
                )

                if latest_popularity < q1:
                    noise_topics.append(topic_data)
                elif q1 <= latest_popularity <= q3:
                    if rising_popularity_only:
                        if _is_rising_popularity(filtered_data, latest_timestamp):
                            weak_signal_topics.append(topic_data)
                        else:
                            noise_topics.append(topic_data)
                    else:
                        weak_signal_topics.append(topic_data)
                else:
                    strong_signal_topics.append(topic_data)

        return _create_dataframes(
            noise_topics, weak_signal_topics, strong_signal_topics, keep_documents
        )

    def classify_signals(
        self, window_size: int, current_date: Timestamp
    ) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """
        Classify signals into weak signal and strong signal dataframes for the considered time window.

        Parameters
        ----------
        window_size : int
            The retrospective window size in days.
        current_date : Timestamp
            The current date selected by the user.

        Returns
        -------
        tuple
            A tuple containing:
            - noise_topics_df : pd.DataFrame
                DataFrame containing noise topics.
            - weak_signal_topics_df : pd.DataFrame
                DataFrame containing weak signal topics.
            - strong_signal_topics_df : pd.DataFrame
                DataFrame containing strong signal topics.
        """
        # Compute threshold values
        window_start, window_end, all_popularity_values, q1, q3 = (
            self._compute_popularity_values_and_thresholds(window_size, current_date)
        )

        # Classify signals
        noise_topics_df, weak_signal_topics_df, strong_signal_topics_df = (
            self._classify_signals(window_start, window_end, q1, q3)
        )
        return noise_topics_df, weak_signal_topics_df, strong_signal_topics_df

    def save_model(self, models_path: Path = MODELS_DIR):
        """
        Save BERTrend model to disk.

        Parameters
        ----------
        models_path : Path, default=MODELS_DIR
            Path to the directory where the model will be saved.

        Notes
        -----
        This method serializes the entire BERTrend object using dill and saves it to the specified path.
        """
        models_path.mkdir(parents=True, exist_ok=True)
        # Serialize BERTrend object (using dill as an improvement of pickle for complex objects)
        with open(models_path / BERTREND_FILE, "wb") as f:
            dill.dump(self, f)
        logger.info(f"BERTrend model saved to: {models_path}")

    @classmethod
    def restore_model(cls, models_path: Path = MODELS_DIR) -> "BERTrend":
        """
        Restore a previously saved BERTrend model from disk.

        Parameters
        ----------
        models_path : Path, default=MODELS_DIR
            Path to the directory where the model was saved.

        Returns
        -------
        BERTrend
            The restored BERTrend model instance.

        Raises
        ------
        FileNotFoundError
            If the specified models_path does not exist.

        Notes
        -----
        This method deserializes a BERTrend object using dill from the specified path.
        """
        if not models_path.exists():
            raise FileNotFoundError(f"models_path={models_path} does not exist")
        logger.info(f"Loading BERTrend model from: {models_path}")
        # Unserialize BERTrend object (using dill as an improvement of pickle for complex objects)
        with open(models_path / BERTREND_FILE, "rb") as f:
            bertrend = dill.load(f)
        return bertrend

    @classmethod
    def save_topic_model(
        cls, period: pd.Timestamp, topic_model: BERTopic, models_path: Path = MODELS_DIR
    ):
        """
        Serialize a topic model to disk for potential reuse.

        Parameters
        ----------
        period : pd.Timestamp
            Timestamp associated with the topic model.
        topic_model : BERTopic
            The BERTopic model to save.
        models_path : Path, default=MODELS_DIR
            Path to the directory where the model will be saved.

        Notes
        -----
        This method saves the BERTopic model along with its associated document and topic information
        to a directory named with the period's date (YYYY-MM-DD format).
        """
        models_path.mkdir(parents=True, exist_ok=True)
        model_dir = models_path / period.strftime("%Y-%m-%d")
        model_dir.mkdir(exist_ok=True)
        embedding_model = topic_model.embedding_model
        topic_model.save(
            model_dir,
            serialization=BERTOPIC_SERIALIZATION,
            save_ctfidf=False,
            save_embedding_model=embedding_model,
        )
        topic_model.doc_info_df.to_pickle(model_dir / DOC_INFO_DF_FILE)
        topic_model.topic_info_df.to_pickle(model_dir / TOPIC_INFO_DF_FILE)

    @classmethod
    def restore_topic_model(
        cls, period: pd.Timestamp, models_path: Path = MODELS_DIR
    ) -> BERTopic | None:
        """
        Restore a previously stored topic model.

        Parameters
        ----------
        period : pd.Timestamp
            Timestamp associated with the topic model to restore.
        models_path : Path, default=MODELS_DIR
            Path to the directory where the model was saved.

        Returns
        -------
        BERTopic or None
            The restored BERTopic model if found, None otherwise.

        Raises
        ------
        FileNotFoundError
            If the specified models_path does not exist.

        Notes
        -----
        This method looks for a directory named with the period's date (YYYY-MM-DD format)
        and attempts to load the BERTopic model along with its associated document and topic information.
        """
        if not models_path.exists():
            raise FileNotFoundError(f"models_path={models_path} does not exist")
        # Restore topic models using the selected serialization type
        for period_dir in models_path.glob(
            r"????-??-??"
        ):  # filter dir that are formatted YYYY-MM-DD
            if period_dir.is_dir():
                if period_dir.name == period.strftime("%Y-%m-%d"):
                    topic_model = BERTopic.load(period_dir)
                    doc_info_df_file = period_dir / DOC_INFO_DF_FILE
                    topic_info_df_file = period_dir / TOPIC_INFO_DF_FILE
                    if doc_info_df_file.exists() and topic_info_df_file.exists():
                        topic_model.doc_info_df = pd.read_pickle(doc_info_df_file)
                        topic_model.topic_info_df = pd.read_pickle(topic_info_df_file)
                    else:
                        logger.warning(
                            f"doc_info_df or topic_info_df not found for period {period_dir.name}"
                        )
                    return topic_model
        return None

    def restore_topic_models(
        self, models_path: Path = MODELS_DIR
    ) -> dict[pd.Timestamp, BERTopic] | None:
        """
        Restore all previously stored topic models for the periods in this BERTrend instance.

        Parameters
        ----------
        models_path : Path, default=MODELS_DIR
            Path to the directory where the models were saved.

        Returns
        -------
        dict or None
            A dictionary mapping timestamps to their corresponding BERTopic models.
            Returns an empty dictionary if no models are found.

        Notes
        -----
        This method attempts to restore topic models for all periods that exist in the
        current BERTrend instance by calling restore_topic_model for each period.
        """
        topic_models = {}
        for ts in self.get_periods():
            tm = self.restore_topic_model(period=ts, models_path=models_path)
            if tm:
                topic_models[ts] = tm
        return topic_models

    def save_signal_evolution_data(
        self,
        window_size: int,
        start_timestamp: pd.Timestamp,
        end_timestamp: pd.Timestamp,
    ) -> Path:
        """
        Save signal evolution data for a range of timestamps.

        Parameters
        ----------
        window_size : int
            The retrospective window size in days.
        start_timestamp : pd.Timestamp
            The start timestamp of the range to process.
        end_timestamp : pd.Timestamp
            The end timestamp of the range to process.

        Returns
        -------
        Path
            Path to the directory where the signal evolution data was saved.

        Notes
        -----
        This method processes each timestamp in the specified range, classifies signals,
        and saves the results to disk for later analysis. The data is saved in a directory
        named 'retrospective_{window_size}_days' within the SIGNAL_EVOLUTION_DATA_DIR.
        """
        save_path = SIGNAL_EVOLUTION_DATA_DIR / f"retrospective_{window_size}_days"
        os.makedirs(save_path, exist_ok=True)

        q1_values, q3_values, timestamps_over_time = [], [], []
        noise_dfs, weak_signal_dfs, strong_signal_dfs = [], [], []

        for current_timestamp in tqdm(
            pd.date_range(
                start=start_timestamp,
                end=end_timestamp,
                freq=pd.Timedelta(days=self.config["granularity"]),
            ),
            desc="Processing timestamps",
        ):
            window_start, window_end, all_popularity_values, q1, q3 = (
                self._compute_popularity_values_and_thresholds(
                    window_size, current_timestamp
                )
            )

            noise_df, weak_signal_df, strong_signal_df = self._classify_signals(
                window_start, window_end, q1, q3, keep_documents=False
            )

            noise_dfs.append(noise_df)
            weak_signal_dfs.append(weak_signal_df)
            strong_signal_dfs.append(strong_signal_df)

            timestamps_over_time.append(current_timestamp)

        # Save the grouped dataframes
        with open(save_path / "noise_dfs_over_time.pkl", "wb") as f:
            pickle.dump(noise_dfs, f)
        with open(save_path / "weak_signal_dfs_over_time.pkl", "wb") as f:
            pickle.dump(weak_signal_dfs, f)
        with open(save_path / "strong_signal_dfs_over_time.pkl", "wb") as f:
            pickle.dump(strong_signal_dfs, f)

        # Save the metadata
        with open(save_path / "metadata.pkl", "wb") as f:
            metadata = {
                "window_size": window_size,
                "granularity": self.config["granularity"],
                "timestamps": timestamps_over_time,
                "q1_values": q1_values,
                "q3_values": q3_values,
            }
            pickle.dump(metadata, f)

        return save_path


def train_new_data(
    reference_timestamp: pd.Timestamp,
    new_data: pd.DataFrame,
    bertrend_models_path: Path,
    embedding_service: EmbeddingService,
    granularity: int,
    language: str,
) -> BERTrend:
    """
    Process new data for incremental trend analysis.

    Parameters
    ----------
    reference_timestamp: pd.Timestamp
        Reference timestamp to use for the topic model.
    new_data : pd.DataFrame
        DataFrame containing new text data to process.
    bertrend_models_path : Path
        Path to the directory where BERTrend models are stored.
    embedding_service : EmbeddingService
        Service for generating text embeddings.
    granularity : int
        Number of days to group documents.
    language : str
        Language of the text data.

    Returns
    -------
    BERTrend
        Updated BERTrend model with the new data incorporated.

    Notes
    -----
    This function performs the following steps:
    1. Loads a previously saved BERTrend model
    2. Trains a new topic model with the new data
    3. Merges the models and updates merge histories
    4. Saves the model and returns it

    If no previous model exists, a new BERTrend model is created.
    """
    logger.debug(f"Processing new data: {len(new_data)} items")
    logger.info(f"Reference timestamp: {reference_timestamp}")

    # Restore previous models
    try:
        logger.info(f"Restoring previous BERTrend models from {bertrend_models_path}")
        bertrend = BERTrend.restore_model(bertrend_models_path)
    except:
        logger.warning("Cannot restore previous models, creating new one")
        # overrides default params
        if language and language in LANGUAGES:
            bertrend = BERTrend(
                topic_model=BERTopicModel({"global": {"language": language}})
            )
        else:
            bertrend = BERTrend(topic_model=BERTopicModel())
        bertrend.config["granularity"] = granularity

    # Embed new data
    embeddings, token_strings, token_embeddings = embedding_service.embed(
        texts=new_data[TEXT_COLUMN]
    )
    embedding_model_name = embedding_service.embedding_model_name

    # Create topic model for new data
    bertrend.train_topic_models(
        {reference_timestamp: new_data},
        embeddings=embeddings,
        embedding_model=embedding_model_name,
        bertrend_models_path=bertrend_models_path,
        save_topic_models=True,
    )

    logger.info(f"BERTrend built from {len(bertrend.doc_groups)} periods")
    # Save model
    bertrend.save_model(models_path=bertrend_models_path)

    return bertrend


def _preprocess_model(
    topic_model: BERTopic, docs: list[str], embeddings: np.ndarray
) -> pd.DataFrame:
    """
    Preprocess a BERTopic model by extracting topic information, document groups, document embeddings, and URLs.

    Parameters
    ----------
    topic_model : BERTopic
        A fitted BERTopic model.
    docs : list[str]
        List of documents.
    embeddings : np.ndarray
        Precomputed document embeddings.

    Returns
    -------
    pd.DataFrame
        A DataFrame with topic information, document groups, document embeddings, and URLs.
        The DataFrame contains columns for Topic, Count, Document_Count, Representation,
        Documents, Embedding, DocEmbeddings, Sources, and URLs.
    """
    topic_info = topic_model.topic_info_df
    doc_info = topic_model.doc_info_df
    doc_groups = doc_info.groupby("Topic")["Paragraph"].apply(list)

    topic_doc_embeddings = []
    topic_embeddings = []
    topic_sources = []
    topic_urls = []

    for topic_docs in doc_groups:
        doc_embeddings = [embeddings[docs.index(doc)] for doc in topic_docs]
        topic_doc_embeddings.append(doc_embeddings)
        topic_embeddings.append(np.mean(doc_embeddings, axis=0))
        topic_sources.append(
            doc_info[doc_info["Paragraph"].isin(topic_docs)]["source"].tolist()
        )
        topic_urls.append(
            doc_info[doc_info["Paragraph"].isin(topic_docs)]["url"].tolist()
        )

    topic_df = pd.DataFrame(
        {
            "Topic": topic_info["Topic"],
            "Count": topic_info["Count"],
            "Document_Count": topic_info["Document_Count"],
            "Representation": topic_info["Representation"],
            "Documents": doc_groups.tolist(),
            "Embedding": topic_embeddings,
            "DocEmbeddings": topic_doc_embeddings,
            "Sources": topic_sources,
            "URLs": topic_urls,
        }
    )

    return topic_df


def _merge_models(
    df1: pd.DataFrame, df2: pd.DataFrame, min_similarity: float, timestamp: pd.Timestamp
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Merge two topic model dataframes based on topic similarity.

    Parameters
    ----------
    df1 : pd.DataFrame
        First topic model dataframe (existing topics).
    df2 : pd.DataFrame
        Second topic model dataframe (new topics).
    min_similarity : float
        Minimum cosine similarity threshold for merging topics.
    timestamp : pd.Timestamp
        Timestamp associated with the merge operation.

    Returns
    -------
    tuple
        A tuple containing:
        - merged_df : pd.DataFrame
            DataFrame with merged topics.
        - merge_history : pd.DataFrame
            DataFrame containing the history of merged topics.
        - new_topics : pd.DataFrame
            DataFrame containing new topics that weren't merged.

    Notes
    -----
    This function computes cosine similarities between topic embeddings and merges
    topics that exceed the minimum similarity threshold. Topics that don't meet the
    threshold are added as new topics.
    """
    merged_df = df1.copy()
    merge_history = []

    embeddings1 = np.stack(df1["Embedding"].values)
    embeddings2 = np.stack(df2["Embedding"].values)

    similarities = cosine_similarity(embeddings1, embeddings2)
    max_similarities = np.max(similarities, axis=0)
    max_similar_topics = df1["Topic"].values[np.argmax(similarities, axis=0)]

    new_topics_mask = max_similarities < min_similarity
    new_topics_data = df2[new_topics_mask].copy()
    new_topics_data["Topic"] = np.arange(
        merged_df["Topic"].max() + 1,
        merged_df["Topic"].max() + 1 + len(new_topics_data),
    )
    new_topics_data["Timestamp"] = timestamp

    merged_df = pd.concat([merged_df, new_topics_data], ignore_index=True)
    new_topics = new_topics_data.copy()

    merge_topics_mask = max_similarities >= min_similarity
    merge_topics_data = df2[merge_topics_mask]

    for max_similar_topic, group in merge_topics_data.groupby(
        max_similar_topics[merge_topics_mask]
    ):
        similar_row = df1[df1["Topic"] == max_similar_topic].iloc[0]
        index = merged_df[merged_df["Topic"] == max_similar_topic].index[0]

        merged_df.at[index, "Count"] += group["Count"].sum()
        merged_df.at[index, "Document_Count"] += group["Document_Count"].sum()

        # Update the 'Documents' field with only the new documents from the current timestamp
        new_documents = [doc for docs in group["Documents"] for doc in docs]
        merged_df.at[index, "Documents"] = similar_row["Documents"] + [
            (timestamp, new_documents)
        ]

        merged_df.at[index, "Sources"] += [
            source for sources in group["Sources"] for source in sources
        ]
        merged_df.at[index, "URLs"] += [url for urls in group["URLs"] for url in urls]

        merge_history.extend(
            {
                "Timestamp": timestamp,
                "Topic1": max_similar_topic,
                "Topic2": row["Topic"],
                "Representation1": similar_row["Representation"],
                "Representation2": row["Representation"],
                "Embedding1": similar_row["Embedding"],
                "Embedding2": row["Embedding"],
                "Similarity": max_similarities[row["Topic"]],
                "Count1": len(similar_row["Documents"]),
                "Count2": len(row["Documents"]),
                "Document_Count1": similar_row["Document_Count"],
                "Document_Count2": row["Document_Count"],
                "Documents1": similar_row["Documents"],
                "Documents2": row["Documents"],
                "Source1": similar_row["Sources"],
                "Source2": row["Sources"],
                "URLs1": similar_row["URLs"],
                "URLs2": row["URLs"],
            }
            for _, row in group.iterrows()
        )

    return merged_df, pd.DataFrame(merge_history), new_topics
