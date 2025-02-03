#  Copyright (c) 2024, RTE (https://www.rte-france.com)
#  See AUTHORS.txt
#  SPDX-License-Identifier: MPL-2.0
#  This file is part of BERTrend.
import os
import pickle
import shutil
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
    CACHE_PATH,
    BERTREND_DEFAULT_CONFIG_PATH,
    load_toml_config,
    SIGNAL_EVOLUTION_DATA_DIR,
)

from bertrend.BERTopicModel import BERTopicModel
from bertrend.config.parameters import (
    DOC_INFO_DF_FILE,
    TOPIC_INFO_DF_FILE,
    DOC_GROUPS_FILE,
    MODELS_TRAINED_FILE,
    EMB_GROUPS_FILE,
    HYPERPARAMS_FILE,
    BERTOPIC_SERIALIZATION,
    SIGNAL_CLASSIF_LOWER_BOUND,
    SIGNAL_CLASSIF_UPPER_BOUND,
)
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

    Key Parameters:
    - embedding_model_name: Name of the embedding model to use
    - granularity: Number of days to group documents
    - min_chars: Minimum character length for documents
    - split_by_paragraph: Whether to split documents by paragraph
    - sample_size: Number of documents to sample
    """

    def __init__(
        self,
        config_file: str | Path = BERTREND_DEFAULT_CONFIG_PATH,
        topic_model: BERTopicModel = None,
    ):
        """
        Initialize a class from a TOML config file.
        `config_file` can be:
            - a `str` representing the TOML file
            - a `Path` to a TOML file

        To see file format and list of parameters: bertrend/config/bertrend_default_config.toml
        """
        # Load configuration file
        self.config_file = config_file
        self.config = self._load_config()

        # Initialize topic model
        self.topic_model = BERTopicModel() if topic_model is None else topic_model

        # State variables of BERTrend
        self._is_fitted = False
        self._are_models_merged = False

        # Variables related to time-based topic models
        # - topic_models: Dictionary of trained BERTopic models for each timestamp.
        self.topic_models: dict[pd.Timestamp, BERTopic] = {}
        # - doc_groups: Dictionary of document groups for each timestamp.
        self.doc_groups: dict[pd.Timestamp, list[str]] = {}
        # - emb_groups: Dictionary of document embeddings for each timestamp.
        self.emb_groups: dict[pd.Timestamp, np.ndarray] = {}

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
        """
        config = load_toml_config(self.config_file)
        return config

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

        Args:
            period (pd.Timestamp): Timestamp of the time period
            group (pd.DataFrame): Group of data associated to that timestamp.
            embedding_model (SentenceTransformer): Sentence transformer model for embeddings.
            embeddings (np.ndarray): Precomputed document embeddings.


        Returns:
            Tuple[BERTopic, List[str], np.ndarray]:
                - topic_model: trained BERTopic models for this period.
                - doc_group: document groups for this period.
                - emb_group: document embeddings for this period.
        """
        docs = group[TEXT_COLUMN].tolist()
        embeddings_subset = embeddings[group.index]

        logger.debug(f"Processing period: {period}")
        logger.debug(f"Number of documents: {len(docs)}")

        logger.debug("Creating topic model...")
        topic_model = self.topic_model.fit(
            docs=docs,
            embedding_model=embedding_model,
            embeddings=embeddings_subset,
        ).topic_model

        logger.debug("Topic model created successfully")

        doc_info_df = topic_model.get_document_info(docs=docs)
        doc_info_df = doc_info_df.rename(columns={"Document": "Paragraph"})
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
        embedding_model: SentenceTransformer,
        embeddings: np.ndarray,
    ):
        """
        Train BERTopic models for each timestamp in the grouped data.

        Stores Tuple[Dict[pd.Timestamp, BERTopic], Dict[pd.Timestamp, List[str]], Dict[pd.Timestamp, np.ndarray]]:
            - topic_models: Dictionary of trained BERTopic models for each timestamp.
            - doc_groups: Dictionary of document groups for each timestamp.
            - emb_groups: Dictionary of document embeddings for each timestamp.

        Args:
            grouped_data (Dict[pd.Timestamp, pd.DataFrame]): Dictionary of grouped data by timestamp.
            embedding_model (SentenceTransformer): Sentence transformer model for embeddings.
            embeddings (np.ndarray): Precomputed document embeddings.
        """
        topic_models = {}
        doc_groups = {}
        emb_groups = {}

        non_empty_groups = [
            (period, group) for period, group in grouped_data.items() if not group.empty
        ]

        # Set up progress bar
        # TODO: tqdm
        # progress_bar = st.progress(0)
        # progress_text = st.empty()

        for i, (period, group) in enumerate(non_empty_groups):
            try:
                logger.info(f"Training topic model {i+1}/{len(non_empty_groups)}...")
                topic_models[period], doc_groups[period], emb_groups[period] = (
                    self._train_by_period(period, group, embedding_model, embeddings)
                )  # TODO: parallelize?
                logger.debug(f"Successfully processed period: {period}")

            except Exception as e:
                logger.error(f"Error processing period {period}: {str(e)}")
                logger.exception("Traceback:")
                continue  # TODO: better error handling

            # Update progress bar
            """
            progress = (i + 1) / len(non_empty_groups)
            progress_bar.progress(progress)
            progress_text.text(
                f"Training BERTopic model for {period} ({i + 1}/{len(non_empty_groups)})"
            )
            """

        self._is_fitted = True

        # Merge the newly obtained topic models with new ones
        # Update topic_models: Dictionary of trained BERTopic models for each timestamp.
        self.topic_models.update(topic_models)
        # Update doc_groups: Dictionary of document groups for each timestamp.
        self.doc_groups.update(doc_groups)
        # Update emb_groups: Dictionary of document embeddings for each timestamp.
        self.emb_groups.update(emb_groups)
        logger.success("Finished training all topic models")

    def merge_all_models(
        self,
        min_similarity: int | None = None,
    ):
        """Merge together all topic models."""
        # Get default BERTrend config if argument is not provided
        if min_similarity is None:
            min_similarity = self.config["min_similarity"]

        # Check if model is fitted
        if not self._is_fitted:
            raise RuntimeError("You must fit the BERTrend model before merging models.")

        topic_dfs = {
            period: _preprocess_model(
                model, self.doc_groups[period], self.emb_groups[period]
            )
            for period, model in self.topic_models.items()
        }

        timestamps = sorted(topic_dfs.keys())

        if len(self.topic_models) < 2:  # beginning of the process, no real merge needed
            logger.warning("This function requires at least two topic models. Ignored")
            self._are_models_merged = False
            return

        assert len(self.topic_models) >= 2

        merged_df_without_outliers = None
        all_merge_histories = []
        all_new_topics = []

        # TODO: tqdm
        merge_df_size_over_time = []

        for i, (current_timestamp, next_timestamp) in tqdm(
            enumerate(zip(timestamps[:-1], timestamps[1:]))
        ):
            df1 = topic_dfs[current_timestamp][
                topic_dfs[current_timestamp]["Topic"] != -1
            ]
            df2 = topic_dfs[next_timestamp][topic_dfs[next_timestamp]["Topic"] != -1]

            if merged_df_without_outliers is None:
                if not (df1.empty or df2.empty):
                    (
                        merged_df_without_outliers,
                        merge_history,
                        new_topics,
                    ) = _merge_models(
                        df1,
                        df2,
                        min_similarity=min_similarity,
                        timestamp=current_timestamp,
                    )
            elif not df2.empty:
                (
                    merged_df_without_outliers,
                    merge_history,
                    new_topics,
                ) = _merge_models(
                    merged_df_without_outliers,
                    df2,
                    min_similarity=min_similarity,
                    timestamp=current_timestamp,
                )
            else:
                continue

            all_merge_histories.append(merge_history)
            all_new_topics.append(new_topics)
            merge_df_size_over_time = merge_df_size_over_time
            merge_df_size_over_time.append(
                (
                    current_timestamp,
                    merged_df_without_outliers["Topic"].max() + 1,
                )
            )

            # progress_bar.progress((i + 1) / len(timestamps))

        all_merge_histories_df = pd.concat(all_merge_histories, ignore_index=True)
        all_new_topics_df = pd.concat(all_new_topics, ignore_index=True)

        self.merged_df = merged_df_without_outliers
        self.all_merge_histories_df = all_merge_histories_df
        self.all_new_topics_df = all_new_topics_df

        logger.success("All models merged successfully")
        self._are_models_merged = True

    def calculate_signal_popularity(
        self,
        decay_factor: float | None = None,
        decay_power: float | None = None,
    ):
        """
        Compute the popularity of signals (topics) over time, accounting for merges and applying decay.
        Updates:
           - topic_sizes (Dict[int, Dict[str, Any]]): Dictionary storing topic sizes and related information over time.
           - topic_last_popularity (Dict[int, float]): Dictionary storing the last known popularity of each topic.
           - topic_last_update (Dict[int, pd.Timestamp]): Dictionary storing the last update timestamp of each topic.

        Args:
            all_merge_histories_df (pd.DataFrame): DataFrame containing all merge histories.
            granularity (int): Granularity of the timestamps in days.
            decay_factor (float): Factor for exponential decay calculation.
            decay_power (float): Power for exponential decay calculation.

        Returns:

        """
        # Get default BERTrend config if argument is not provided
        if decay_factor is None:
            decay_factor = self.config["decay_factor"]
        if decay_power is None:
            decay_power = self.config["decay_power"]

        # Check if models are merged
        if not self._are_models_merged:
            raise RuntimeWarning(
                "You must merge topic models first before computing signal popularity."
            )

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
        Computes the popularity values and thresholds for the considered time window.

        Args:
            window_size (int): The retrospective window size in days.
            current_date (datetime): The current date selected by the user.

        Returns:
            Tuple[Timestamp,Timestamp, list, float, float,]:
                window_start, window_end indicates the start / end periods.
                all_popularities_values
                The q1 and q3 values representing the 10th and 90th percentiles of popularity values,
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

        Args:
            window_start (pd.Timestamp): The start timestamp of the window.
            window_end (pd.Timestamp): The end timestamp of the window.
            q1 (float): The 10th percentile of popularity values.
            q3 (float): The 50th percentile of popularity values.
            rising_popularity_only (bool): Whether to consider only rising popularity topics as weak signals.
            keep_documents (bool): Whether to keep track of the documents or not.

        Returns:
            Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
                - noise_topics_df: DataFrame containing noise topics.
                - weak_signal_topics_df: DataFrame containing weak signal topics.
                - strong_signal_topics_df: DataFrame containing strong signal topics.
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

        Args:
            window_size (int): The retrospective window size in days.
            current_date (datetime): The current date selected by the user.


        Returns:
            Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
                - noise_topics_df: DataFrame containing noise topics.
                - weak_signal_topics_df: DataFrame containing weak signal topics.
                - strong_signal_topics_df: DataFrame containing strong signal topics.
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

    def save_models(self, models_path: Path = MODELS_DIR):
        if models_path.exists():
            shutil.rmtree(models_path)
        models_path.mkdir(parents=True, exist_ok=True)

        # Save topic models using the selected serialization type
        for period, topic_model in self.topic_models.items():
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

        # Save topic model parameters
        with open(CACHE_PATH / HYPERPARAMS_FILE, "wb") as f:
            pickle.dump(self.topic_model, f)
        # Save doc_groups file
        with open(CACHE_PATH / DOC_GROUPS_FILE, "wb") as f:
            pickle.dump(self.doc_groups, f)
        # Save emb_groups file
        with open(CACHE_PATH / EMB_GROUPS_FILE, "wb") as f:
            pickle.dump(self.emb_groups, f)
        # Save the models_trained flag
        with open(CACHE_PATH / MODELS_TRAINED_FILE, "wb") as f:
            pickle.dump(self._is_fitted, f)

        logger.info(f"Models saved to: {models_path}")

    @classmethod
    def restore_models(cls, models_path: Path = MODELS_DIR) -> "BERTrend":
        if not models_path.exists():
            raise FileNotFoundError(f"models_path={models_path} does not exist")

        logger.info(f"Loading models from: {models_path}")

        # Create BERTrend object
        bertrend = cls()

        # load topic model parameters
        with open(CACHE_PATH / HYPERPARAMS_FILE, "rb") as f:
            bertrend.topic_model = pickle.load(f)
        # load doc_groups file
        with open(CACHE_PATH / DOC_GROUPS_FILE, "rb") as f:
            bertrend.doc_groups = pickle.load(f)
        # load emb_groups file
        with open(CACHE_PATH / EMB_GROUPS_FILE, "rb") as f:
            bertrend.emb_groups = pickle.load(f)
        # load the models_trained flag
        with open(CACHE_PATH / MODELS_TRAINED_FILE, "rb") as f:
            bertrend._is_fitted = pickle.load(f)

        # Restore topic models using the selected serialization type
        topic_models = {}
        for period_dir in models_path.iterdir():
            if period_dir.is_dir():
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

                period = pd.Timestamp(period_dir.name.replace("_", ":"))
                topic_models[period] = topic_model
        bertrend.topic_models = topic_models

        return bertrend

    def save_signal_evolution_data(
        self,
        window_size: int,
        start_timestamp: pd.Timestamp,
        end_timestamp: pd.Timestamp,
    ) -> Path:
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


def _preprocess_model(
    topic_model: BERTopic, docs: list[str], embeddings: np.ndarray
) -> pd.DataFrame:
    """
    Preprocess a BERTopic model by extracting topic information, document groups, document embeddings, and URLs.

    Args:
        topic_model (BERTopic): A fitted BERTopic model.
        docs (List[str]): List of documents.
        embeddings (np.ndarray): Precomputed document embeddings.

    Returns:
        pd.DataFrame: A DataFrame with topic information, document groups, document embeddings, and URLs.
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
