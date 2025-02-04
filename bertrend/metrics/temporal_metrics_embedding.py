"""
TempTopic: An addon to BERTopic that evaluates the model's dynamic topic modeling using embeddings

The TempTopic class extends BERTopic for dynamic topic modeling evaluation, incorporating metrics such as 
Temporal Topic Representation Stability and Temporal Topic Embedding Stability. This approach provides a 
comprehensive analysis of how topics evolve over time, focusing on their embeddings and vocabulary consistency.

Key Features:
- Temporal Topic Representation Stability: Assesses the stability of topic representations over different timestamps.
- Temporal Topic Embedding Stability: Evaluates the consistency of topic embeddings over time.
- Overall Topic Stability: Combines representation stability and embedding stability for a comprehensive stability measure.
- Visualization: Various plotting functions to visualize topic evolution, stability metrics, and overall topic stability.

Requirements:
- A trained BERTopic model
- A list of documents and their corresponding timestamps
- Embeddings for documents and words
- Optionally, a list of pre-assigned topics for each document

Example Usage:

```python
from bertopic import BERTopic
from temporal_metrics_embedding import TempTopic

# Assuming `documents`, `embeddings`, `word_embeddings`, `token_strings`, and `timestamps` are prepared

topic_model = BERTopic()
topics, probs = topic_model.fit_transform(documents)

# Initialize TempTopic with the BERTopic model, documents, embeddings, and timestamps
metrics = TempTopic(topic_model=topic_model, docs=documents, embeddings=embeddings,
                      word_embeddings=word_embeddings, token_strings=token_strings, timestamps=timestamps)

# Fit the TempTopic model
metrics.fit(window_size=2, k=1)

# Calculate and plot stability metrics
metrics.calculate_temporal_representation_stability()
metrics.calculate_topic_embedding_stability()
"""

#  Copyright (c) 2024, RTE (https://www.rte-france.com)
#  See AUTHORS.txt
#  SPDX-License-Identifier: MPL-2.0
#  This file is part of BERTrend.

import pandas as pd
from loguru import logger
import numpy as np
from thefuzz import fuzz
from sklearn.metrics.pairwise import cosine_similarity
from nltk.stem import PorterStemmer
from bertopic import BERTopic
from tqdm import tqdm
import itertools
from scipy.sparse import csr_matrix
from sklearn.preprocessing import normalize
import re

from bertrend import OUTPUT_PATH


class TempTopic:
    def __init__(
        self,
        topic_model: BERTopic,
        docs: list[str],
        embeddings: list[list[float]],
        word_embeddings: list[list[list[float]]],
        token_strings: list[list[str]],
        timestamps: list[str] | list[int],
        topics: list[int] = None,
        evolution_tuning: bool = True,
        global_tuning: bool = False,
    ):
        """
        Initializes the TempTopic object with a BERTopic model, a list of documents, embeddings, and timestamps.

        Parameters:
        - topic_model: A trained BERTopic model.
        - docs: A list of documents (strings).
        - embeddings: List of document embeddings.
        - word_embeddings: List of word embeddings.
        - token_strings: List of token strings corresponding to the word embeddings.
        - timestamps: A list of timestamps corresponding to each document. The list can contain strings or integers.
        - topics: An optional list of topics corresponding to each document.
        - evolution_tuning: Boolean to fine-tune the c-TF-IDF matrix at timestamp t by averaging it with the c-TF-IDF at t-1.
        - global_tuning: Boolean indicating whether to apply global tuning to align topics with the global c-TF-IDF representation.
        """
        if not isinstance(topic_model, BERTopic):
            raise TypeError("topic_model must be an instance of BERTopic.")
        if not isinstance(docs, list) or not all(isinstance(doc, str) for doc in docs):
            raise TypeError("docs must be a list of strings.")
        if not isinstance(timestamps, list) or not all(
            isinstance(t, (str, int, float)) for t in timestamps
        ):
            raise TypeError("timestamps must be a list of str, int or float.")
        if topics is not None and (
            not isinstance(topics, list)
            or not all(isinstance(topic, int) for topic in topics)
        ):
            raise TypeError("topics, if provided, must be a list of integers.")

        # Ensure all inputs have the same length
        if topics is not None and not (len(docs) == len(timestamps) == len(topics)):
            raise ValueError(
                "Lengths of docs, timestamps, and topics must all be the same."
            )
        elif not (len(docs) == len(timestamps)):
            raise ValueError("Lengths of docs and timestamps must be the same.")

        self.topic_model = topic_model
        self.docs = docs
        self.embeddings = embeddings
        self.word_embeddings = word_embeddings
        self.token_strings = token_strings
        self.timestamps = timestamps
        self.topics = topics if topics is not None else self.topic_model.topics_
        self.evolution_tuning = evolution_tuning
        self.global_tuning = global_tuning

        self.final_df = None
        self.representation_embeddings_df = None
        self.stemmer = PorterStemmer()

        self.debug_file = OUTPUT_PATH / "match_debugging.txt"
        open(self.debug_file, "w").close()

    def fit(
        self,
        window_size: int = 2,
        k: int = 1,
        double_agg: bool = True,
        doc_agg: str = "mean",
        global_agg: str = "max",
    ):
        """
        Fits the TempTopic model to calculate and store topic dynamics over time.

        Parameters:
        - window_size: Size of the window for temporal analysis.
        - k: Number of nearest neighbors for stability calculation.
        - double_agg: Boolean to apply double aggregation.
        - doc_agg: Aggregation method for document embeddings.
        - global_agg: Aggregation method for global embeddings.
        """
        self._topics_over_time()
        self._calculate_representation_embeddings(
            double_agg=double_agg, doc_agg=doc_agg, global_agg=global_agg
        )
        self.calculate_temporal_representation_stability(window_size=window_size, k=k)
        self.calculate_topic_embedding_stability(window_size=window_size)

    def _topics_over_time(self) -> pd.DataFrame:
        """
        Extends the existing method to include document embeddings and the mean topic embedding for each topic at each timestamp.

        Returns:
        A pandas DataFrame containing topics, their top words, frequencies, timestamps, and embeddings.
        """
        documents = pd.DataFrame(
            {
                "Document": self.docs,
                "Timestamps": self.timestamps,
                "Topic": self.topics,
                "Document_Embeddings": list(self.embeddings),
                "Token_Embeddings": self.word_embeddings,
                "Token_Strings": self.token_strings,
            }
        )

        # Remove rows associated with outlier topic
        documents = documents[documents["Topic"] != -1]

        # Normalize the global c-TF-IDF representation for tuning purposes
        global_c_tf_idf = normalize(
            self.topic_model.c_tf_idf_, axis=1, norm="l1", copy=False
        )

        # Ensure all topics are processed in order
        all_topics = sorted(list(documents.Topic.unique()))
        all_topics_indices = {topic: index for index, topic in enumerate(all_topics)}

        # Sort documents by their timestamps for sequential processing
        documents.sort_values("Timestamps", inplace=True)
        timestamps = documents["Timestamps"].unique()

        topics_over_time = []  # Accumulates the final data for each timestamp
        document_per_topic_list = (
            []
        )  # Tracks documents associated with each topic at each timestamp

        previous_c_tf_idf = None
        previous_topics = None

        for index, timestamp in tqdm(
            enumerate(timestamps), desc="Processing timestamps"
        ):
            # Select documents for the current timestamp
            selection = documents.loc[documents.Timestamps == timestamp, :]
            # Aggressively preprocess text before vocabulary extraction through custom function for French language
            selection["Document"] = selection["Document"].apply(
                self._aggressive_text_preprocessing
            )

            # Aggregate documents by topic to compute c-TF-IDF and collect embeddings
            topic_aggregated_docs_for_ctfidf = selection.groupby(
                ["Topic"], as_index=False
            ).agg(
                {
                    "Document": " ".join,  # Combine documents for each topic
                    "Timestamps": "count",  # Count of documents per topic
                    "Document_Embeddings": lambda x: list(
                        x
                    ),  # Collect embeddings in a list
                    "Token_Embeddings": lambda x: list(
                        itertools.chain.from_iterable(x)
                    ),  # Flatten and collect token embeddings
                    "Token_Strings": lambda x: list(
                        itertools.chain.from_iterable(x)
                    ),  # Flatten and collect token strings
                }
            )

            # Compute c-TF-IDF for the current selection
            c_tf_idf, words = self.topic_model._c_tf_idf(
                topic_aggregated_docs_for_ctfidf, fit=False
            )

            # Aggregate documents to calculate mean embeddings for each topic
            detailed_topic_aggregated_docs = selection.groupby(
                "Topic", as_index=False
            ).agg(
                {
                    "Document": lambda docs: list(docs),
                    "Document_Embeddings": lambda x: list(
                        x
                    ),  # Collect embeddings in a list
                    "Token_Embeddings": lambda x: list(
                        x
                    ),  # Collect token embeddings in a list
                    "Token_Strings": lambda x: list(
                        x
                    ),  # Collect token strings in a list
                }
            )
            detailed_topic_aggregated_docs["Timestamp"] = timestamp

            # Calculate mean embeddings for each topic
            detailed_topic_aggregated_docs["Embedding"] = (
                detailed_topic_aggregated_docs["Document_Embeddings"].apply(
                    lambda x: np.mean(x, axis=0)
                )
            )

            document_per_topic_list.append(detailed_topic_aggregated_docs)

            # Evolution tuning
            if self.evolution_tuning and index != 0 and previous_c_tf_idf is not None:
                current_topics = sorted(
                    list(topic_aggregated_docs_for_ctfidf.Topic.values)
                )
                overlapping_topics = sorted(
                    list(set(previous_topics).intersection(set(current_topics)))
                )
                current_overlap_idx = [
                    current_topics.index(topic) for topic in overlapping_topics
                ]
                previous_overlap_idx = [
                    previous_topics.index(topic) for topic in overlapping_topics
                ]

                c_tf_idf_lil = c_tf_idf.tolil()
                c_tf_idf_lil[current_overlap_idx] = (
                    (
                        c_tf_idf[current_overlap_idx]
                        + previous_c_tf_idf[previous_overlap_idx]
                    )
                    / 2.0
                ).tolil()
                c_tf_idf = csr_matrix(c_tf_idf_lil)

            # Global tuning
            if self.global_tuning:
                selected_topics = [
                    all_topics_indices[topic]
                    for topic in topic_aggregated_docs_for_ctfidf.Topic.values
                ]
                c_tf_idf = (global_c_tf_idf[selected_topics] + c_tf_idf) / 2.0

            # Extract the words per topic
            # IMPORTANT: Pass only document and topic columns, other columns with nested data such as token_embeddings will
            # result in a TypeError: unhashable type: 'numpy.ndarray' from Pandas
            # Removing the other columns doesn't change anything, since they'll be used later in the code and not here
            words_per_topic = self.topic_model._extract_words_per_topic(
                words,
                selection[["Document", "Topic"]],
                c_tf_idf,
                calculate_aspects=False,
            )
            topic_frequency = pd.Series(
                topic_aggregated_docs_for_ctfidf.Timestamps.values,
                index=topic_aggregated_docs_for_ctfidf.Topic,
            ).to_dict()

            # Fill dataframe with results, now including document embeddings and the mean topic embedding
            topics_at_timestamp = [
                (
                    topic,
                    ", ".join([word for word, _ in values]),
                    topic_frequency[topic],
                    timestamp,
                    detailed_topic_aggregated_docs.loc[
                        detailed_topic_aggregated_docs["Topic"] == topic,
                        "Document_Embeddings",
                    ].values[0],
                    detailed_topic_aggregated_docs.loc[
                        detailed_topic_aggregated_docs["Topic"] == topic, "Embedding"
                    ].values[0],
                    detailed_topic_aggregated_docs.loc[
                        detailed_topic_aggregated_docs["Topic"] == topic,
                        "Token_Embeddings",
                    ].values[0],
                    detailed_topic_aggregated_docs.loc[
                        detailed_topic_aggregated_docs["Topic"] == topic,
                        "Token_Strings",
                    ].values[0],
                )
                for topic, values in words_per_topic.items()
            ]

            topics_over_time.extend(topics_at_timestamp)

            previous_topics = sorted(
                list(topic_aggregated_docs_for_ctfidf.Topic.values)
            )
            previous_c_tf_idf = c_tf_idf.copy()

        columns = [
            "Topic",
            "Words",
            "Frequency",
            "Timestamp",
            "Document_Embeddings",
            "Embedding",
            "Token_Embeddings",
            "Token_Strings",
        ]
        self.final_df = pd.DataFrame(topics_over_time, columns=columns)

        return self.final_df

    def _fuzzy_match_and_embed(
        self,
        phrase: str,
        token_strings: list[list[str]],
        token_embeddings: list[np.ndarray],
        topic_id: int,
        timestamp: str,
        window_size: int,
    ) -> tuple[str, np.ndarray]:
        """
        Matches a phrase to the most similar window in token_strings using fuzzy matching and returns the corresponding embedding.

        Parameters:
        - phrase: The phrase to match.
        - token_strings: List of token strings.
        - token_embeddings: List of token embeddings.
        - topic_id: The topic ID.
        - timestamp: The timestamp of the topic.
        - window_size: The size of the window for fuzzy matching.

        Returns:
        - Tuple containing the best matched phrase and its embedding.
        """
        phrase_tokens = phrase.split()
        best_match = None
        best_score = 0
        best_embedding = None

        for doc_idx, doc_tokens in enumerate(token_strings):
            for i in range(len(doc_tokens) - len(phrase_tokens) + 1):
                window = doc_tokens[i : i + len(phrase_tokens)]
                window_str = " ".join(window)

                score = fuzz.token_set_ratio(phrase, window_str)

                if score > best_score:
                    best_score = score
                    best_match = window_str
                    best_embedding = np.mean(
                        token_embeddings[doc_idx][i : i + len(phrase_tokens)], axis=0
                    )

        if best_score > 80:
            return best_match, best_embedding
        else:
            self._log_failed_match(
                phrase, token_strings, topic_id, timestamp, best_match, best_score
            )
            return None, None

    def _log_failed_match(
        self,
        phrase: str,
        token_strings: list[list[str]],
        topic_id: int,
        timestamp: str,
        best_match: str,
        best_score: int,
    ):
        """
        Logs failed matches for debugging purposes.

        Parameters:
        - phrase: The phrase that failed to match.
        - token_strings: List of token strings.
        - topic_id: The topic ID.
        - timestamp: The timestamp of the topic.
        - best_match: The best matched phrase.
        - best_score: The best score achieved.

        Due to the way the _extract_words_per_topic in BERTopic is coded, more specifically this part :
            ...
            # Get top 30 words per topic based on c-TF-IDF score
            topics = {label: [(words[word_index], score)
                            if word_index is not None and score > 0
                            else ("", 0.00001)
                            for word_index, score in zip(indices[index][::-1], scores[index][::-1])
                            ]
                    for index, label in enumerate(labels)}
            ...
        Sometimes, empty strings get returned as a representation of a topic at a specific timestamp, but
        is gracefully handled in the TempTopic class.

        """

        with open(self.debug_file, "a", encoding="utf-8") as f:
            f.write(f"{'#'*50}\n")
            f.write(f"Failed match for Topic {topic_id} at Timestamp {timestamp}\n")
            f.write(f"Phrase to match: '{phrase}'\n")
            f.write(f"Best match: '{best_match}' with score {best_score}\n\n")
            f.write("Documents for this topic and timestamp:\n")
            for idx, doc in enumerate(token_strings):
                f.write(f"Document {idx}:\n")
                f.write(" ".join(doc) + "\n\n")
            f.write(f"{'#'*50}\n\n")

    def _calculate_representation_embeddings(
        self,
        double_agg: bool = True,
        doc_agg: str = "mean",
        global_agg: str = "max",
        window_size: int = 10,
    ):
        """
        Calculates embeddings for topic representations using fuzzy matching.

        Parameters:
        - double_agg: Boolean to apply double aggregation.
        - doc_agg: Aggregation method for document embeddings.
        - global_agg: Aggregation method for global embeddings.
        - window_size: The size of the window for fuzzy matching.
        """
        representation_embeddings = []

        for _, row in self.final_df.iterrows():
            topic_id = row["Topic"]
            timestamp = row["Timestamp"]
            representation = [phrase.lower() for phrase in row["Words"].split(", ")]

            token_strings = row["Token_Strings"]
            token_embeddings = row["Token_Embeddings"]

            embedding_list = []
            updated_representation = []

            for phrase in representation:
                matched_phrase, embedding = self._fuzzy_match_and_embed(
                    phrase,
                    token_strings,
                    token_embeddings,
                    topic_id,
                    timestamp,
                    window_size,
                )
                if embedding is not None:
                    embedding_list.append(embedding)
                    updated_representation.append(matched_phrase)
                else:
                    logger.warning(
                        f"No embedding found for '{phrase}' in topic {topic_id} at timestamp {timestamp}"
                    )

            if not embedding_list:
                logger.warning(
                    f"No valid embeddings found for topic {topic_id} at timestamp {timestamp}. Using zero embedding and empty representation."
                )
                embedding_list = [
                    np.zeros(self.embeddings[0].shape)
                ]  # Add a zero embedding
                updated_representation = [""]  # Add an empty representation

            representation_embeddings.append(
                {
                    "Topic ID": topic_id,
                    "Timestamp": timestamp,
                    "Representation": ", ".join(updated_representation),
                    "Representation Embeddings": embedding_list,
                }
            )

        self.representation_embeddings_df = pd.DataFrame(representation_embeddings)
        logger.info(
            f"Created representation_embeddings_df with shape {self.representation_embeddings_df.shape}"
        )
        logger.info(
            f"Detailed debugging information for failed matches has been written to {self.debug_file}"
        )

        # Consistency check
        topics_in_final_df = set(self.final_df["Topic"].unique())
        topics_in_repr_df = set(self.representation_embeddings_df["Topic ID"].unique())

        if topics_in_final_df != topics_in_repr_df:
            missing_topics = topics_in_final_df - topics_in_repr_df
            extra_topics = topics_in_repr_df - topics_in_final_df
            logger.warning(
                f"Discrepancy in topics: {len(missing_topics)} topics missing from representation_embeddings_df, {len(extra_topics)} extra topics in representation_embeddings_df"
            )
            logger.warning(f"Missing topics: {missing_topics}")
            logger.warning(f"Extra topics: {extra_topics}")

    def calculate_temporal_representation_stability(
        self, window_size: int = 2, k: int = 1
    ) -> tuple[pd.DataFrame, float]:
        """
        Calculates the Temporal Representation Stability (TRS) scores for each topic.

        Parameters:
        - window_size: Size of the window for temporal analysis.
        - k: Number of nearest neighbors for stability calculation.

        Returns:
        - Tuple containing a DataFrame with TRS scores and the average TRS score.
        """
        if window_size < 2:
            raise ValueError("window_size must be 2 or above.")

        stability_scores = []
        topic_counts = self.final_df.groupby("Topic")["Timestamp"].nunique()
        valid_topics = set(topic_counts[topic_counts > 1].index.tolist())
        grouped_topics = self.representation_embeddings_df[
            self.representation_embeddings_df["Topic ID"].isin(valid_topics)
        ].groupby("Topic ID")
        all_topics = set(self.final_df["Topic"].unique())
        processed_topics = set()

        for topic_id, group in grouped_topics:
            processed_topics.add(topic_id)
            sorted_group = group.sort_values("Timestamp")

            for i in range(len(sorted_group) - window_size + 1):
                start_row = sorted_group.iloc[i]
                end_row = sorted_group.iloc[i + window_size - 1]

                start_embeddings = start_row["Representation Embeddings"]
                end_embeddings = end_row["Representation Embeddings"]

                if len(start_embeddings) == 0 or len(end_embeddings) == 0:
                    logger.warning(
                        f"Skipping topic {topic_id} for timestamps {start_row['Timestamp']} to {end_row['Timestamp']} due to empty embeddings."
                    )
                    continue

                similarity_scores = []
                for start_embedding in start_embeddings:
                    start_embedding = np.array(start_embedding).reshape(1, -1)
                    end_embeddings_2d = np.array(end_embeddings).reshape(
                        len(end_embeddings), -1
                    )

                    cosine_similarities = cosine_similarity(
                        start_embedding, end_embeddings_2d
                    )[0]
                    top_k_indices = cosine_similarities.argsort()[-k:][::-1]
                    top_k_similarities = cosine_similarities[top_k_indices]
                    similarity_scores.extend(top_k_similarities)

                avg_similarity = np.mean(similarity_scores)

                stability_scores.append(
                    {
                        "Topic ID": topic_id,
                        "Start Timestamp": start_row["Timestamp"],
                        "End Timestamp": end_row["Timestamp"],
                        "Start Representation": start_row["Representation"],
                        "End Representation": end_row["Representation"],
                        "Representation Stability Score": avg_similarity,
                    }
                )

        # Add empty entries for topics with no representation stability scores
        missing_topics = all_topics - processed_topics
        for topic_id in missing_topics:
            logger.warning(
                f"Topic {topic_id} has no representation stability scores. Adding empty entry."
            )
            stability_scores.append(
                {
                    "Topic ID": topic_id,
                    "Start Timestamp": None,
                    "End Timestamp": None,
                    "Start Representation": None,
                    "End Representation": None,
                    "Representation Stability Score": 0.0,  # or np.nan if you prefer
                }
            )

        self.representation_stability_scores_df = pd.DataFrame(stability_scores)
        self.avg_representation_stability_score = (
            self.representation_stability_scores_df[
                "Representation Stability Score"
            ].mean()
        )

        logger.info(
            f"Calculated representation stability for {len(processed_topics)} topics. {len(missing_topics)} topics had no scores."
        )

        return (
            self.representation_stability_scores_df,
            self.avg_representation_stability_score,
        )

    def calculate_topic_embedding_stability(
        self, window_size: int = 2
    ) -> tuple[pd.DataFrame, float]:
        """
        Calculates the Temporal Topic Embedding Stability (TTES) scores for each topic.

        Parameters:
        - window_size: Size of the window for temporal analysis.

        Returns:
        - Tuple containing a DataFrame with TTES scores and the average TTES score.
        """
        if window_size < 2:
            raise ValueError("window_size must be 2 or above.")

        stability_scores = []
        topic_counts = self.final_df.groupby("Topic")["Timestamp"].nunique()
        valid_topics = set(topic_counts[topic_counts > 1].index.tolist())
        grouped_topics = self.final_df[
            self.final_df["Topic"].isin(valid_topics)
        ].groupby("Topic")
        all_topics = set(self.final_df["Topic"].unique())
        processed_topics = set()

        for topic_id, group in grouped_topics:
            processed_topics.add(topic_id)
            sorted_group = group.sort_values("Timestamp")

            for i in range(len(sorted_group) - window_size + 1):
                start_row = sorted_group.iloc[i]
                end_row = sorted_group.iloc[i + window_size - 1]

                start_embedding = start_row["Embedding"]
                end_embedding = end_row["Embedding"]

                if start_embedding is None or end_embedding is None:
                    logger.warning(
                        f"Skipping topic {topic_id} for timestamps {start_row['Timestamp']} to {end_row['Timestamp']} due to missing embeddings."
                    )
                    continue

                similarity = cosine_similarity([start_embedding], [end_embedding])[0][0]

                stability_scores.append(
                    {
                        "Topic ID": topic_id,
                        "Start Timestamp": start_row["Timestamp"],
                        "End Timestamp": end_row["Timestamp"],
                        "Topic Stability Score": similarity,
                    }
                )

        # Add empty entries for topics with no embedding stability scores
        missing_topics = all_topics - processed_topics
        for topic_id in missing_topics:
            logger.warning(
                f"Topic {topic_id} has no embedding stability scores. Adding empty entry."
            )
            stability_scores.append(
                {
                    "Topic ID": topic_id,
                    "Start Timestamp": None,
                    "End Timestamp": None,
                    "Topic Stability Score": 0.0,  # or np.nan if you prefer
                }
            )

        self.topic_stability_scores_df = pd.DataFrame(stability_scores)
        self.avg_topic_stability_score = self.topic_stability_scores_df[
            "Topic Stability Score"
        ].mean()

        logger.info(
            f"Calculated topic embedding stability for {len(processed_topics)} topics. {len(missing_topics)} topics had no scores."
        )

        return self.topic_stability_scores_df, self.avg_topic_stability_score

    def calculate_overall_topic_stability(
        self, window_size: int = 2, k: int = 1, alpha: float = 0.5
    ) -> pd.DataFrame:
        """Calculates the Overall Topic Stability (OTS) score by combining representation stability and embedding stability.

        Parameters:
        - window_size: Size of the window for temporal analysis.
        - k: Number of nearest neighbors for stability calculation.
        - alpha: Weight for combining representation and embedding stability.

        Returns:
        - DataFrame containing the overall stability scores.
        """
        (
            representation_stability_df,
            _,
        ) = self.calculate_temporal_representation_stability(window_size, k)
        topic_stability_df, _ = self.calculate_topic_embedding_stability(window_size)

        merged_df = pd.merge(
            representation_stability_df,
            topic_stability_df,
            on=["Topic ID", "Start Timestamp", "End Timestamp"],
        )

        merged_df["Overall Stability Score"] = (
            alpha * merged_df["Topic Stability Score"]
            + (1 - alpha) * merged_df["Representation Stability Score"]
        )

        topic_timestamps = (
            merged_df.groupby("Topic ID")["Start Timestamp"].nunique() + 1
        )

        self.overall_stability_df = (
            merged_df.groupby("Topic ID")["Overall Stability Score"]
            .mean()
            .reset_index()
        )
        self.overall_stability_df["Number of Timestamps"] = topic_timestamps

        min_score = self.overall_stability_df["Overall Stability Score"].min()
        max_score = self.overall_stability_df["Overall Stability Score"].max()
        self.overall_stability_df["Normalized Stability Score"] = (
            self.overall_stability_df["Overall Stability Score"] - min_score
        ) / (max_score - min_score)

        return self.overall_stability_df

    def find_similar_topic_pairs(
        self, similarity_threshold: float = 0.8
    ) -> list[list[tuple[int, int, str]]]:
        """
        Finds similar topic pairs based on cosine similarity.

        Parameters:
        - similarity_threshold: Threshold for cosine similarity to consider topics as similar.

        Returns:
        - List of similar topic pairs.
        """
        topic_ids = self.final_df["Topic"].unique()
        num_topics = len(topic_ids)

        similar_topic_pairs = []

        for i in range(num_topics):
            for j in range(i + 1, num_topics):
                topic_i = topic_ids[i]
                topic_j = topic_ids[j]

                topic_i_data = self.final_df[self.final_df["Topic"] == topic_i]
                topic_j_data = self.final_df[self.final_df["Topic"] == topic_j]

                common_timestamps = set(topic_i_data["Timestamp"]) & set(
                    topic_j_data["Timestamp"]
                )

                for timestamp in common_timestamps:
                    embedding_i = topic_i_data[topic_i_data["Timestamp"] == timestamp][
                        "Embedding"
                    ].iloc[0]
                    embedding_j = topic_j_data[topic_j_data["Timestamp"] == timestamp][
                        "Embedding"
                    ].iloc[0]

                    similarity = cosine_similarity([embedding_i], [embedding_j])[0][0]

                    if similarity >= similarity_threshold:
                        topic_pair = (topic_i, topic_j, timestamp)

                        if not any(
                            topic_pair in sublist for sublist in similar_topic_pairs
                        ):
                            similar_topic_pairs.append([topic_pair])
                        else:
                            for sublist in similar_topic_pairs:
                                if any(pair[:2] == topic_pair[:2] for pair in sublist):
                                    sublist.append(topic_pair)
                                    break

        return similar_topic_pairs

    def _aggressive_text_preprocessing(self, text: str) -> str:
        """
        Aggressively preprocess French text by:
        - Replacing hyphens and similar characters with spaces
        - Removing specific prefixes
        - Removing all punctuation
        - Replacing special characters with spaces (preserving accented characters, util Latin extensions, and newlines)
        - Normalizing superscripts and subscripts
        - Splitting words containing capitals in the middle (while avoiding splitting fully capitalized words)
        - Lowercasing all text
        - Replacing multiple spaces with a single space

        Args:
            text (str): The input French text to preprocess.

        Returns:
            str: The aggressively preprocessed French text.
        """
        # Replace hyphens and similar characters with spaces
        text = re.sub(r"\b(-|/|;|:)", " ", text)

        # Remove specific prefixes
        text = re.sub(r"\b(l'|L'|D'|d'|l’|L’|D’|d’)", " ", text)

        # Remove all punctuation (excluding newlines)
        text = re.sub(r"[^\w\s\nàâçéèêëîïôûùüÿñæœ]", " ", text)

        # Normalize superscripts and subscripts
        # Replace all superscripts and subscripts with their corresponding regular characters
        superscript_map = str.maketrans("⁰¹²³⁴⁵⁶⁷⁸⁹", "0123456789")
        subscript_map = str.maketrans("₀₁₂₃₄₅₆₇₈₉", "0123456789")
        text = text.translate(superscript_map)
        text = text.translate(subscript_map)

        # Split words that contain capitals in the middle but avoid splitting fully capitalized words
        text = re.sub(r"(?<=[a-z])(?=[A-Z])", " ", text)

        # Lowercase all text
        text = text.lower()

        # Replace multiple spaces with a single space
        text = re.sub(r"[ \t]+", " ", text)

        return text
