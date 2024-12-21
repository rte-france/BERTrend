"""
An addon to BERTopic that allows the evaluation of the model's dynamic topic modeling

Link to paper: https://arxiv.org/abs/2309.08627


The TempTopic class extends the capabilities of BERTopic for dynamic topic modeling evaluation, incorporating metrics such as 
Temporal Topic Coherence (TTC), Temporal Topic Smoothness (TTS), and Temporal Topic Quality (TTQ). This approach enables a 
comprehensive analysis of how topics evolve over time, considering their coherence, smoothness, and overall quality within a temporal context.

Key Features:
- Temporal Topic Coherence (TTC): Assesses the consistency and relevance of topics over different timestamps.
- Temporal Topic Smoothness (TTS): Evaluates the evolution of topics over time, focusing on the stability of their vocabulary.
- Temporal Topic Quality (TTQ): Combines TTC and TTS to provide an overall measure of topic quality over time, factoring in both coherence and smoothness.

Requirements:
- A trained BERTopic model
- A list of documents and their corresponding timestamps
- Optionally, a list of pre-assigned topics for each document

The TempTopic class supports additional features like evolution tuning and global tuning, which fine-tune 
the temporal analysis by adjusting the c-TF-IDF representations of topics across timestamps.

Example Usage:

```python
from bertopic import BERTopic
from temporal_metrics import TempTopic

# Assuming `documents` is your list of documents and `timestamps` is the list of their corresponding timestamps

topic_model = BERTopic()
topics, probs = topic_model.fit_transform(documents)

# Initialize TempTopic with the BERTopic model, documents, and timestamps
metrics = TempTopic(topic_model=topic_model, docs=documents, timestamps=timestamps)

# Calculate Temporal Topic Coherence
ttc_scores_df, avg_ttc = metrics.calculate_temporal_coherence()

# Calculate Temporal Topic Smoothness
tts_scores_df, avg_tts = metrics.calculate_temporal_smoothness()

# Calculate Temporal Topic Quality
ttq_scores_df, avg_ttq = metrics.calculate_temporal_quality()

# Plot temporal topic coherence
metrics.plot_temporal_topic_metrics(metric='coherence')

# Plot temporal topic smoothness
metrics.plot_temporal_topic_metrics(metric='smoothness')

# Plot temporal topic quality
metrics.plot_temporal_topic_quality()
"""

#  Copyright (c) 2024, RTE (https://www.rte-france.com)
#  See AUTHORS.txt
#  SPDX-License-Identifier: MPL-2.0
#  This file is part of BERTrend.

from sklearn.preprocessing import normalize
import pandas as pd
from typing import List, Union, Tuple, Dict
from tqdm import tqdm
from bertopic import BERTopic
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from scipy.sparse import lil_matrix
import itertools
import plotly.graph_objects as go
import plotly.express as px


class TempTopic:
    """
    An addon to BERTopic that allows the evaluation of the model's dynamic topic modeling.
    Incorporates metrics such as Temporal Topic Coherence (TTC), Temporal Topic Smoothness (TTS),
    and Temporal Topic Quality (TTQ) for comprehensive analysis of topic evolution over time.
    """

    def __init__(
        self,
        topic_model: BERTopic,
        docs: List[str],
        timestamps: Union[List[str], List[int]],
        topics: List[int] = None,
        evolution_tuning: bool = True,
        global_tuning: bool = False,
    ):
        """
        Initialize the TempTopic object with a BERTopic model, a list of documents, their timestamps,
        and optionally a list of topics.

        Parameters:
        - topic_model (BERTopic): A trained BERTopic model.
        - docs (List[str]): A list of documents (strings).
        - timestamps (Union[List[str], List[int]]): A list of timestamps corresponding to each document.
        - topics (List[int], optional): A list of topics corresponding to each document.
        - evolution_tuning (bool): Fine-tune the c-TF-IDF matrix at timestamp t by averaging it with the c-TF-IDF at t-1.
        - global_tuning (bool): Apply global tuning to align topics with the global c-TF-IDF representation.
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
        self.timestamps = timestamps
        self.topics = topics if topics is not None else self.topic_model.topics_
        self.global_tuning = global_tuning
        self.evolution_tuning = evolution_tuning

        self.final_df = None
        self.co_occurrence_matrix = None

        self.average_tts = None
        self.average_ttc = None

        self.tts_scores_df = None
        self.ttc_scores_df = None

    def _topics_over_time(self):
        """
        Calculates and sets as a property a DataFrame containing topics over time with their respective words and frequencies.

        Returns:
        - pd.DataFrame: Topics, their top words, frequencies, and timestamps.
        """
        # Prepare documents DataFrame
        documents = pd.DataFrame(
            {"Document": self.docs, "Timestamps": self.timestamps, "Topic": self.topics}
        )

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

        for index, timestamp in tqdm(enumerate(timestamps), desc="Initial processing"):
            # Select documents for the current timestamp
            selection = documents.loc[documents.Timestamps == timestamp, :]

            # Aggregate documents by topic to compute c-TF-IDF
            documents_per_topic = selection.groupby(["Topic"], as_index=False).agg(
                {
                    "Document": " ".join,  # Combine documents for each topic
                    "Timestamps": "count",  # Count of documents per topic
                }
            )

            # Compute c-TF-IDF for the current selection
            c_tf_idf, words = self.topic_model._c_tf_idf(documents_per_topic, fit=False)

            # Additional aggregation to maintain document lists per topic
            documents_per_topic_2 = selection.groupby("Topic", as_index=False).agg(
                {"Document": lambda docs: list(docs)}
            )
            documents_per_topic_2["Timestamp"] = timestamp
            document_per_topic_list.append(documents_per_topic_2)

            # Fine-tune the c-TF-IDF matrix at timestamp t by averaging it with the c-TF-IDF matrix at timestamp t-1
            if self.evolution_tuning and index != 0:
                current_topics = sorted(list(documents_per_topic.Topic.values))
                overlapping_topics = sorted(
                    list(set(previous_topics).intersection(set(current_topics)))
                )

                current_overlap_idx = [
                    current_topics.index(topic) for topic in overlapping_topics
                ]
                previous_overlap_idx = [
                    previous_topics.index(topic) for topic in overlapping_topics
                ]

                c_tf_idf.tolil()[current_overlap_idx] = (
                    (
                        c_tf_idf[current_overlap_idx]
                        + previous_c_tf_idf[previous_overlap_idx]
                    )
                    / 2.0
                ).tolil()

            # Fine-tune the timestamp c-TF-IDF representation based on the global c-TF-IDF representation
            if self.global_tuning:
                selected_topics = [
                    all_topics_indices[topic]
                    for topic in documents_per_topic.Topic.values
                ]
                c_tf_idf = (global_c_tf_idf[selected_topics] + c_tf_idf) / 2.0

            # Extract the words per topic
            words_per_topic = self.topic_model._extract_words_per_topic(
                words, selection, c_tf_idf, calculate_aspects=False
            )
            topic_frequency = pd.Series(
                documents_per_topic.Timestamps.values, index=documents_per_topic.Topic
            ).to_dict()

            # Fill dataframe with results
            topics_at_timestamp = [
                (
                    topic,
                    ", ".join([words[0] for words in values][:10]),
                    topic_frequency[topic],
                    timestamp,
                )
                for topic, values in words_per_topic.items()
            ]
            topics_over_time.extend(topics_at_timestamp)

            if self.evolution_tuning:
                previous_topics = sorted(list(documents_per_topic.Topic.values))
                previous_c_tf_idf = c_tf_idf.copy()

        topics_over_time_df = pd.DataFrame(
            topics_over_time, columns=["Topic", "Words", "Frequency", "Timestamp"]
        )
        self.final_df = topics_over_time_df.merge(
            pd.concat(document_per_topic_list), on=["Topic", "Timestamp"], how="left"
        )

    def calculate_temporal_coherence(
        self, window_size: int = 2, epsilon: float = 1e-12
    ) -> Tuple[pd.DataFrame, float]:
        """
        Calculate the Temporal Topic Coherence (TTC) scores for topics across different timestamps.

        Parameters:
        - window_size (int): Temporal window size to calculate coherence between topic pairs.
        - epsilon (float): A small value added to probabilities to prevent log of 0 in TTC score.

        Returns:
        - Tuple[pd.DataFrame, float]: DataFrame with coherence scores and average coherence score.
        """
        if window_size < 2:
            raise ValueError("window_size must be 2 or above.")

        if self.final_df is None:
            self._topics_over_time()

        # Preprocess and create a dictionary of unique words
        unique_words = set()
        for list_of_words in self.final_df["Words"]:
            unique_words.update([word.strip() for word in list_of_words.split(",")])
        dictionary = list(unique_words)

        # Vectorize documents
        vectorizer = CountVectorizer(vocabulary=dictionary)
        X = vectorizer.fit_transform(self.docs)

        # Calculate word frequencies
        word_freq = dict(
            zip(vectorizer.get_feature_names_out(), np.asarray(X.sum(axis=0)).ravel())
        )

        # Build co-occurrence matrix
        if self.co_occurrence_matrix is None:
            self.co_occurrence_matrix = lil_matrix(
                (len(dictionary), len(dictionary)), dtype=np.int32
            )
            for doc in tqdm(self.docs, desc="Building word co-occurrence matrix"):
                transformed_doc = vectorizer.transform([doc])
                indices = transformed_doc.nonzero()[1]
                for i, j in itertools.combinations(indices, 2):
                    self.co_occurrence_matrix[i, j] += 1
                    self.co_occurrence_matrix[j, i] += 1

            # Fill diagonal with word frequencies
            for word, freq in word_freq.items():
                index = vectorizer.vocabulary_.get(word)
                if index is not None:
                    self.co_occurrence_matrix[index, index] = freq

            self.co_occurrence_matrix = self.co_occurrence_matrix.tocsr()

        # Initialize DataFrame for TTC scores
        ttc_scores_df = pd.DataFrame(
            columns=[
                "Topic ID",
                "Start Timestamp",
                "End Timestamp",
                "Start Topic",
                "End Topic",
                "TTC Score",
            ]
        )

        # Group by topic and iterate
        grouped_topics = self.final_df.groupby("Topic")
        rows_list = []  # For efficient DataFrame construction
        for topic_id, topic_group in grouped_topics:
            sorted_topic_group = topic_group.sort_values("Timestamp")
            for i in range(len(sorted_topic_group) - window_size + 1):
                t1_row = sorted_topic_group.iloc[i]
                t2_row = sorted_topic_group.iloc[i + window_size - 1]

                words_t1 = t1_row["Words"].split(", ")
                words_t2 = t2_row["Words"].split(", ")

                npmi = self._calculate_npmi(
                    words_t1,
                    words_t2,
                    word_freq,
                    self.co_occurrence_matrix,
                    vectorizer,
                    epsilon,
                    len(self.docs),
                )

                # Prepare row for DataFrame
                rows_list.append(
                    {
                        "Topic ID": topic_id,
                        "Start Timestamp": t1_row["Timestamp"],
                        "End Timestamp": t2_row["Timestamp"],
                        "Start Topic": words_t1,
                        "End Topic": words_t2,
                        "TTC Score": npmi,
                    }
                )

        self.ttc_scores_df = pd.DataFrame(rows_list)
        self.average_ttc = self.ttc_scores_df["TTC Score"].mean()
        return self.ttc_scores_df, self.average_ttc

    def _calculate_npmi(
        self,
        words_t1: List[str],
        words_t2: List[str],
        word_freq: Dict[str, int],
        co_occurrence_matrix: lil_matrix,
        vectorizer: CountVectorizer,
        epsilon: float,
        total_docs: int,
    ) -> float:
        """
        Calculate Normalized Pointwise Mutual Information (NPMI) between two sets of words.

        Parameters:
        - words_t1 (List[str]): First set of words.
        - words_t2 (List[str]): Second set of words.
        - word_freq (Dict[str, int]): Dictionary of word frequencies.
        - co_occurrence_matrix (lil_matrix): Co-occurrence matrix of words.
        - vectorizer (CountVectorizer): Vectorizer used for word tokenization.
        - epsilon (float): Small value to prevent division by zero.
        - total_docs (int): Total number of documents.

        Returns:
        - float: NPMI score.
        """
        npmi = 0
        for word_i in words_t1:
            for word_j in words_t2:
                pi = word_freq.get(word_i, 0) / total_docs
                pj = word_freq.get(word_j, 0) / total_docs
                pij = (
                    co_occurrence_matrix[
                        vectorizer.vocabulary_.get(word_i),
                        vectorizer.vocabulary_.get(word_j),
                    ]
                    / total_docs
                )
                npmi_value = (np.log(pij + epsilon) - np.log(pi * pj)) / (
                    -np.log(pij + epsilon)
                )
                npmi += npmi_value
        npmi /= len(words_t1) * len(words_t2)
        return npmi

    def calculate_temporal_smoothness(
        self, window_size: int = 2
    ) -> Tuple[pd.DataFrame, float]:
        """
        Calculate the Temporal Topic Smoothness (TTS) for topics within a dataset across different timestamps.

        Parameters:
        - window_size (int): Temporal window size to calculate smoothness.

        Returns:
        - Tuple[pd.DataFrame, float]: DataFrame with smoothness scores and average smoothness score.
        """
        if window_size < 2:
            raise ValueError("window_size must be 2 or above.")

        if self.final_df is None:
            self._topics_over_time()

        # Initialize a DataFrame to hold the TTS scores for each topic and timestamp
        tts_scores_df = pd.DataFrame(
            columns=[
                "Topic ID",
                "Start Timestamp",
                "End Timestamp",
                "Start Topic",
                "TTS Score",
            ]
        )

        grouped_topics = self.final_df.groupby("Topic")

        # Iterate over each group of topics
        for topic_id, topic_group in tqdm(grouped_topics):
            sorted_topic_group = topic_group.sort_values("Timestamp")

            # Iterate through the sorted_topic_group by pairs of consecutive rows
            for i in range(len(sorted_topic_group) - window_size + 1):
                t1_row = sorted_topic_group.iloc[i]
                words_t1 = t1_row["Words"].split(", ")

                # Initialize sum for redundancy
                redundancy_sum = 0

                # Retrieve all topics from i + 1 to i + window_size - 1 and store their representations (Words column) in c_tild
                c_tild = (
                    sorted_topic_group.iloc[i + 1 : i + window_size]["Words"]
                    .apply(lambda x: x.split(", "))
                    .tolist()
                )

                # Calculate redundancy sum
                for word in words_t1:
                    for topic_words in c_tild:
                        redundancy_sum += word in topic_words

                # Normalize the sum by dividing by the length of C_tild - 1 to get TTS
                tts = redundancy_sum / ((window_size - 1) * len(words_t1))

                # Calculate end timestamp
                end_timestamp = sorted_topic_group.iloc[i + window_size - 1][
                    "Timestamp"
                ]

                # Store the TTS score in the DataFrame
                tts_scores_df = pd.concat(
                    [
                        tts_scores_df,
                        pd.DataFrame(
                            [
                                {
                                    "Topic ID": topic_id,
                                    "Start Timestamp": t1_row["Timestamp"],
                                    "End Timestamp": end_timestamp,
                                    "Start Topic": words_t1,
                                    "TTS Score": tts,
                                }
                            ]
                        ),
                    ]
                )

        self.tts_scores_df = tts_scores_df
        self.average_tts = self.tts_scores_df["TTS Score"].mean()
        return self.tts_scores_df, self.average_tts

    def calculate_temporal_quality(
        self, window_size: int = 2, epsilon: float = 1e-12
    ) -> Tuple[pd.DataFrame, float]:
        """
        Calculate the Temporal Topic Quality (TTQ) for each topic over time.

        Parameters:
        - window_size (int): The number of timestamps to consider for each sliding window calculation.
        - epsilon (float): A small number to stabilize the calculations.

        Returns:
        - Tuple[pd.DataFrame, float]: DataFrame with TTQ scores and average TTQ score.
        """
        # Ensure that we have the necessary TTC and TTS scores calculated
        self.calculate_temporal_coherence(window_size, epsilon)
        self.calculate_temporal_smoothness(window_size)

        # Initialize the DataFrame to hold the TTQ scores
        ttq_scores_df = pd.DataFrame(columns=["Topic ID", "TTQ Score"])

        # Iterate over each unique topic
        for topic_id in self.final_df["Topic"].unique():
            # Get the TTC and TTS scores for the current topic
            topic_ttc = self.ttc_scores_df[self.ttc_scores_df["Topic ID"] == topic_id]
            topic_tts = self.tts_scores_df[self.tts_scores_df["Topic ID"] == topic_id]

            ttq_scores = []
            count_valid_windows = 0

            # Iterate over all possible start timestamps
            for start_timestamp in self.final_df["Timestamp"].unique():
                # Get TTC and TTS values for the current window, if they exist; otherwise, assume a value of 0
                ttc_value = topic_ttc[topic_ttc["Start Timestamp"] == start_timestamp][
                    "TTC Score"
                ].values
                ttc_value = ttc_value[0] if ttc_value.size > 0 else 0

                tts_value = topic_tts[topic_tts["Start Timestamp"] == start_timestamp][
                    "TTS Score"
                ].values
                tts_value = tts_value[0] if tts_value.size > 0 else 0

                # Only count the window if there was a TTC or TTS score found
                if ttc_value or tts_value:
                    count_valid_windows += 1

                # Calculate the TTQ score for the current window and topic
                ttq_score = ttc_value * tts_value
                ttq_scores.append(ttq_score)

            # Calculate the average TTQ score for the topic, considering only windows with valid data
            avg_ttq_score = (
                sum(ttq_scores) / count_valid_windows if count_valid_windows else 0
            )
            ttq_scores_df = pd.concat(
                [
                    ttq_scores_df,
                    pd.DataFrame(
                        {"Topic ID": [topic_id], "TTQ Score": [avg_ttq_score]}
                    ),
                ]
            )

        self.ttq_scores_df = ttq_scores_df
        self.average_ttq = (
            self.ttq_scores_df["TTQ Score"].mean()
            if not self.ttq_scores_df.empty
            else 0
        )
        return self.ttq_scores_df, self.average_ttq

    def plot_temporal_topic_metrics(
        self, metric: str, darkmode: bool = True, topics_to_show: List[int] = None
    ):
        """
        Plot temporal topic coherence/smoothness with an option to select specific topics.

        Parameters:
        - metric (str): The metric to plot ('coherence' or 'smoothness').
        - darkmode (bool): For the aesthetic of the plot.
        - topics_to_show (List[int], optional): List of topic IDs to display. If None or empty, show all.
        """
        if metric == "coherence":
            metric_column = "TTC Score"
            df = self.ttc_scores_df
        elif metric == "smoothness":
            metric_column = "TTS Score"
            df = self.tts_scores_df

        # Initialize a Plotly figure
        fig = go.Figure(
            layout=go.Layout(template="plotly_dark" if darkmode else "plotly")
        )

        # Determine which topics to plot
        if topics_to_show is None or not topics_to_show:
            topics_to_show = self.final_df["Topic"].unique()

        # Plot each topic ID
        for topic_id in topics_to_show:
            # Filter data for the current topic ID
            topic_data = df[df["Topic ID"] == topic_id]

            # Sort the data by 'Start Timestamp' to ensure chronological order
            topic_data = topic_data.sort_values(by="Start Timestamp")

            # Get topic words, split them, take the first three, and format them
            topic_words_str = self.final_df[self.final_df["Topic"] == topic_id][
                "Words"
            ].iloc[0]
            topic_words_list = topic_words_str.split(", ")[
                :3
            ]  # Split and take first three
            topic_words = "_".join(topic_words_list)

            # Add a scatter plot for the current topic ID with custom label and initially deactivated
            fig.add_trace(
                go.Scatter(
                    x=topic_data["Start Timestamp"],
                    y=topic_data[metric_column],
                    mode="lines+markers",
                    name=f"{topic_id}_{topic_words}",
                    text=topic_words_str,
                    hoverinfo="text+x+y",
                    visible="legendonly",  # Initially deactivated in legend
                )
            )

        if metric == "coherence":
            fig.add_shape(
                type="rect",
                xref="paper",
                yref="y",
                x0=0,
                y0=0,
                x1=1,
                y1=float("inf"),
                fillcolor="green",
                opacity=0.2,
                layer="below",
                line_width=0,
            )

        # Update layout
        fig.update_layout(
            title="Temporal Topic " + metric.capitalize(),
            xaxis_title="Timestamp",
            yaxis_title="Temporal " + metric.capitalize() + " Score",
            legend_title="Topic",
            hovermode="closest",
        )

        # Show the figure
        fig.show()

    def plot_temporal_topic_quality(
        self, darkmode: bool = True, topics_to_show: List[int] = None
    ):
        """
        Plot temporal topic quality as a histogram with a color gradient from blue to red.

        Parameters:
        - darkmode (bool): For the aesthetic of the plot.
        - topics_to_show (List[int], optional): List of topic IDs to display. If None or empty, show all.
        """
        metric_column = "TTQ Score"
        df = self.ttq_scores_df

        # If topics_to_show is None or empty, include all topics; otherwise, filter
        if topics_to_show is None or len(topics_to_show) == 0:
            topics_to_show = df["Topic ID"].unique()
        df = df[df["Topic ID"].isin(topics_to_show)]

        # Order by Topic ID for x-axis display
        df = df.sort_values(by="Topic ID")

        # Normalize TTQ Score for color mapping
        df["ScoreNormalized"] = (df[metric_column] - df[metric_column].min()) / (
            df[metric_column].max() - df[metric_column].min()
        )
        df["Color"] = df["ScoreNormalized"].apply(
            lambda x: px.colors.diverging.RdYlGn[
                int(x * (len(px.colors.diverging.RdYlGn) - 1))
            ]
        )

        # Initialize a Plotly figure with a dark theme
        fig = go.Figure(
            layout=go.Layout(template="plotly_dark" if darkmode else "plotly")
        )

        # Create a histogram (bar chart) for the topic quality scores
        for _, row in df.iterrows():
            topic_id = row["Topic ID"]
            metric_value = row[metric_column]
            words = (
                self.final_df[self.final_df["Topic"] == topic_id]["Words"].values[0]
                if topic_id in self.final_df["Topic"].values
                else "No words"
            )

            fig.add_trace(
                go.Bar(
                    x=[topic_id],
                    y=[metric_value],
                    marker_color=row["Color"],
                    name=f"Topic {topic_id}",
                    hovertext=f"Topic {topic_id} Representation: {words}",
                    hoverinfo="text+y",
                )
            )

        # Update layout for histogram
        fig.update_layout(
            title="Temporal Topic Quality Scores",
            xaxis=dict(
                title="Topic ID",
            ),
            yaxis=dict(title="Temporal Quality Score"),
            showlegend=False,
        )

        # Show the figure
        fig.show()
