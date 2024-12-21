#  Copyright (c) 2024, RTE (https://www.rte-france.com)
#  See AUTHORS.txt
#  SPDX-License-Identifier: MPL-2.0
#  This file is part of BERTrend.

import os
import pickle
from pathlib import Path
from typing import Dict, List, Tuple, Any

import numpy as np
import pandas as pd
import scipy
from bertopic import BERTopic
from loguru import logger
from tqdm import tqdm

from bertrend.llm_utils.openai_client import OpenAI_Client
from bertrend.parameters import (
    SIGNAL_CLASSIF_LOWER_BOUND,
    SIGNAL_CLASSIF_UPPER_BOUND,
)
from bertrend import SIGNAL_EVOLUTION_DATA_DIR, LLM_CONFIG
from bertrend.trend_analysis.prompts import get_prompt, save_html_output


def detect_weak_signals_zeroshot(
    topic_models: Dict[pd.Timestamp, BERTopic],
    zeroshot_topic_list: List[str],
    granularity: int,
    decay_factor: float = 0.01,
    decay_power: float = 2,
) -> Dict[str, Dict[pd.Timestamp, Dict[str, any]]]:
    """
    Detect weak signals based on the zero-shot list of topics to monitor.

    Args:
        topic_models (Dict[pd.Timestamp, BERTopic]): Dictionary of BERTopic models for each timestamp.
        zeroshot_topic_list (List[str]): List of topics to monitor for weak signals.
        granularity (int): The granularity of the timestamps in days.
        decay_factor (float): The decay factor for exponential decay.
        decay_power (float): The decay power for exponential decay.

    Returns:
        Dict[str, Dict[pd.Timestamp, Dict[str, any]]]: Dictionary of weak signal trends for each monitored topic.
    """
    weak_signal_trends = {}

    min_timestamp = min(topic_models.keys())
    max_timestamp = max(topic_models.keys())
    timestamps = pd.date_range(
        start=min_timestamp, end=max_timestamp, freq=pd.Timedelta(days=granularity)
    )

    for topic in zeroshot_topic_list:
        weak_signal_trends[topic] = {}
        topic_last_popularity = {}
        topic_last_update = {}

        for timestamp in timestamps:
            # Check if the current timestamp has a corresponding topic model that was trained
            # This is useful in scenarios where we have time skips in the data we were dealing with
            # For example : Our data spans from jan 2024 to dec 2024 with a big gap in summer, and we used
            # a monthly granularity, this means that timestamps of june, july, aug... won't have a corresponding
            # topic model because no model was trained on that period due to non-existant data

            if timestamp in topic_models:
                topic_info = topic_models[timestamp].topic_info_df
                topic_data = topic_info[topic_info["Name"] == topic]

                if not topic_data.empty:
                    # zeroshot topic found in the model's output corresponding to current timestamp
                    representation = topic_data["Representation"].values[0]
                    representative_docs = topic_data["Representative_Docs"].values[0]
                    count = topic_data["Count"].values[0]
                    document_count = topic_data["Document_Count"].values[0]

                    if topic not in topic_last_popularity:
                        # If the First occurrence of the topic
                        topic_last_popularity[topic] = document_count
                        topic_last_update[topic] = timestamp
                    else:
                        # if not first appearance but receives an update in current timestamp, increase popularity
                        document_count += topic_last_popularity[topic]
                        topic_last_popularity[topic] = document_count
                        topic_last_update[topic] = timestamp

                    weak_signal_trends[topic][timestamp] = {
                        "Representation": representation,
                        "Representative_Docs": representative_docs,
                        "Count": count,
                        "Document_Count": document_count,
                    }
                else:
                    # Topic not found in the current timestamp, apply decay to previously seen topics
                    weak_signal_trends[topic][timestamp] = _apply_decay(
                        topic,
                        timestamp,
                        topic_last_popularity,
                        topic_last_update,
                        granularity,
                        decay_factor,
                        decay_power,
                    )
            else:
                # Timestamp not in topic_models, apply decay to previously seen topics
                weak_signal_trends[topic][timestamp] = _apply_decay(
                    topic,
                    timestamp,
                    topic_last_popularity,
                    topic_last_update,
                    granularity,
                    decay_factor,
                    decay_power,
                )

    return weak_signal_trends


def _apply_decay(
    topic,
    timestamp,
    topic_last_popularity,
    topic_last_update,
    granularity,
    decay_factor,
    decay_power,
):
    """Helper function to apply decay to topic popularity."""
    last_popularity = topic_last_popularity.get(topic, 0)
    last_update = topic_last_update.get(topic, timestamp)

    time_diff = timestamp - last_update
    periods_since_last_update = time_diff // pd.Timedelta(days=granularity)

    decayed_popularity = last_popularity * np.exp(
        -decay_factor * (periods_since_last_update**decay_power)
    )

    topic_last_popularity[topic] = decayed_popularity

    return {
        "Representation": None,
        "Representative_Docs": None,
        "Count": 0,
        "Document_Count": decayed_popularity,
    }


def _filter_data(data, window_end, keep_documents):
    """Helper function to filter data based on window_end."""
    return {
        "Timestamps": [ts for ts in data["Timestamps"] if ts <= window_end],
        "Popularity": [
            pop
            for ts, pop in zip(data["Timestamps"], data["Popularity"])
            if ts <= window_end
        ],
        "Representation": [
            rep
            for ts, rep in zip(data["Timestamps"], data["Representations"])
            if ts <= window_end
        ],
        "Documents": (
            [doc for ts, docs in data["Documents"] if ts <= window_end for doc in docs]
            if keep_documents
            else []
        ),
        "Sources": [sources for ts, sources in data["Sources"] if ts <= window_end],
        "Docs_Count": [
            count
            for ts, count in zip(data["Timestamps"], data["Docs_Count"])
            if ts <= window_end
        ],
        "Paragraphs_Count": [
            count
            for ts, count in zip(data["Timestamps"], data["Paragraphs_Count"])
            if ts <= window_end
        ],
        "Source_Diversity": [
            div
            for ts, div in zip(data["Timestamps"], data["Source_Diversity"])
            if ts <= window_end
        ],
    }


def _is_rising_popularity(filtered_data, latest_timestamp):
    """Helper function to check if popularity is rising."""
    retrospective_start = latest_timestamp - pd.Timedelta(days=14)
    retrospective_data = [
        (timestamp, popularity)
        for timestamp, popularity in zip(
            filtered_data["Timestamps"], filtered_data["Popularity"]
        )
        if retrospective_start <= timestamp <= latest_timestamp
    ]

    if len(retrospective_data) >= 2:
        x = range(len(retrospective_data))
        y = [popularity for _, popularity in retrospective_data]
        slope, _, _, _, _ = scipy.stats.linregress(x, y)
        return slope > 0
    return True


def _create_df(topics, keep_documents):
    df = pd.DataFrame(
        [
            {
                "Topic": topic,
                "Representation": filtered_data["Representation"][-1],
                "Latest_Popularity": latest_popularity,
                "Docs_Count": docs_count,
                "Paragraphs_Count": paragraphs_count,
                "Latest_Timestamp": latest_timestamp,
                "Documents": filtered_data["Documents"] if keep_documents else [],
                "Sources": {
                    source for sources in filtered_data["Sources"] for source in sources
                },
                "Source_Diversity": source_diversity,
            }
            for topic, latest_popularity, latest_timestamp, docs_count, paragraphs_count, source_diversity, filtered_data in topics
        ]
    )

    # if not df.empty: df = df[df['Latest_Popularity'] >= 0.01] # Remove signals that faded away by filtering on latest popularity
    return df


def _create_dataframes(
    noise_topics, weak_signal_topics, strong_signal_topics, keep_documents
):
    """Helper function to create DataFrames for each category."""

    return (
        _create_df(noise_topics, keep_documents),
        _create_df(weak_signal_topics, keep_documents),
        _create_df(strong_signal_topics, keep_documents),
    )


def _initialize_new_topic(
    topic_sizes, topic_last_popularity, topic_last_update, topic, timestamp, row
):
    """Initialize a new topic with its first data point."""
    topic_sizes[topic]["Timestamps"] = [timestamp]
    topic_sizes[topic]["Popularity"] = [row["Document_Count1"]]
    topic_sizes[topic]["Representation"] = "_".join(row["Representation1"])
    topic_sizes[topic]["Documents"] = [(timestamp, row["Documents1"])]
    topic_sizes[topic]["Sources"] = [(timestamp, row["Source1"])]
    topic_sizes[topic]["Docs_Count"] = [row["Document_Count1"]]
    topic_sizes[topic]["Paragraphs_Count"] = [row["Count1"]]
    topic_sizes[topic]["Source_Diversity"] = [len(set(row["Source1"]))]
    topic_sizes[topic]["Representations"] = [topic_sizes[topic]["Representation"]]

    topic_last_popularity[topic] = row["Document_Count1"]
    topic_last_update[topic] = timestamp


def update_existing_topic(
    topic_sizes,
    topic_last_popularity,
    topic_last_update,
    topic,
    timestamp,
    granularity,
    row,
):
    """Update an existing topic with new data."""
    next_timestamp = timestamp + granularity

    topic_sizes[topic]["Timestamps"].append(next_timestamp)
    topic_sizes[topic]["Popularity"].append(
        topic_last_popularity[topic] + row["Document_Count2"]
    )
    topic_sizes[topic]["Representation"] = "_".join(row["Representation2"])
    topic_sizes[topic]["Documents"].append((next_timestamp, row["Documents2"]))
    topic_sizes[topic]["Sources"].append((next_timestamp, row["Source2"]))
    topic_sizes[topic]["Docs_Count"].append(
        topic_sizes[topic]["Docs_Count"][-1] + row["Document_Count2"]
    )
    topic_sizes[topic]["Paragraphs_Count"].append(
        topic_sizes[topic]["Paragraphs_Count"][-1] + row["Count2"]
    )

    all_sources = [
        source for _, sources in topic_sizes[topic]["Sources"] for source in sources
    ]
    all_sources.extend(row["Source2"])
    topic_sizes[topic]["Source_Diversity"].append(len(set(all_sources)))
    topic_sizes[topic]["Representations"].append(topic_sizes[topic]["Representation"])

    topic_last_popularity[topic] = topic_last_popularity[topic] + row["Document_Count2"]
    topic_last_update[topic] = next_timestamp  # Update to next_timestamp


def _apply_decay_to_inactive_topics(
    topic_sizes,
    topic_last_popularity,
    topic_last_update,
    updated_topics,
    topics_updated_next,
    current_timestamp,
    granularity,
    decay_factor,
    decay_power,
):
    """Apply decay to topics that were not updated in the current timestamp or the next."""
    all_topics = set(topic_last_update.keys())
    inactive_topics = all_topics - updated_topics - topics_updated_next

    for topic in inactive_topics:
        last_popularity = topic_last_popularity[topic]
        last_update = topic_last_update[topic]

        time_diff = current_timestamp - last_update
        periods_since_last_update = time_diff // granularity

        if periods_since_last_update > 0:
            decayed_popularity = last_popularity * np.exp(
                -decay_factor * (periods_since_last_update**decay_power)
            )

            topic_sizes[topic]["Timestamps"].append(current_timestamp)
            topic_sizes[topic]["Popularity"].append(decayed_popularity)
            topic_sizes[topic]["Docs_Count"].append(
                topic_sizes[topic]["Docs_Count"][-1]
            )
            topic_sizes[topic]["Paragraphs_Count"].append(
                topic_sizes[topic]["Paragraphs_Count"][-1]
            )
            topic_sizes[topic]["Source_Diversity"].append(
                topic_sizes[topic]["Source_Diversity"][-1]
            )
            topic_sizes[topic]["Representations"].append(
                topic_sizes[topic]["Representation"]
            )
            topic_last_popularity[topic] = decayed_popularity


def classify_signals(
    topic_sizes: Dict[int, Dict[str, Any]],
    window_start: pd.Timestamp,
    window_end: pd.Timestamp,
    q1: float,
    q3: float,
    rising_popularity_only: bool = True,
    keep_documents: bool = True,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Classify signals into weak signal and strong signal dataframes.

    Args:
        topic_sizes (Dict[int, Dict[str, Any]]): Dictionary storing topic sizes and related information.
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

    sorted_topics = sorted(topic_sizes.items(), key=lambda x: x[0])

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
                filtered_data["Docs_Count"][-1] if filtered_data["Docs_Count"] else 0
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


def save_signal_evolution_data(
    all_merge_histories_df: pd.DataFrame,
    topic_sizes: Dict[int, Dict[str, Any]],
    topic_last_popularity: Dict[int, float],
    topic_last_update: Dict[int, pd.Timestamp],
    window_size: int,
    granularity: int,
    start_timestamp: pd.Timestamp,
    end_timestamp: pd.Timestamp,
) -> Path:
    window_size_timedelta = pd.Timedelta(days=window_size)
    granularity_timedelta = pd.Timedelta(days=granularity)

    save_path = SIGNAL_EVOLUTION_DATA_DIR / f"retrospective_{window_size}_days"
    os.makedirs(save_path, exist_ok=True)

    q1_values, q3_values, timestamps_over_time = [], [], []
    noise_dfs, weak_signal_dfs, strong_signal_dfs = [], [], []

    for current_timestamp in tqdm(
        pd.date_range(
            start=start_timestamp, end=end_timestamp, freq=granularity_timedelta
        ),
        desc="Processing timestamps",
    ):
        window_end = current_timestamp + granularity_timedelta
        window_start = window_end - granularity_timedelta - window_size_timedelta

        all_popularity_values = [
            popularity
            for topic, data in topic_sizes.items()
            for timestamp, popularity in zip(data["Timestamps"], data["Popularity"])
            if window_start <= timestamp <= current_timestamp and popularity > 1 ^ -5
        ]

        if all_popularity_values:
            q1 = np.percentile(all_popularity_values, SIGNAL_CLASSIF_LOWER_BOUND)
            q3 = np.percentile(all_popularity_values, SIGNAL_CLASSIF_UPPER_BOUND)
        else:
            q1, q3 = 0, 0

        q1_values.append(q1)
        q3_values.append(q3)

        noise_df, weak_signal_df, strong_signal_df = classify_signals(
            topic_sizes, window_start, window_end, q1, q3, keep_documents=False
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
            "granularity": granularity,
            "timestamps": timestamps_over_time,
            "q1_values": q1_values,
            "q3_values": q3_values,
        }
        pickle.dump(metadata, f)

    return save_path


def analyze_signal(
    topic_number, current_date, all_merge_histories_df, granularity, language
):
    topic_merge_rows = all_merge_histories_df[
        all_merge_histories_df["Topic1"] == topic_number
    ].sort_values("Timestamp")
    topic_merge_rows_filtered = topic_merge_rows[
        topic_merge_rows["Timestamp"] <= current_date
    ]

    if not topic_merge_rows_filtered.empty:
        content_summary = "\n".join(
            [
                f"Timestamp: {row.Timestamp.strftime('%Y-%m-%d')}\n"
                f"Topic representation: {row.Representation1}\n"
                f"{' '.join(f'- {doc}' for doc in row.Documents1 if isinstance(doc, str))}\n"
                f"Timestamp: {(row.Timestamp + pd.Timedelta(days=granularity)).strftime('%Y-%m-%d')}\n"
                f"Topic representation: {row.Representation2}\n"
                f"{' '.join(f'- {doc}' for doc in row.Documents2 if isinstance(doc, str))}\n"
                for row in topic_merge_rows_filtered.itertuples()
            ]
        )

        try:
            openai_client = OpenAI_Client(
                api_key=LLM_CONFIG["api_key"],
                endpoint=LLM_CONFIG["endpoint"],
                model=LLM_CONFIG["model"],
            )

            # First prompt: Generate summary
            logger.debug("First prompt - generate summary")
            summary_prompt = get_prompt(
                language,
                "topic_summary",
                topic_number=topic_number,
                content_summary=content_summary,
            )
            summary = openai_client.generate(
                system_prompt=LLM_CONFIG["system_prompt"],
                user_prompt=summary_prompt,
                temperature=LLM_CONFIG["temperature"],
                max_tokens=LLM_CONFIG["max_tokens"],
            )

            # Second prompt: Analyze weak signal
            logger.debug("Second prompt - analyze weak signal")
            weak_signal_prompt = get_prompt(
                language, "weak_signal", summary_from_first_prompt=summary
            )
            weak_signal_analysis = openai_client.generate(
                system_prompt=LLM_CONFIG["system_prompt"],
                user_prompt=weak_signal_prompt,
                temperature=LLM_CONFIG["temperature"],
                max_tokens=LLM_CONFIG["max_tokens"],
            )

            # Third prompt: Generate HTML format
            logger.debug("Third prompt - generate html format")
            html_format_prompt = get_prompt(
                language=language,
                prompt_type="html_format",
                topic_summary=summary,
                weak_signal_analysis=weak_signal_analysis,
            )
            formatted_html = openai_client.generate(
                system_prompt=LLM_CONFIG["system_prompt"],
                user_prompt=html_format_prompt,
                temperature=LLM_CONFIG["temperature"],
                max_tokens=LLM_CONFIG["max_tokens"],
            )

            # Save the formatted HTML
            save_html_output(formatted_html)

            return summary, weak_signal_analysis, formatted_html

        except Exception as e:
            error_msg = f"An error occurred while generating the analysis: {str(e)}"
            logger.error(error_msg)
            raise Exception(error_msg)

    else:
        error_msg = f"No data available for topic {topic_number} within the specified date range. Please enter a valid topic number."
        logger.error(error_msg)
        raise Exception(error_msg)
