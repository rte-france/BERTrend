#  Copyright (c) 2024, RTE (https://www.rte-france.com)
#  See AUTHORS.txt
#  SPDX-License-Identifier: MPL-2.0
#  This file is part of BERTrend.

from typing import List, Tuple
import pandas as pd
from bertopic import BERTopic
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity


def preprocess_model(
    topic_model: BERTopic, docs: List[str], embeddings: np.ndarray
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


def merge_models(
    df1: pd.DataFrame, df2: pd.DataFrame, min_similarity: float, timestamp: pd.Timestamp
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    merged_df = df1.copy()
    merge_history = []
    new_topics = []

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
