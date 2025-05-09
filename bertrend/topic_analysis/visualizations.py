#  Copyright (c) 2024, RTE (https://www.rte-france.com)
#  See AUTHORS.txt
#  SPDX-License-Identifier: MPL-2.0
#  This file is part of BERTrend.
import numpy as np
import pandas as pd
import umap
from plotly import express as px, graph_objects as go


def plot_topics_over_time(
    topics_over_time, dynamic_topics_list, topic_model, time_split=None, width=900
):
    if dynamic_topics_list != "":
        if ":" in dynamic_topics_list:
            dynamic_topics_list = [
                i
                for i in range(
                    int(dynamic_topics_list.split(":")[0]),
                    int(dynamic_topics_list.split(":")[1]),
                )
            ]
        else:
            dynamic_topics_list = [int(i) for i in dynamic_topics_list.split(",")]
        fig = topic_model.visualize_topics_over_time(
            topics_over_time,
            topics=dynamic_topics_list,
            width=width,
            title="",
        )
        if time_split:
            fig.add_vline(
                x=time_split,
                line_width=3,
                line_dash="dash",
                line_color="black",
                opacity=1,
            )
        return fig
    return None


def plot_docs_repartition_over_time(df: pd.DataFrame, freq: str):
    """
    Plot document distribution over time
    """
    count = df.groupby(pd.Grouper(key="timestamp", freq=freq), as_index=False).size()
    count["timestamp"] = count["timestamp"].dt.strftime("%Y-%m-%d")

    fig = px.bar(count, x="timestamp", y="size")
    return fig


def plot_remaining_docs_repartition_over_time(
    df_base: pd.DataFrame, df_remaining: pd.DataFrame, freq: str
):
    """
    Plot remaining document distribution over time
    """
    df = pd.concat([df_base, df_remaining])

    # Get split time value
    split_time = str(df_remaining["timestamp"].min())

    # Print aggregated docs
    df["timestamp"] = pd.to_datetime(df["timestamp"])

    count = df.groupby(pd.Grouper(key="timestamp", freq=freq), as_index=False).size()
    count["timestamp"] = count["timestamp"].dt.strftime("%Y-%m-%d")
    # Split to set a different color to each DF
    count["category"] = [
        "Base" if time < split_time else "Remaining" for time in count["timestamp"]
    ]

    fig = px.bar(
        count,
        x="timestamp",
        y="size",
        color="category",
        color_discrete_map={
            "Base": "light blue",  # default plotly color to match main page graphs
            "Remaining": "orange",
        },
    )
    return fig


# TempTopic output visualization functions
def plot_topic_evolution(
    temptopic,
    granularity,
    topics_to_show=None,
    n_neighbors=15,
    min_dist=0.1,
    metric="cosine",
    color_palette="Plotly",
):
    topic_counts = temptopic.final_df.groupby("Topic")["Timestamp"].nunique()
    valid_topics = set(topic_counts[topic_counts > 1].index.tolist())
    all_topics = sorted(set(temptopic.final_df["Topic"].unique()) & set(valid_topics))
    topics_to_include = sorted(set(topics_to_show or all_topics) & set(valid_topics))

    topic_data = {}
    for topic_id in topics_to_include:
        topic_df = temptopic.final_df[temptopic.final_df["Topic"] == topic_id]

        if len(topic_df["Timestamp"].unique()) > 1:
            timestamps = pd.to_datetime(
                topic_df["Timestamp"], format="%Y-%m-%d %H:%M:%S"
            )
            topic_data[topic_id] = {
                "embeddings": topic_df["Embedding"].tolist(),
                "timestamps": timestamps,
                "words": topic_df["Words"].tolist(),
            }

    if not topic_data:
        fig = go.Figure()
        fig.add_annotation(
            text="No topics to display",
            xref="paper",
            yref="paper",
            x=0.5,
            y=0.5,
            showarrow=False,
        )
        return fig

    all_embeddings = np.vstack([data["embeddings"] for data in topic_data.values()])
    reducer = umap.UMAP(
        n_neighbors=n_neighbors,
        n_components=2,
        min_dist=min_dist,
        metric=metric,
        random_state=42,
    )
    all_embeddings_umap = reducer.fit_transform(all_embeddings)

    start_idx = 0
    for topic_id, data in topic_data.items():
        end_idx = start_idx + len(data["embeddings"])
        data["embeddings_umap"] = all_embeddings_umap[start_idx:end_idx]
        start_idx = end_idx

    fig = go.Figure()

    for topic_id, data in topic_data.items():
        topic_words = ", ".join(
            data["words"][0].split(", ")[:3]
        )  # Get first 3 words of the topic
        fig.add_trace(
            go.Scatter3d(
                x=data["embeddings_umap"][:, 0],
                y=data["embeddings_umap"][:, 1],
                z=data["timestamps"],
                mode="lines+markers",
                name=f"Topic {topic_id}: {topic_words}",
                text=[
                    f"Topic: {topic_id}<br>Timestamp: {t}<br>Words: {w}"
                    for t, w in zip(data["timestamps"], data["words"])
                ],
                hoverinfo="text",
                visible="legendonly",
            )
        )

    fig.update_layout(
        scene=dict(xaxis_title="UMAP 1", yaxis_title="UMAP 2", zaxis_title="Timestamp"),
        width=1000,
        height=1000,
    )

    return fig


def plot_temporal_stability_metrics(
    temptopic, metric, darkmode=True, topics_to_show=None
):
    if darkmode:
        fig = go.Figure(layout=go.Layout(template="plotly_dark"))
    else:
        fig = go.Figure()

    topic_counts = temptopic.final_df.groupby("Topic")["Timestamp"].nunique()
    valid_topics = set(topic_counts[topic_counts > 1].index.tolist())
    all_topics = sorted(set(temptopic.final_df["Topic"].unique()) & valid_topics)
    topics_to_include = sorted(set(topics_to_show or all_topics) & valid_topics)

    if metric == "topic_stability":
        df = temptopic.topic_stability_scores_df
        score_column = "Topic Stability Score"
        title = "Temporal Topic Stability"
    elif metric == "representation_stability":
        df = temptopic.representation_stability_scores_df
        score_column = "Representation Stability Score"
        title = "Temporal Representation Stability"
    else:
        raise ValueError(
            "Invalid metric. Choose 'topic_stability' or 'representation_stability'."
        )

    for topic_id in topics_to_include:
        topic_data = df[df["Topic ID"] == topic_id].sort_values(by="Start Timestamp")

        if topic_data.empty:
            continue

        topic_words = temptopic.final_df[temptopic.final_df["Topic"] == topic_id][
            "Words"
        ].iloc[0]
        topic_words = "_".join(topic_words.split(", ")[:3])

        x = topic_data["Start Timestamp"]
        y = topic_data[score_column]

        hover_text = []
        for _, row in topic_data.iterrows():
            if metric == "topic_stability":
                hover_text.append(
                    f"Topic: {topic_id}<br>Timestamp: {row['Start Timestamp']}<br>Score: {row[score_column]:.4f}"
                )
            else:
                hover_text.append(
                    f"Topic: {topic_id}<br>Timestamp: {row['Start Timestamp']}<br>Score: {row[score_column]:.4f}<br>Representation: {row['Start Representation']}"
                )

        fig.add_trace(
            go.Scatter(
                x=x,
                y=y,
                mode="lines+markers",
                name=f"{topic_id}_{topic_words}",
                text=hover_text,
                hoverinfo="text",
                line=dict(shape="spline", smoothing=0.9),
                visible="legendonly",
            )
        )

    if not fig.data:
        fig.add_annotation(
            text="No topics to display",
            xref="paper",
            yref="paper",
            x=0.5,
            y=0.5,
            showarrow=False,
        )

    fig.data = sorted(fig.data, key=lambda trace: int(trace.name.split("_")[0]))
    fig.update_layout(
        title=title,
        xaxis_title="Timestamp",
        yaxis_title=f'{metric.replace("_", " ").capitalize()} Score',
        legend_title="Topic",
        hovermode="closest",
    )

    return fig


def plot_overall_topic_stability(
    temptopic, darkmode=True, normalize=False, topics_to_show=None
):
    if temptopic.overall_stability_df is None:
        temptopic.calculate_overall_topic_stability()

    df = temptopic.overall_stability_df

    if topics_to_show is not None and len(topics_to_show) > 0:
        df = df[df["Topic ID"].isin(topics_to_show)]

    df = df.sort_values(by="Topic ID")

    metric_column = (
        "Normalized Stability Score" if normalize else "Overall Stability Score"
    )
    df["ScoreNormalized"] = df[metric_column]
    df["Color"] = df["ScoreNormalized"].apply(
        lambda x: px.colors.diverging.RdYlGn[
            int(x * (len(px.colors.diverging.RdYlGn) - 1))
        ]
    )

    fig = go.Figure(layout=go.Layout(template="plotly_dark" if darkmode else "plotly"))

    for _, row in df.iterrows():
        topic_id = row["Topic ID"]
        metric_value = row[metric_column]
        words = temptopic.final_df[temptopic.final_df["Topic"] == topic_id][
            "Words"
        ].iloc[0]
        num_timestamps = row["Number of Timestamps"]

        fig.add_trace(
            go.Bar(
                x=[topic_id],
                y=[metric_value],
                marker_color=row["Color"],
                name=f"Topic {topic_id}",
                hovertext=f"Topic {topic_id}<br>Words: {words}<br>Score: {metric_value:.4f}<br>Timestamps: {num_timestamps}",
                hoverinfo="text",
                text=[num_timestamps],
                textposition="outside",
            )
        )

    fig.update_layout(
        title="Overall Topic Stability Scores",
        yaxis=dict(range=[0, 1]),
        yaxis_title="Overall Topic Stability Score",
        showlegend=False,
    )

    return fig
