#  Copyright (c) 2024, RTE (https://www.rte-france.com)
#  See AUTHORS.txt
#  SPDX-License-Identifier: MPL-2.0
#  This file is part of BERTrend.

import pandas as pd
import plotly.graph_objects as go
from bertopic import BERTopic
from pandas import Timestamp
from plotly_resampler import FigureWidgetResampler

# Visualization Settings
SANKEY_NODE_PAD = 15
SANKEY_NODE_THICKNESS = 20
SANKEY_LINE_COLOR = "black"
SANKEY_LINE_WIDTH = 0.5


def plot_num_topics(topic_models: dict[pd.Timestamp, BERTopic]) -> go.Figure:
    """
    Plot the number of topics detected for each model.

    Args:
        topic_models (Dict[pd.Timestamp, BERTopic]): A dictionary of BERTopic models, where the key is the timestamp and the value is the corresponding model.
    """
    num_topics = [len(model.get_topic_info()) for model in topic_models.values()]
    fig_num_topics = go.Figure(data=[go.Bar(x=list(topic_models.keys()), y=num_topics)])
    fig_num_topics.update_layout(
        title="Number of Topics Detected",
        xaxis_title="Time Period",
        yaxis_title="Number of Topics",
    )
    return fig_num_topics


def plot_size_outliers(topic_models: dict[pd.Timestamp, BERTopic]) -> go.Figure:
    """
    Plot the size of the outlier topic for each model.

    Args:
        topic_models (Dict[pd.Timestamp, BERTopic]): A dictionary of BERTopic models, where the key is the timestamp and the value is the corresponding model.
    """
    outlier_sizes = [
        (
            model.get_topic_info()
            .loc[model.get_topic_info()["Topic"] == -1]["Count"]
            .values[0]
            if -1 in model.get_topic_info()["Topic"].values
            else 0
        )
        for model in topic_models.values()
    ]
    fig_outlier_sizes = go.Figure(
        data=[go.Bar(x=list(topic_models.keys()), y=outlier_sizes)]
    )
    fig_outlier_sizes.update_layout(
        title="Size of Outlier Topic",
        xaxis_title="Time Period",
        yaxis_title="Size of Outlier Topic",
    )
    return fig_outlier_sizes


def _prepare_source_topic_data(doc_info_df: pd.DataFrame) -> pd.DataFrame:
    """
    Prepare data for plotting topics per source by counting the number of unique documents for each source and topic combination.

    Args:
        doc_info_df (pd.DataFrame): The document information DataFrame containing 'source', 'Topic', 'document_id', and 'Representation' columns.

    Returns:
        pd.DataFrame: A DataFrame with 'source', 'Topic', 'Count', and 'Representation' columns.
    """
    source_topic_counts = (
        doc_info_df.groupby(["source", "Topic"])["document_id"]
        .nunique()
        .reset_index(name="Count")
    )
    topic_representations = (
        doc_info_df.groupby("Topic")["Representation"].first().to_dict()
    )
    source_topic_counts["Representation"] = source_topic_counts["Topic"].map(
        topic_representations
    )
    source_topic_counts = source_topic_counts.sort_values(
        ["source", "Count"], ascending=[True, False]
    )
    return source_topic_counts


def plot_topics_for_model(selected_model: BERTopic) -> go.Figure:
    """
    Plot the topics discussed per source for the selected model.

    Args:
        selected_model (BERTopic): A BERTopic model (selected at a specific timestamp)
    """
    source_topic_counts = _prepare_source_topic_data(selected_model.doc_info_df)

    fig = go.Figure()

    for topic, topic_data in source_topic_counts.groupby("Topic"):
        fig.add_trace(
            go.Bar(
                x=topic_data["source"],
                y=topic_data["Count"],
                name=str(topic)
                + "_"
                + "_".join(topic_data["Representation"].iloc[0][:5]),
                hovertemplate="Source: %{x}<br>Topic: %{customdata}<br>Number of documents: %{y}<extra></extra>",
                customdata=topic_data["Representation"],
            )
        )

    fig.update_layout(
        title="Talked About Topics per Source",
        xaxis_title="Source",
        yaxis_title="Number of Paragraphs",
        barmode="stack",
        legend_title="Topics",
    )

    return fig


def create_topic_size_evolution_figure(topic_sizes: dict, topic_ids=None) -> go.Figure:
    fig = go.Figure()

    if topic_ids is None:
        # If topic_ids is not provided, include all topics
        sorted_topics = sorted(topic_sizes.items(), key=lambda x: x[0])
    else:
        # If topic_ids is provided, filter the topics based on the specified IDs
        sorted_topics = [
            (topic_id, topic_sizes[topic_id])
            for topic_id in topic_ids
            if topic_id in topic_sizes
        ]

    # Create traces for each selected topic
    for topic, data in sorted_topics:
        fig.add_trace(
            go.Scatter(
                x=data["Timestamps"],
                y=data["Popularity"],
                mode="lines+markers",
                name=f"Topic {topic} : {data['Representations'][0].split('_')[:3]}",
                hovertemplate="Topic: %{text}<br>Timestamp: %{x}<br>Popularity: %{y}<br>Representation: %{customdata}<extra></extra>",
                text=[f"Topic {topic}"] * len(data["Timestamps"]),
                customdata=[rep for rep in data["Representations"]],
                line_shape="spline",
            )
        )

    fig.update_layout(
        title="Signal Evolution",
        xaxis_title="Timestamp",
        yaxis_title="Popularity",
        hovermode="closest",
    )
    return fig


def plot_topic_size_evolution(
    fig: go.Figure,
    current_date,
    window_start: Timestamp,
    window_end: Timestamp,
    all_popularity_values: list[float],
    q1: float,
    q3: float,
):
    """
    Plot the evolution of topic sizes over time with colored overlays for signal regions.

    Args:
        fig (FigureWidgetResampler): The cached figure to plot.
        current_date (datetime): The current date selected by the user.
        window_start (Timestamp): The start of the retrospective window size in days.
        window_end (Timestamp): The end of the retrospective window size in days.
        all_popularity_values (List[float]): The list of popularity values.
        q1 (float): the 10th percentile of popularity values,
        q3 (float): the 90th percentile of popularity values,
    """

    # Add colored overlays for signal regions
    y_max = max(all_popularity_values) if all_popularity_values else 1

    # Update the figure layout
    fig.update_layout(
        title="Popularity Evolution",
        xaxis_title="Timestamp",
        yaxis_title="Popularity",
        hovermode="closest",
        xaxis_range=[window_start, window_end],
        yaxis_range=[0, y_max],
        xaxis=dict(type="date", tickformat="%Y-%m-%d"),
    )

    fig.update_yaxes(type="log")

    # Add vertical line for current date
    fig.add_shape(
        type="line",
        x0=current_date,
        x1=current_date,
        y0=0,
        y1=1,
        yref="paper",
        line=dict(color="red", width=2, dash="dash"),
    )

    fig.add_annotation(
        x=current_date,
        y=1,
        yref="paper",
        text="Current Date",
        showarrow=True,
        textangle=0,
        xanchor="left",
        yanchor="bottom",
        bgcolor="rgba(255, 255, 255, 0.8)",
    )

    # Noise region (grey)
    fig.add_hrect(
        y0=-100 * y_max, y1=q1, fillcolor="rgba(128, 128, 128, 0.2)", line_width=0
    )

    # Weak signal region (orange)
    fig.add_hrect(y0=q1, y1=q3, fillcolor="rgba(255, 165, 0, 0.2)", line_width=0)

    # Strong signal region (green)
    fig.add_hrect(y0=q3, y1=y_max * 100, fillcolor="rgba(0, 255, 0, 0.2)", line_width=0)

    return fig


def plot_newly_emerged_topics(all_new_topics_df: pd.DataFrame) -> go.Figure:
    """
    Plot the newly emerged topics over time.

    Args:
        all_new_topics_df (pd.DataFrame): The DataFrame containing information about newly emerged topics.
    """
    fig_new_topics = go.Figure()

    for timestamp, topics_df in all_new_topics_df.groupby("Timestamp"):
        fig_new_topics.add_trace(
            go.Scatter(
                x=[timestamp] * len(topics_df),
                y=topics_df["Document_Count"],
                text=topics_df["Topic"],
                mode="markers",
                marker=dict(
                    size=topics_df["Document_Count"],
                    sizemode="area",
                    sizeref=2.0 * max(topics_df["Count"]) / (40.0**2),
                    sizemin=4,
                ),
                hovertemplate=(
                    "Timestamp: %{x}<br>"
                    "Topic ID: %{text}<br>"
                    "Count: %{y}<br>"
                    "Representation: %{customdata}<extra></extra>"
                ),
                customdata=topics_df["Representation"],
            )
        )

    fig_new_topics.update_layout(
        title="Newly Emerged Topics",
        xaxis_title="Timestamp",
        yaxis_title="Topic Size",
        showlegend=False,
    )

    return fig_new_topics


def _transform_merge_histories_for_sankey(df: pd.DataFrame) -> pd.DataFrame:
    """
    Transform the merge histories DataFrame to prepare it for creating a Sankey diagram.

    The function performs the following steps:
    1. Creates a 'Timestamp_Index' column that maps each timestamp to an index.
    2. Groups by 'Topic1' and collects the list of timestamp indices where each 'Topic1' value appears.
    3. Initializes variables to store the source, destination, representation, timestamp, and count values.
    4. Initializes dictionaries to store the mapping of (topic1, timestamp_index) to the new destination topic and the merged count.
    5. Groups by 'Timestamp' and processes each row to generate the source and destination topics, update the merged count, and determine the timestamp for the current row.
    6. Creates a new DataFrame with the transformed data, including source, destination, representation, timestamp, and count values.
    7. Converts lists to tuples in the 'Representation' column.
    8. Groups by 'Timestamp', 'Source', and 'Destination', and keeps the row with the smallest 'Count' value for each group.

    Args:
        df (pd.DataFrame): The merge histories DataFrame.

    Returns:
        pd.DataFrame: The transformed DataFrame suitable for creating a Sankey diagram.
    """
    # Create a copy of the original dataframe
    transformed_df = df.copy()

    # Create a column 'Timestamp_Index' that maps each timestamp to an index
    timestamps = transformed_df["Timestamp"].unique()
    timestamp_index_map = {
        timestamp: index for index, timestamp in enumerate(timestamps)
    }
    transformed_df["Timestamp_Index"] = transformed_df["Timestamp"].map(
        timestamp_index_map
    )

    # Group by Topic1 and collect the list of timestamp indices where each Topic1 value appears
    topic1_timestamp_indices = (
        transformed_df.groupby("Topic1")["Timestamp_Index"].apply(list).to_dict()
    )

    # Initialize variables to store the source, destination, representation, timestamp, and count values
    src_values = []
    dest_values = []
    representation_values = []
    timestamp_values = []
    count_values = []

    # Initialize a dictionary to store the mapping of (topic1, timestamp_index) to the new destination topic
    topic1_dest_map = {}

    # Initialize a dictionary to store the mapping of (topic1, timestamp_index) to the merged count
    topic1_count_map = {}

    # Group by Timestamp and process each row
    for timestamp, group in transformed_df.groupby("Timestamp"):
        for _, row in group.iterrows():
            topic1 = row["Topic1"]
            topic2 = row["Topic2"]
            representation1 = row["Representation1"]
            representation2 = row["Representation2"]
            timestamp_index = row["Timestamp_Index"]
            count1 = row["Count1"]
            count2 = row["Count2"]
            doc_count1 = row["Document_Count1"]
            doc_count2 = row["Document_Count2"]

            # Generate the source values for Topic1 and Topic2
            src_topic1 = f"{timestamp_index}_1_{topic1}"
            src_topic2 = f"{timestamp_index}_2_{topic2}"

            # Check if (topic1, timestamp_index) has a destination topic in the topic1_dest_map
            if (topic1, timestamp_index) in topic1_dest_map:
                dest_topic = topic1_dest_map[(topic1, timestamp_index)]

                # Update the merged count for the destination topic
                topic1_count_map[(topic1, timestamp_index)] += doc_count2
                count_merged = topic1_count_map[(topic1, timestamp_index)]
            else:
                # Find the next timestamp index where Topic1 appears
                topic1_future_timestamp_indices = [
                    idx
                    for idx in topic1_timestamp_indices[topic1]
                    if idx > timestamp_index
                ]

                if topic1_future_timestamp_indices:
                    next_timestamp_index = topic1_future_timestamp_indices[0]
                    dest_topic = f"{next_timestamp_index}_1_{topic1}"
                else:
                    # If Topic1 doesn't appear in any future timestamps, create a new destination topic
                    dest_topic = f"{timestamp_index}_1_{topic1}_new"

                # Store the mapping of (topic1, timestamp_index) to the new destination topic
                topic1_dest_map[(topic1, timestamp_index)] = dest_topic

                # Initialize the merged count for the destination topic
                topic1_count_map[(topic1, timestamp_index)] = doc_count1 + doc_count2
                count_merged = topic1_count_map[(topic1, timestamp_index)]

            # Determine the timestamp for the current row
            if "_2_" in src_topic2:
                # If the source contains '_2_', find the next available timestamp
                next_timestamp = (
                    timestamps[timestamp_index + 1]
                    if timestamp_index + 1 < len(timestamps)
                    else timestamp
                )
            else:
                # Otherwise, use the current timestamp
                next_timestamp = timestamp

            # Append the source, destination, representation, timestamp, and count values to the respective lists
            src_values.extend([src_topic1, src_topic2])
            dest_values.extend([dest_topic, dest_topic])
            representation_values.extend([representation1, representation2])
            timestamp_values.extend([timestamp, next_timestamp])
            count_values.extend([doc_count1, count_merged])

    # Create a new dataframe with the source, destination, representation, timestamp, and count values
    transformed_df_new = pd.DataFrame(
        {
            "Source": src_values,
            "Destination": dest_values,
            "Representation": representation_values,
            "Timestamp": timestamp_values,
            "Count": count_values,
        }
    )

    # Convert lists to tuples in the 'Representation' column
    transformed_df_new["Representation"] = transformed_df_new["Representation"].apply(
        lambda x: tuple(x) if isinstance(x, list) else x
    )

    # Group by Timestamp, Source, and Destination, and keep the row with the smallest Count value for each group
    transformed_df_new = transformed_df_new.loc[
        transformed_df_new.groupby(["Timestamp", "Source", "Destination"])[
            "Count"
        ].idxmin()
    ]

    return transformed_df_new


def create_sankey_diagram_plotly(
    all_merge_histories_df: pd.DataFrame, search_term: str = None, max_pairs: int = None
):
    """
    Create a Sankey diagram to visualize the topic merging process.

    Args:
        all_merge_histories_df (pd.DataFrame): The DataFrame containing all merge histories.
        search_term (str): Optional search term to filter topics by keyword.
        max_pairs (int): Maximum number of topic pairs to display.

    Returns:
        go.Figure: The Plotly figure representing the Sankey diagram.
    """

    # Transform the merge histories DataFrame to prepare it for creating a Sankey diagram
    transformed_df = _transform_merge_histories_for_sankey(all_merge_histories_df)

    # Filter the dataframe based on the search term if provided
    if search_term:
        # Perform recursive search to find connected nodes
        def find_connected_nodes(node, connected_nodes):
            if node not in connected_nodes:
                connected_nodes.add(node)
                connected_df = transformed_df[
                    (transformed_df["Source"] == node)
                    | (transformed_df["Destination"] == node)
                ]
                for _, row in connected_df.iterrows():
                    find_connected_nodes(row["Source"], connected_nodes)
                    find_connected_nodes(row["Destination"], connected_nodes)

        # Find nodes that match the search term
        matching_nodes = set(
            transformed_df[
                transformed_df["Representation"].apply(
                    lambda x: search_term.lower() in str(x).lower()
                )
            ]["Source"]
        )

        # Find connected nodes
        connected_nodes = set()
        for node in matching_nodes:
            find_connected_nodes(node, connected_nodes)

        # Filter the dataframe based on connected nodes
        transformed_df = transformed_df[
            (transformed_df["Source"].isin(connected_nodes))
            | (transformed_df["Destination"].isin(connected_nodes))
        ]

    # Create nodes and links for the Sankey Diagram
    nodes = []
    links = []
    colors = [
        "#1f77b4",
        "#ff7f0e",
        "#2ca02c",
        "#d62728",
        "#9467bd",
        "#8c564b",
        "#e377c2",
        "#7f7f7f",
        "#bcbd22",
        "#17becf",
    ]

    for _, row in transformed_df.iterrows():
        source_node = row["Source"]
        target_node = row["Destination"]
        timestamp = row["Timestamp"]
        representation = ", ".join(row["Representation"])  # Convert tuple to string
        count = row["Count"]

        # Extract the topic IDs from the source and destination nodes
        source_topic_id = source_node.split("_")[-1]
        target_topic_id = target_node.split("_")
        if target_topic_id[-1] == "new":
            target_topic_id = target_topic_id[-2]
        else:
            target_topic_id = target_topic_id[-1]

        # Generate label for source node
        source_label = ", ".join(representation.split(", ")[:5])

        # Add source node if not already present
        if source_node not in [node["name"] for node in nodes]:
            nodes.append(
                {
                    "name": source_node,
                    "label": source_label,
                    "color": colors[len(nodes) % len(colors)],
                }
            )

        # Add target node if not already present
        if target_node not in [node["name"] for node in nodes]:
            nodes.append(
                {"name": target_node, "color": colors[len(nodes) % len(colors)]}
            )

        # Add link between source and target nodes
        link = {
            "source": source_node,
            "target": target_node,
            "value": count,
            "timestamp": timestamp,
            "source_topic_id": source_topic_id,
            "target_topic_id": target_topic_id,
            "representation": representation,
        }
        if link not in links:
            links.append(link)

    # Limit the number of pairs displayed based on the max_pairs parameter
    if max_pairs is not None:
        links = links[:max_pairs]

    # Create the Sankey Diagram
    fig = go.Figure(
        data=[
            go.Sankey(
                node=dict(
                    pad=SANKEY_NODE_PAD,
                    thickness=SANKEY_NODE_THICKNESS,
                    line=dict(color=SANKEY_LINE_COLOR, width=SANKEY_LINE_WIDTH),
                    label=[node.get("label", "") for node in nodes],
                    color=[node["color"] for node in nodes],
                ),
                link=dict(
                    source=[
                        nodes.index(
                            next(
                                (
                                    node
                                    for node in nodes
                                    if node["name"] == link["source"]
                                ),
                                None,
                            )
                        )
                        for link in links
                    ],
                    target=[
                        nodes.index(
                            next(
                                (
                                    node
                                    for node in nodes
                                    if node["name"] == link["target"]
                                ),
                                None,
                            )
                        )
                        for link in links
                    ],
                    value=[link["value"] for link in links],
                    customdata=[
                        (
                            link["timestamp"],
                            link["source_topic_id"],
                            link["target_topic_id"],
                            link["representation"],
                            link["value"],
                        )
                        for link in links
                    ],
                    hovertemplate="Timestamp: %{customdata[0]}<br />"
                    + "Source Topic ID: %{customdata[1]}<br />"
                    + "Target Topic ID: %{customdata[2]}<br />"
                    "Representation: %{customdata[3]}<br />"
                    + "Document Covered: %{customdata[4]}<extra></extra>",
                    color=[colors[i % len(colors)] for i in range(len(links))],
                ),
                arrangement="snap",
            )
        ]
    )

    # Update the layout
    fig.update_layout(title_text="Topic Merging Process", font_size=15, height=1500)

    return fig
