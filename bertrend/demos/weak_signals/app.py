#  Copyright (c) 2024, RTE (https://www.rte-france.com)
#  See AUTHORS.txt
#  SPDX-License-Identifier: MPL-2.0
#  This file is part of BERTrend.

import pickle
import shutil

import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from bertopic import BERTopic
from loguru import logger

from bertrend import (
    DATA_PATH,
    MODELS_DIR,
    ZEROSHOT_TOPICS_DATA_DIR,
    SIGNAL_EVOLUTION_DATA_DIR,
    CACHE_PATH,
)
from bertrend.BERTrend import BERTrend
from bertrend.services.embedding_service import EmbeddingService
from bertrend.topic_model import TopicModel
from bertrend.demos.weak_signals.messages import (
    MODEL_MERGING_COMPLETE_MESSAGE,
    NO_CACHE_WARNING,
    CACHE_PURGED_MESSAGE,
    MODELS_RESTORED_MESSAGE,
    EMBEDDINGS_CALCULATED_MESSAGE,
    NO_DATA_WARNING,
    MODEL_TRAINING_COMPLETE_MESSAGE,
    STATE_SAVED_MESSAGE,
    STATE_RESTORED_MESSAGE,
    MODELS_SAVED_MESSAGE,
    NO_MODELS_WARNING,
    NO_GRANULARITY_WARNING,
    NO_DATASET_WARNING,
)
from bertrend.trend_analysis.weak_signals import (
    detect_weak_signals_zeroshot,
    save_signal_evolution_data,
    analyze_signal,
)
from bertrend.utils.data_loading import (
    load_and_preprocess_data,
    group_by_days,
    find_compatible_files,
    TEXT_COLUMN,
)
from bertrend.parameters import *
from session_state_manager import SessionStateManager
from bertrend.trend_analysis.visualizations import (
    plot_num_topics_and_outliers,
    plot_topics_per_timestamp,
    plot_topic_size_evolution,
    create_topic_size_evolution_figure,
    plot_newly_emerged_topics,
    create_sankey_diagram,
    PLOTLY_BUTTON_SAVE_CONFIG,
)


# UI Settings
PAGE_TITLE = "BERTrend - Trend Analysis"
LAYOUT = "wide"


def save_state():
    state_file = CACHE_PATH / STATE_FILE
    embeddings_file = CACHE_PATH / EMBEDDINGS_FILE

    # Save the selected files (list of filenames)
    selected_files = SessionStateManager.get("selected_files", [])

    state = SessionStateManager.get_multiple(
        "selected_files",
        "min_chars",
        "split_by_paragraph",
        "timeframe_slider",
        "language",
        "embedding_model_name",
        "embedding_model",
        "sample_size",
        "min_similarity",
        "zeroshot_min_similarity",
        "embedding_dtype",
        "data_embedded",
    )

    state["selected_files"] = selected_files

    with open(state_file, "wb") as f:
        pickle.dump(state, f)

    np.save(embeddings_file, SessionStateManager.get_embeddings())
    st.success(STATE_SAVED_MESSAGE)


def restore_state():
    state_file = CACHE_PATH / STATE_FILE
    embeddings_file = CACHE_PATH / EMBEDDINGS_FILE

    if state_file.exists() and embeddings_file.exists():
        with open(state_file, "rb") as f:
            state = pickle.load(f)

        # Restore the selected files
        selected_files = state.get("selected_files", [])
        SessionStateManager.set("selected_files", selected_files)

        # Restore other states
        SessionStateManager.set_multiple(**state)
        SessionStateManager.set("embeddings", np.load(embeddings_file))
        st.success(STATE_RESTORED_MESSAGE)

        # Update the multiselect widget with restored selected files
        st.session_state["selected_files"] = selected_files
    else:
        st.warning("No saved state found.")


def restore_models():
    if not MODELS_DIR.exists():
        st.warning(NO_MODELS_WARNING)
        return

    topic_models = {}
    for period_dir in MODELS_DIR.iterdir():
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

    SessionStateManager.set("topic_models", topic_models)

    for file, key in [(DOC_GROUPS_FILE, "doc_groups"), (EMB_GROUPS_FILE, "emb_groups")]:
        file_path = CACHE_PATH / file
        if file_path.exists():
            with open(file_path, "rb") as f:
                SessionStateManager.set(key, pickle.load(f))
        else:
            logger.warning(f"{file} not found.")

    granularity_file = CACHE_PATH / GRANULARITY_FILE
    if granularity_file.exists():
        with open(granularity_file, "rb") as f:
            SessionStateManager.set("granularity_select", pickle.load(f))
    else:
        logger.warning(NO_GRANULARITY_WARNING)

    # Restore the models_trained flag
    models_trained_file = CACHE_PATH / MODELS_TRAINED_FILE
    if models_trained_file.exists():
        with open(models_trained_file, "rb") as f:
            # FIXME! set bertrend first!
            SessionStateManager.set("models_trained", pickle.load(f))
    else:
        logger.warning("Models trained flag not found.")

    hyperparams_file = CACHE_PATH / HYPERPARAMS_FILE
    if hyperparams_file.exists():
        with open(hyperparams_file, "rb") as f:
            SessionStateManager.set_multiple(**pickle.load(f))
    else:
        logger.warning("Hyperparameters file not found.")


def purge_cache():
    if CACHE_PATH.exists():
        shutil.rmtree(CACHE_PATH)
        st.success(CACHE_PURGED_MESSAGE)
    else:
        st.warning(NO_CACHE_WARNING)


def main():
    st.set_page_config(
        page_title=PAGE_TITLE, layout=LAYOUT, initial_sidebar_state="expanded"
    )

    st.title(":part_alternation_mark: " + PAGE_TITLE)

    # Set the main flags
    SessionStateManager.get_or_set("data_embedded", False)
    SessionStateManager.get_or_set("popularity_computed", False)

    # Sidebar
    with st.sidebar:
        st.header("Settings and Controls")

        # State Management
        st.subheader("State Management")

        if st.button("Restore Previous Run", use_container_width=True):
            restore_state()
            restore_models()
            st.success(MODELS_RESTORED_MESSAGE)

        if st.button("Purge Cache", use_container_width=True):
            purge_cache()

        if st.button("Clear session state", use_container_width=True):
            SessionStateManager.clear()

        # BERTopic Hyperparameters
        st.subheader("BERTopic Hyperparameters")
        with st.expander("Embedding Model Settings", expanded=False):
            language = st.selectbox("Select Language", LANGUAGES, key="language")
            embedding_dtype = st.selectbox(
                "Embedding Dtype", EMBEDDING_DTYPES, key="embedding_dtype"
            )

            embedding_models = (
                ENGLISH_EMBEDDING_MODELS
                if language == "English"
                else FRENCH_EMBEDDING_MODELS
            )
            embedding_model_name = st.selectbox(
                "Embedding Model", embedding_models, key="embedding_model_name"
            )

        for expander, params in [
            (
                "UMAP Hyperparameters",
                [
                    (
                        "umap_n_components",
                        "UMAP n_components",
                        DEFAULT_UMAP_N_COMPONENTS,
                        2,
                        100,
                    ),
                    (
                        "umap_n_neighbors",
                        "UMAP n_neighbors",
                        DEFAULT_UMAP_N_NEIGHBORS,
                        2,
                        100,
                    ),
                ],
            ),
            (
                "HDBSCAN Hyperparameters",
                [
                    (
                        "hdbscan_min_cluster_size",
                        "HDBSCAN min_cluster_size",
                        DEFAULT_HDBSCAN_MIN_CLUSTER_SIZE,
                        2,
                        100,
                    ),
                    (
                        "hdbscan_min_samples",
                        "HDBSCAN min_sample",
                        DEFAULT_HDBSCAN_MIN_SAMPLES,
                        1,
                        100,
                    ),
                ],
            ),
            (
                "Vectorizer Hyperparameters",
                [
                    ("top_n_words", "Top N Words", DEFAULT_TOP_N_WORDS, 1, 50),
                    ("min_df", "min_df", DEFAULT_MIN_DF, 1, 50),
                ],
            ),
        ]:
            with st.expander(expander, expanded=False):
                for key, label, default, min_val, max_val in params:
                    st.number_input(
                        label,
                        value=default,
                        min_value=min_val,
                        max_value=max_val,
                        key=key,
                    )

                if expander == "HDBSCAN Hyperparameters":
                    st.selectbox(
                        "Cluster Selection Method",
                        HDBSCAN_CLUSTER_SELECTION_METHODS,
                        key="hdbscan_cluster_selection_method",
                    )
                elif expander == "Vectorizer Hyperparameters":
                    st.selectbox(
                        "N-Gram range",
                        VECTORIZER_NGRAM_RANGES,
                        key="vectorizer_ngram_range",
                    )

        with st.expander("Merging Hyperparameters", expanded=False):
            st.slider(
                "Minimum Similarity for Merging",
                0.0,
                1.0,
                DEFAULT_MIN_SIMILARITY,
                0.01,
                key="min_similarity",
            )

        with st.expander("Zero-shot Parameters", expanded=False):
            st.slider(
                "Zeroshot Minimum Similarity",
                0.0,
                1.0,
                DEFAULT_ZEROSHOT_MIN_SIMILARITY,
                0.01,
                key="zeroshot_min_similarity",
            )

    # Main content
    tab1, tab2, tab3 = st.tabs(["Data Loading", "Model Training", "Results Analysis"])

    with tab1:
        st.header("Data Loading and Preprocessing")

        # Find files in the current directory and subdirectories
        compatible_extensions = ["csv", "parquet", "json", "jsonl"]
        selected_files = st.multiselect(
            "Select one or more datasets",
            find_compatible_files(DATA_PATH, compatible_extensions),
            default=SessionStateManager.get("selected_files", []),
            key="selected_files",
        )

        if not selected_files:
            st.warning(NO_DATASET_WARNING)
            return

        # Display number input and checkbox for preprocessing options
        col1, col2 = st.columns(2)
        with col1:
            min_chars = st.number_input(
                "Minimum Characters",
                value=MIN_CHARS_DEFAULT,
                min_value=0,
                max_value=1000,
                key="min_chars",
            )
        with col2:
            split_by_paragraph = st.checkbox(
                "Split text by paragraphs", value=False, key="split_by_paragraph"
            )

        # Load and preprocess each selected file, then concatenate them
        dfs = []
        for selected_file, ext in selected_files:
            file_path = DATA_PATH / selected_file
            df = load_and_preprocess_data(
                (file_path, ext), language, min_chars, split_by_paragraph
            )
            dfs.append(df)

        if not dfs:
            st.warning(
                "No data available after preprocessing. Please check the selected files and preprocessing options."
            )
        else:
            df = pd.concat(dfs, ignore_index=True)

            # Deduplicate using all columns
            df = df.drop_duplicates()

            # Select timeframe
            min_date, max_date = df["timestamp"].dt.date.agg(["min", "max"])
            start_date, end_date = st.slider(
                "Select Timeframe",
                min_value=min_date,
                max_value=max_date,
                value=(min_date, max_date),
                key="timeframe_slider",
            )

            # Filter and sample the DataFrame
            df_filtered = df[
                (df["timestamp"].dt.date >= start_date)
                & (df["timestamp"].dt.date <= end_date)
            ]
            df_filtered = df_filtered.sort_values(by="timestamp").reset_index(drop=True)

            sample_size = st.number_input(
                "Sample Size",
                value=SAMPLE_SIZE_DEFAULT or len(df_filtered),
                min_value=1,
                max_value=len(df_filtered),
                key="sample_size",
            )
            if sample_size < len(df_filtered):
                df_filtered = df_filtered.sample(n=sample_size, random_state=42)

            df_filtered = df_filtered.sort_values(by="timestamp").reset_index(drop=True)

            SessionStateManager.set("timefiltered_df", df_filtered)
            st.write(
                f"Number of documents in selected timeframe: {len(SessionStateManager.get_dataframe('timefiltered_df'))}"
            )
            st.dataframe(
                SessionStateManager.get_dataframe("timefiltered_df")[
                    [TEXT_COLUMN, "timestamp"]
                ],
                use_container_width=True,
            )

            # Embed documents
            if st.button("Embed Documents"):
                embedding_service = EmbeddingService()

                with st.spinner("Embedding documents..."):
                    embedding_dtype = SessionStateManager.get("embedding_dtype")
                    embedding_model_name = SessionStateManager.get(
                        "embedding_model_name"
                    )

                    texts = SessionStateManager.get_dataframe("timefiltered_df")[
                        TEXT_COLUMN
                    ].tolist()

                    try:
                        embedding_model, embeddings = embedding_service.embed_documents(
                            texts=texts,
                            embedding_model_name=embedding_model_name,
                            embedding_dtype=embedding_dtype,
                        )

                        SessionStateManager.set("embedding_model", embedding_model)
                        SessionStateManager.set("embeddings", embeddings)
                        SessionStateManager.set("data_embedded", True)

                        st.success(EMBEDDINGS_CALCULATED_MESSAGE)
                        save_state()
                    except Exception as e:
                        st.error(
                            f"An error occurred while embedding documents: {str(e)}"
                        )

    with tab2:
        st.header("Model Training")

        # Select granularity
        st.number_input(
            "Select Granularity",
            value=DEFAULT_GRANULARITY,
            min_value=1,
            max_value=30,
            key="granularity_select",
            help="Number of days to split the data by",
        )

        # Show documents per grouped timestamp
        with st.expander("Documents per Timestamp", expanded=True):
            grouped_data = group_by_days(
                SessionStateManager.get_dataframe("timefiltered_df"),
                day_granularity=SessionStateManager.get("granularity_select"),
            )
            non_empty_timestamps = [
                timestamp
                for timestamp, group in grouped_data.items()
                if not group.empty
            ]
            if non_empty_timestamps:
                selected_timestamp = st.select_slider(
                    "Select Timestamp",
                    options=non_empty_timestamps,
                    key="timestamp_slider",
                )
                selected_docs = grouped_data[selected_timestamp]
                st.dataframe(
                    selected_docs[
                        ["timestamp", TEXT_COLUMN, "document_id", "source", "url"]
                    ],
                    use_container_width=True,
                )
            else:
                st.warning(NO_DATA_WARNING)

        if not SessionStateManager.get("data_embedded", False):
            st.warning("Please embed data before proceeding to model training.")
            st.stop()
        else:
            # Zero-shot topic definition
            zeroshot_topic_list = st.text_input(
                "Enter zero-shot topics (separated by /)", value=""
            )
            zeroshot_topic_list = [
                topic.strip()
                for topic in zeroshot_topic_list.split("/")
                if topic.strip()
            ]

            if st.button("Train Models"):
                with st.spinner("Training models..."):
                    # FIXME: called twice (see above)
                    grouped_data = group_by_days(
                        SessionStateManager.get_dataframe("timefiltered_df"),
                        day_granularity=SessionStateManager.get("granularity_select"),
                    )

                    logger.debug(SessionStateManager.get("language"))

                    topic_model = TopicModel(
                        umap_n_components=SessionStateManager.get("umap_n_components"),
                        umap_n_neighbors=SessionStateManager.get("umap_n_neighbors"),
                        hdbscan_min_cluster_size=SessionStateManager.get(
                            "hdbscan_min_cluster_size"
                        ),
                        hdbscan_min_samples=SessionStateManager.get(
                            "hdbscan_min_samples"
                        ),
                        hdbscan_cluster_selection_method=SessionStateManager.get(
                            "hdbscan_cluster_selection_method"
                        ),
                        vectorizer_ngram_range=SessionStateManager.get(
                            "vectorizer_ngram_range"
                        ),
                        min_df=SessionStateManager.get("min_df"),
                        top_n_words=SessionStateManager.get("top_n_words"),
                        language=SessionStateManager.get("language"),
                    )

                    bertrend = BERTrend(
                        topic_model=topic_model,
                        zeroshot_topic_list=zeroshot_topic_list,
                        zeroshot_min_similarity=SessionStateManager.get(
                            "zeroshot_min_similarity"
                        ),
                    )
                    bertrend.train_topic_models(
                        grouped_data=grouped_data,
                        embedding_model=SessionStateManager.get("embedding_model"),
                        embeddings=SessionStateManager.get_embeddings(),
                    )

                    # TODO: A supprimer / adapter - cf save/restore
                    SessionStateManager.set_multiple(
                        doc_groups=bertrend.doc_groups,
                        emb_groups=bertrend.emb_groups,
                    )

                    st.success(MODEL_TRAINING_COMPLETE_MESSAGE)

                    bertrend.save_models()
                    st.success(MODELS_SAVED_MESSAGE)

                    # Store bertrend object
                    SessionStateManager.set("bertrend", bertrend)

            if (
                "bertrend" not in st.session_state
                or not SessionStateManager.get("bertrend")._is_fitted
            ):
                st.stop()
            else:
                if st.button("Merge Models"):
                    with st.spinner("Merging models..."):
                        # TODO: encapsulate into a merging function
                        SessionStateManager.get("bertrend").merge_models(
                            min_similarity=SessionStateManager.get("min_similarity"),
                        )

                        SessionStateManager.get("bertrend").calculate_signal_popularity(
                            granularity=SessionStateManager.get("granularity_select"),
                        )

                        SessionStateManager.set("popularity_computed", True)

                    st.success(MODEL_MERGING_COMPLETE_MESSAGE)

    with tab3:
        st.header("Results Analysis")

        if not SessionStateManager.get("data_embedded", False):
            st.warning(
                "Please embed data and train models before proceeding to analysis."
            )
            st.stop()

        elif not SessionStateManager.get("bertrend")._is_fitted:
            st.warning("Please train models before proceeding to analysis.")
            st.stop()

        else:
            topic_models = SessionStateManager.get("bertrend").topic_models
            st.subheader("Topic Overview")
            plot_num_topics_and_outliers(topic_models)
            plot_topics_per_timestamp(topic_models)

            # Display zeroshot signal trend
            if zeroshot_topic_list:
                st.subheader("Zero-shot Weak Signal Trends")
                weak_signal_trends = detect_weak_signals_zeroshot(
                    topic_models,
                    zeroshot_topic_list,
                    SessionStateManager.get("granularity_select"),
                )
                with st.expander("Zero-shot Weak Signal Trends", expanded=False):
                    fig_trend = go.Figure()
                    for topic, weak_signal_trend in weak_signal_trends.items():
                        timestamps = list(weak_signal_trend.keys())
                        popularity = [
                            weak_signal_trend[timestamp]["Document_Count"]
                            for timestamp in timestamps
                        ]
                        hovertext = [
                            f"Topic: {topic}<br>Timestamp: {timestamp}<br>Popularity: {weak_signal_trend[timestamp]['Document_Count']}<br>Representation: {weak_signal_trend[timestamp]['Representation']}"
                            for timestamp in timestamps
                        ]
                        fig_trend.add_trace(
                            go.Scatter(
                                x=timestamps,
                                y=popularity,
                                mode="lines+markers",
                                name=topic,
                                hovertext=hovertext,
                                hoverinfo="text",
                            )
                        )
                    fig_trend.update_layout(
                        title="Popularity of Zero-Shot Topics",
                        xaxis_title="Timestamp",
                        yaxis_title="Popularity",
                    )
                    st.plotly_chart(
                        fig_trend,
                        config=PLOTLY_BUTTON_SAVE_CONFIG,
                        use_container_width=True,
                    )

                    # Display the dataframe with zeroshot topics information
                    zeroshot_topics_data = [
                        {
                            "Topic": topic,
                            "Timestamp": timestamp,
                            "Representation": data["Representation"],
                            "Representative_Docs": data["Representative_Docs"],
                            "Count": data["Count"],
                            "Document_Count": data["Document_Count"],
                        }
                        for topic, weak_signal_trend in weak_signal_trends.items()
                        for timestamp, data in weak_signal_trend.items()
                    ]
                    zeroshot_topics_df = pd.DataFrame(zeroshot_topics_data)
                    st.dataframe(zeroshot_topics_df, use_container_width=True)

                    # Save the zeroshot topics data to a JSON file
                    json_file_path = ZEROSHOT_TOPICS_DATA_DIR
                    json_file_path.mkdir(parents=True, exist_ok=True)

                    zeroshot_topics_df.to_json(
                        json_file_path / ZEROSHOT_TOPICS_DATA_FILE,
                        orient="records",
                        date_format="iso",
                        indent=4,
                    )
                    st.success(f"Zeroshot topics data saved to {json_file_path}")

            if not SessionStateManager.get("popularity_computed", False):
                st.warning("Please merge models to view additional analyses.")
                st.stop()

            else:
                # Display merged signal trend
                st.subheader("Topic Size Evolution")
                st.dataframe(
                    SessionStateManager.get("bertrend").all_merge_histories_df[
                        [
                            "Timestamp",
                            "Topic1",
                            "Topic2",
                            "Representation1",
                            "Representation2",
                            "Document_Count1",
                            "Document_Count2",
                        ]
                    ]
                )

                with st.expander("Topic Popularity Evolution", expanded=True):
                    window_size = st.number_input(
                        "Retrospective Period (days)",
                        min_value=1,
                        max_value=MAX_WINDOW_SIZE,
                        value=DEFAULT_WINDOW_SIZE,
                        key="window_size",
                    )

                    all_merge_histories_df = SessionStateManager.get(
                        "bertrend"
                    ).all_merge_histories_df
                    min_datetime = (
                        all_merge_histories_df["Timestamp"].min().to_pydatetime()
                    )
                    max_datetime = (
                        all_merge_histories_df["Timestamp"].max().to_pydatetime()
                    )

                    current_date = st.slider(
                        "Current date",
                        min_value=min_datetime,
                        max_value=max_datetime,
                        step=pd.Timedelta(
                            days=SessionStateManager.get("granularity_select")
                        ),
                        format="YYYY-MM-DD",
                        help="""The earliest selectable date corresponds to the earliest timestamp when topics were merged 
                        (with the smallest possible value being the earliest timestamp in the provided data). 
                        The latest selectable date corresponds to the most recent topic merges, which is at most equal 
                        to the latest timestamp in the data minus the provided granularity.""",
                    )

                    plot_topic_size_evolution(
                        create_topic_size_evolution_figure(),
                        window_size,
                        SessionStateManager.get("granularity_select"),
                        current_date,
                        min_datetime,
                        max_datetime,
                    )

                    # Save Signal Evolution Data to investigate later on in a separate notebook
                    start_date, end_date = st.select_slider(
                        "Select date range for saving signal evolution data:",
                        options=pd.date_range(
                            start=min_datetime,
                            end=max_datetime,
                            freq=pd.Timedelta(
                                days=SessionStateManager.get("granularity_select")
                            ),
                        ),
                        value=(min_datetime, max_datetime),
                        format_func=lambda x: x.strftime("%Y-%m-%d"),
                    )

                    if st.button("Save Signal Evolution Data"):
                        try:
                            save_path = save_signal_evolution_data(
                                all_merge_histories_df=all_merge_histories_df,
                                topic_sizes=dict(
                                    SessionStateManager.get("bertrend").topic_sizes
                                ),
                                topic_last_popularity=SessionStateManager.get(
                                    "bertrend"
                                ).topic_last_popularity,
                                topic_last_update=SessionStateManager.get(
                                    "bertrend"
                                ).topic_last_update,
                                window_size=SessionStateManager.get("window_size"),
                                granularity=SessionStateManager.get(
                                    "granularity_select"
                                ),
                                start_timestamp=pd.Timestamp(start_date),
                                end_timestamp=pd.Timestamp(end_date),
                            )
                            st.success(
                                f"Signal evolution data saved successfully at {save_path}"
                            )
                        except Exception as e:
                            st.error(
                                f"Error encountered while saving signal evolution data: {e}"
                            )

                # Analyze signal
                st.subheader("Signal Analysis")
                topic_number = st.number_input(
                    "Enter a topic number to take a closer look:", min_value=0, step=1
                )

                if st.button("Analyze signal"):
                    try:
                        language = SessionStateManager.get("language")
                        with st.expander("Signal Interpretation", expanded=True):
                            with st.spinner("Analyzing signal..."):
                                summary, analysis, formatted_html = analyze_signal(
                                    topic_number,
                                    current_date,
                                    all_merge_histories_df,
                                    SessionStateManager.get("granularity_select"),
                                    language,
                                )

                                # Check if the HTML file was created successfully
                                output_file_path = (
                                    Path(__file__).parent / "signal_llm.html"
                                )
                                if output_file_path.exists():
                                    # Read the HTML file
                                    with open(
                                        output_file_path, "r", encoding="utf-8"
                                    ) as file:
                                        html_content = file.read()
                                    # Display the HTML content
                                    st.html(html_content)
                                else:
                                    st.warning(
                                        "HTML generation failed. Displaying markdown instead."
                                    )
                                    # Fallback to displaying markdown if HTML generation fails
                                    col1, col2 = st.columns(
                                        spec=[0.5, 0.5], gap="medium"
                                    )
                                    with col1:
                                        st.markdown(summary)
                                    with col2:
                                        st.markdown(analysis)

                    except Exception as e:
                        st.error(f"Error while trying to generate signal summary: {e}")

                # Create the Sankey Diagram
                st.subheader("Topic Evolution")
                create_sankey_diagram(
                    SessionStateManager.get("bertrend").all_merge_histories_df
                )

                if SessionStateManager.get("bertrend").all_new_topics_df is not None:
                    st.subheader("Newly Emerged Topics")
                    plot_newly_emerged_topics(
                        SessionStateManager.get("bertrend").all_new_topics_df
                    )

                if st.button("Retrieve Topic Counts"):
                    with st.spinner("Retrieving topic counts..."):
                        # Number of topics per individual topic model
                        individual_model_topic_counts = [
                            (timestamp, model.topic_info_df["Topic"].max() + 1)
                            for timestamp, model in topic_models.items()
                        ]
                        df_individual_models = pd.DataFrame(
                            individual_model_topic_counts,
                            columns=["timestamp", "num_topics"],
                        )

                        # Number of topics per cumulative merged model
                        cumulative_merged_topic_counts = SessionStateManager.get(
                            "merge_df_size_over_time", []
                        )
                        df_cumulative_merged = pd.DataFrame(
                            cumulative_merged_topic_counts,
                            columns=["timestamp", "num_topics"],
                        )

                        # Convert to JSON
                        json_individual_models = df_individual_models.to_json(
                            orient="records", date_format="iso", indent=4
                        )
                        json_cumulative_merged = df_cumulative_merged.to_json(
                            orient="records", date_format="iso", indent=4
                        )

                        # Save individual model topic counts
                        json_file_path = (
                            SIGNAL_EVOLUTION_DATA_DIR
                            / f"retrospective_{window_size}_days"
                        )
                        json_file_path.mkdir(parents=True, exist_ok=True)

                        (
                            json_file_path / INDIVIDUAL_MODEL_TOPIC_COUNTS_FILE
                        ).write_text(json_individual_models)

                        # Save cumulative merged model topic counts
                        (
                            json_file_path / CUMULATIVE_MERGED_TOPIC_COUNTS_FILE
                        ).write_text(json_cumulative_merged)

                        st.success(
                            f"Topic counts for individual and cumulative merged models saved to {json_file_path}"
                        )


if __name__ == "__main__":
    main()
