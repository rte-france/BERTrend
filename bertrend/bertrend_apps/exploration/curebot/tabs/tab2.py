import streamlit as st

from bertrend.bertrend_apps.exploration.curebot.app_utils import (
    URL_COLUMN,
    display_representative_documents,
    display_source_distribution,
    get_website_name,
)


def show() -> None:
    # Check if a model is trained
    if "topic_model" not in st.session_state:
        st.warning("Veuillez ajouter des données et lancer la détection des sujets.")
    else:
        # Show sidebar with topic list
        show_topic_list()

        # Display selected topic info
        display_topic_info(st.session_state.get("selected_topic_number", 0))


def show_topic_list():
    """
    Show topic list in the sidebar.
    Each topic is a button that, when clicked, sets session_state:
        - "selected_topic_number": topic number, this topic is displayed in results tab.
    """
    with st.sidebar:
        with st.expander("Sujets détectés", expanded=True):
            # Topics list
            for _, topic in st.session_state["topics_info"].iterrows():
                # Display button for each topic
                topic_number = topic["Topic"]
                button_title = topic["llm_description"]
                st.button(
                    str(topic_number + 1) + " - " + button_title,
                    width="stretch",
                    on_click=set_topic_selection,
                    args=(topic_number,),
                )


def set_topic_selection(topic_number: int):
    """Set "selected_topic_number" in the session state."""
    st.session_state["selected_topic_number"] = topic_number


def display_topic_info(topic_number: int):
    """Display "selected_topic_number" associated topic information."""
    # Filter df to get only the selected topic
    selected_topic_df = st.session_state["df"][
        st.session_state["df"]["topics"] == topic_number
    ]

    # Get topic info
    docs_count = len(selected_topic_df)
    key_words = st.session_state["topics_info"].iloc[topic_number]["Representation"]
    llm_description = st.session_state["topics_info"].iloc[topic_number][
        "llm_description"
    ]

    # Display topic info
    st.write(f"# {llm_description}")
    st.write(f"### {docs_count} documents")
    st.markdown(f"### #{' #'.join(key_words)}")

    # Get unique sources
    sources = selected_topic_df[URL_COLUMN].apply(get_website_name).unique()

    # Multi-select for sources
    selected_sources = st.multiselect(
        "Sélectionner les sources à afficher :",
        options=["All"] + sorted(list(sources)),
        default=["All"],
    )

    # Create two columns
    col21, col22 = st.columns([0.3, 0.7])

    with col21:
        # Pass the full representative_df to display_source_distribution
        display_source_distribution(selected_topic_df, selected_sources)

    with col22:
        # Filter the dataframe only for document display
        if "All" not in selected_sources:
            filtered_df = selected_topic_df[
                selected_topic_df[URL_COLUMN]
                .apply(get_website_name)
                .isin(selected_sources)
            ]
        else:
            filtered_df = selected_topic_df
        display_representative_documents(filtered_df)
