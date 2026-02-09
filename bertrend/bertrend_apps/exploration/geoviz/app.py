#  Copyright (c) 2024-2026, RTE (https://www.rte-france.com)
#  See AUTHORS.txt
#  SPDX-License-Identifier: MPL-2.0
#  This file is part of BERTrend.

import backend  # Importing our isolated logic
import folium
import pandas as pd
import streamlit as st
from dotenv import load_dotenv
from folium.plugins import MarkerCluster
from streamlit_folium import st_folium

# Load environment variables at module import
load_dotenv(override=True)

# --- Page Config ---
st.set_page_config(page_title="GeoNews Agent", layout="wide", page_icon="🌍")

st.title("🌍 GeoNews Mapper")
st.markdown("""
    Extract geographical insights from news articles using AI. 
    This tool detects cities, departments, and regions and maps them.
""")

# --- Sidebar: Configuration ---
with st.sidebar:
    st.header("Settings")

    st.markdown("---")
    st.markdown("**Hierarchy Logic:**")
    st.caption("1. **Ville** (City)")
    st.caption("2. **Département**")
    st.caption("3. **Région**")

# --- Main Input Section ---
tab1, tab2 = st.tabs(["🔗 Paste URLs", "📂 Upload Data"])

input_data = []
input_type = ""

examples = [
    "https://www.francebleu.fr/provence-alpes-cote-d-azur/bouches-du-rhone-13/ligne-tht-en-camargue-malgre-les-risques-pour-les-especes-protegees-rte-maintient-son-projet-5901415",
    "https://maritima.fr/actualites/economie/fos-sur-mer/10613/ligne-tres-haute-tension-vers-fos-letat-veut-que-rte-poursuive-les-travaux",
]

with tab1:
    urls_input = st.text_area(
        "Enter News URLs (one per line)",
        value="\n".join(examples),
        height=150,
        placeholder="https://www.lemonde.fr/...\nhttps://www.lefigaro.fr/...",
    )
    if urls_input:
        input_data = [url.strip() for url in urls_input.split("\n") if url.strip()]
        input_type = "url"

with tab2:
    uploaded_file = st.file_uploader(
        "Upload CSV or Excel (Column must be named 'text' or 'url')",
        type=["csv", "xlsx"],
    )
    if uploaded_file:
        if uploaded_file.name.endswith(".csv"):
            df_upload = pd.read_csv(uploaded_file)
        else:
            df_upload = pd.read_excel(uploaded_file)

        # Detect column
        if "url" in df_upload.columns:
            input_data = df_upload["url"].tolist()
            input_type = "url"
        elif "text" in df_upload.columns:
            input_data = df_upload["text"].tolist()
            input_type = "text"
        else:
            st.error("File must contain a 'url' or 'text' column.")

# --- Execution ---
if st.button("🚀 Extract & Map Locations"):
    if not input_data:
        st.warning("Please provide input URLs or text.")
    else:
        with st.spinner("🕵️‍♂️ Agent is reading articles and extracting locations..."):
            # Call the backend
            result_df = backend.extract_geo_data(input_data, input_type)

            if result_df.empty:
                st.error("No locations found or extraction failed.")
            else:
                st.session_state["geo_data"] = result_df
                st.success(f"Extracted {len(result_df)} locations!")

# --- Visualization Section ---
if "geo_data" in st.session_state:
    df = st.session_state["geo_data"]

    st.divider()

    # 1. Search / Filters
    col_filter1, col_filter2 = st.columns(2)
    with col_filter1:
        # Get unique regions, filter out Nones
        regions = ["All"] + sorted([x for x in df["region"].unique() if x])
        selected_region = st.selectbox("Filter by Region", regions)

    with col_filter2:
        search_query = st.text_input("Search for a specific City or Department")

    # Apply filters
    filtered_df = df.copy()
    if selected_region != "All":
        filtered_df = filtered_df[filtered_df["region"] == selected_region]

    if search_query:
        filtered_df = filtered_df[
            filtered_df["city"].str.contains(search_query, case=False, na=False)
            | filtered_df["department"].str.contains(search_query, case=False, na=False)
        ]

    # 2. Map Layout
    col_map, col_data = st.columns([2, 1])

    with col_map:
        st.subheader("Interactive Map")

        # Initialize Map centered on France
        m = folium.Map(location=[46.603354, 1.888334], zoom_start=6)

        # Marker Cluster Group (Prevents overwhelm)
        marker_cluster = MarkerCluster().add_to(m)

        for idx, row in filtered_df.iterrows():
            # Determine what to show as the title
            if row["specific_location"]:
                display_title = row["specific_location"]
                display_subtitle = f"{row['city']}, {row['country']}"
            else:
                display_title = row["city"]
                display_subtitle = f"{row['department'] or ''} - {row['country']}"

            html = f"""
            <b>{display_title}</b><br>
            <span style='color:grey; font-size:10px'>{display_subtitle}</span><br>
            <hr style='margin:5px 0'>
            <i>"{row["snippet"]}"</i><br>
            <a href='{row["source"]}' target='_blank'>Source</a>
            """

            folium.Marker(
                location=[row["lat"], row["lon"]],
                tooltip=row["city"],
                popup=folium.Popup(html, max_width=300),
                icon=folium.Icon(color="blue", icon="info-sign"),
            ).add_to(marker_cluster)

        st_folium(m, width="100%", height=600)

    with col_data:
        st.subheader("Extracted Data")
        st.dataframe(
            filtered_df[["city", "department", "region", "snippet"]],
            hide_index=True,
            height=600,
        )

    # Export
    csv = filtered_df.to_csv(index=False).encode("utf-8")
    st.download_button(
        "📥 Download Data as CSV",
        data=csv,
        file_name="geo_news_data.csv",
        mime="text/csv",
    )
