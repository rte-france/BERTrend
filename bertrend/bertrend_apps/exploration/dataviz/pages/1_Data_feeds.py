#  Copyright (c) 2024, RTE (https://www.rte-france.com)
#  See AUTHORS.txt
#  SPDX-License-Identifier: MPL-2.0
#  This file is part of Wattelse, a NLP application suite.
import math

import pandas as pd
import pydeck as pdk
import streamlit as st

from bertrend.utils.data_loading import TIMESTAMP_COLUMN
from bertrend.bertrend_apps.exploration.geolocalization.entity_utils import (
    geojson_to_dataframe,
    geolocalize_data,
)
from bertrend.bertrend_apps.exploration.geolocalization import load_nlp

GEODATA_COLUM = "geodata"


def get_coords(geojson_data):
    geo_df = geojson_to_dataframe(geojson_data)
    # selection of overall coordinates for the whole doc
    # TODO: heuristics to be defined, basic example below (doc location represented by the one of its first entity)
    if len(geo_df) > 0:
        return (
            geo_df.iloc[0]["latitude"],
            geo_df.iloc[0]["longitude"],
            geo_df.iloc[0]["label"],
        )
    else:
        return math.nan, math.nan, math.nan


def plot_geojson_on_map(df: pd.DataFrame):
    # Use st.map to plot GeoJSON data
    # Convert the GeoDataFrame to a DataFrame
    df[["latitude", "longitude", "label"]] = (
        df[GEODATA_COLUM].apply(get_coords).apply(pd.Series)
    )
    print(df)
    st.map(df.dropna())
    # st.pydeck_chart(create_pydeck_chart(df.dropna()))


def create_pydeck_chart(geo_dataframe: pd.DataFrame):
    # Create a PyDeck layer with lat,lon  data
    layer = pdk.Layer(
        "ScatterplotLayer",
        data=geo_dataframe,
        get_position="[longitute, latitude]",
        get_color="[255, 0, 0]",
        get_radius=1000,
        pickable=True,
    )

    # Create a PyDeck view
    view_state = pdk.ViewState(
        longitude=geo_dataframe["longitude"].mean(),
        latitude=geo_dataframe["latitude"].mean(),
        zoom=5,
    )

    # Create a PyDeck deck
    deck = pdk.Deck(
        map_style="mapbox://styles/mapbox/light-v9",
        layers=[layer],
        initial_view_state=view_state,
    )

    return deck


@st.cache_resource
def load_nlp_wrapper():
    return load_nlp()


nlp = load_nlp_wrapper()

st.title("Browse data")

# Load selected DataFrame
choose_data(DATA_DIR, ["*.csv", "*.jsonl*"])

df = (
    load_data_wrapper(
        f"{st.session_state['data_folder']}/{st.session_state['data_name']}"
    )
    .sort_values(by=TIMESTAMP_COLUMN, ascending=False)
    .reset_index(drop=True)
).reset_index()

with st.expander("Data content"):
    st.dataframe(df, hide_index=True)

# data overview
data_overview(df)

# Géolocalization of data
if st.button("Geolocalize data"):
    with st.expander("Data geolocalization"):
        with st.spinner("Extraction of locations from text"):
            df[GEODATA_COLUM] = geolocalize_data(nlp, df)
            plot_geojson_on_map(df)
