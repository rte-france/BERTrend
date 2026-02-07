#  Copyright (c) 2024-2026, RTE (https://www.rte-france.com)
#  See AUTHORS.txt
#  SPDX-License-Identifier: MPL-2.0
#  This file is part of BERTrend.

import json
import os

import pandas as pd
from loguru import logger
from SPARQLWrapper import JSON, SPARQLWrapper
from tqdm import tqdm

TEXT_COLUMN = "text"


def geolocalize_data(nlp, df: pd.DataFrame) -> pd.DataFrame:
    """Creates geojson data associated to text entries of the dataframe"""
    text_col = "processed_text" if "processed_text" in df.columns else TEXT_COLUMN
    logger.info("Geolocalizing data...")
    geojson_data_list = []

    for doc in tqdm(nlp.pipe(df[text_col], n_process=os.cpu_count() - 1)):
        # DBPedia resources
        resources = doc._.dbpedia_raw_result.get("Resources")
        features = []
        if resources:
            for r in resources:
                if "DBpedia:Location" in r["@types"].split(","):
                    features += get_geojson_features_from_dbpedia_resource(r["@URI"])
        geojson_data = {"type": "FeatureCollection", "features": features}
        geojson_data_list.append(geojson_data)
        logger.debug(geojson_data)
    return geojson_data_list


def geojson_to_dataframe(geojson_data) -> pd.DataFrame:
    features = geojson_data["features"]

    # Initialize lists to store latitude, longitude, and label data
    latitudes = []
    longitudes = []
    labels = []

    # Extract data from GeoJSON features
    for feature in features:
        geometry = feature["geometry"]
        properties = feature["properties"]

        # Extract latitude and longitude
        latitude, longitude = geometry["coordinates"]
        latitudes.append(latitude)
        longitudes.append(longitude)

        # Extract label (if available)
        label = properties.get("label", None)
        labels.append(label)

    # Create a DataFrame
    df = pd.DataFrame({"latitude": latitudes, "longitude": longitudes, "label": labels})

    return df


def get_geojson_features_from_dbpedia_resource(resource_uri):
    # Define the DBpedia SPARQL endpoint
    sparql_endpoint = "http://fr.dbpedia.org/sparql"

    # Set up SPARQLWrapper
    sparql = SPARQLWrapper(sparql_endpoint)

    # Define the SPARQL query to retrieve GeoJSON data
    sparql_query = f"""
        PREFIX geo: <http://www.w3.org/2003/01/geo/wgs84_pos#>
        SELECT DISTINCT ?subject ?lat ?long ?label
        WHERE {{
            <{resource_uri}> geo:lat ?lat ;
                            geo:long ?long ;
                            rdfs:label ?label .
        }}
    """

    # Set the SPARQL query and request JSON format
    sparql.setQuery(sparql_query)
    sparql.setReturnFormat(JSON)

    # Execute the SPARQL query
    results = sparql.query().convert()

    # Process the results and create GeoJSON
    features = []
    for result in results["results"]["bindings"]:
        feature = {
            "type": "Feature",
            "geometry": {
                "type": "Point",
                "coordinates": [
                    float(result["long"]["value"]),
                    float(result["lat"]["value"]),
                ],
            },
            "properties": {"label": result["label"]["value"]},
        }
        features.append(feature)
        break  # to avoid similar entries (ex. "Berlin" in different languages)

    return features


if __name__ == "__main__":
    # Specify the DBpedia resource URI
    dbpedia_resource_uri = "http://fr.dbpedia.org/resource/Berlin"

    # Get GeoJSON data for the specified resource
    geojson_data = get_geojson_features_from_dbpedia_resource(dbpedia_resource_uri)

    # Print the GeoJSON data
    print(json.dumps(geojson_data, indent=2))
