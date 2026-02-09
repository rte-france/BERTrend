from typing import List, Optional

import pandas as pd
from geopy.extra.rate_limiter import RateLimiter
from geopy.geocoders import Nominatim
from goose3 import Goose
from pydantic import BaseModel, Field

from bertrend.llm_utils.openai_client import OpenAI_Client


# --- Data Models ---
class GeoLocation(BaseModel):
    specific_location: Optional[str] = Field(
        None,
        description="A specific landmark, building, district, or street (e.g., 'Tour Eiffel', 'Gare de Lyon', 'Brooklyn Bridge').",
    )
    city: str = Field(..., description="The city or municipality.")
    department: Optional[str] = Field(
        None,
        description="The administrative department, state, or province (e.g., 'Gironde', 'California').",
    )
    region: Optional[str] = Field(
        None,
        description="The larger administrative region (e.g., 'Nouvelle-Aquitaine').",
    )
    country: str = Field(
        ...,
        description="The country where this location is found (e.g., 'France', 'USA').",
    )
    context_snippet: str = Field(
        ...,
        description="A short snippet of text justifying why this location was extracted.",
    )


class ArticleLocations(BaseModel):
    locations: List[GeoLocation]


# --- Helper Functions ---


def fetch_text_from_url(url: str) -> str:
    """
    Uses Goose3 to extract the main content of the article,
    stripping out ads, menus, and html clutter.
    """
    try:
        g = Goose()
        article = g.extract(url=url)
        text = article.cleaned_text
        g.close()

        if not text:
            return ""

        # Truncate if extremely long to save OpenAI tokens
        return text[:15000]

    except Exception as e:
        print(f"Goose extraction error for {url}: {e}")
        return ""


def geocode_locations(df: pd.DataFrame) -> pd.DataFrame:
    """
    Adds Lat/Lon using a hierarchical strategy to ensure the most precise
    location is found while maintaining a fallback for broader searches.
    """
    # Unique user_agent prevents 403 errors
    geolocator = Nominatim(user_agent="geo_news_agent_app_v5")
    geocode = RateLimiter(geolocator.geocode, min_delay_seconds=1)

    lats = []
    lons = []

    print("Geocoding locations...")

    for _, row in df.iterrows():
        location = None

        # 1. Strategy: Specific Landmark Precision
        # Query: "Tour Eiffel, Paris, France"
        if row.get("specific_location"):
            query = f"{row['specific_location']}, {row['city']}, {row['country']}"
            try:
                location = geocode(query)
            except:
                pass

        # 2. Strategy: Administrative Precision
        # Query: "Paris, Ile-de-France, France"
        if not location and row.get("department"):
            query = f"{row['city']}, {row['department']}, {row['country']}"
            try:
                location = geocode(query)
            except:
                pass

        # 3. Strategy: City Fallback
        # Query: "Paris, France"
        if not location:
            query = f"{row['city']}, {row['country']}"
            try:
                location = geocode(query)
            except:
                pass

        if location:
            lats.append(location.latitude)
            lons.append(location.longitude)
        else:
            lats.append(None)
            lons.append(None)

    df["lat"] = lats
    df["lon"] = lons

    # Return only rows where we successfully found coordinates
    return df.dropna(subset=["lat", "lon"])


# --- Main Agent Function ---


def extract_geo_data(inputs: List[str], input_type: str) -> pd.DataFrame:
    """
    Main pipeline:
    1. Gets text (using Goose3 if URL).
    2. Calls OpenAI to extract structured hierarchy including specific landmarks and country.
    3. Aggregates results and geocodes them.
    """
    client = OpenAI_Client()
    all_results = []

    for item in inputs:
        # 1. Prepare Text
        if input_type == "url":
            text_content = fetch_text_from_url(item)
            source_id = item
        else:
            text_content = item
            source_id = "Uploaded Text"

        # Skip empty extractions
        if not text_content or len(text_content) < 50:
            continue

        # 2. Call OpenAI Agent
        try:
            system_prompt = (
                "You are a geographical extraction specialist. Extract all relevant physical locations mentioned in the text. "
                "For each location, provide the following hierarchy:\n"
                "1. **Specific Location**: A landmark, building, square, or district (e.g., 'Louvre Museum', 'La Défense'). If none, leave null.\n"
                "2. **City**: The municipality.\n"
                "3. **Department**: The state, province, or department.\n"
                "4. **Country**: Crucial. Infer it if not explicit (e.g., London -> UK).\n"
                "Provide a context snippet explaining the relevance."
            )

            extracted = client.parse(
                system_prompt=system_prompt,
                user_prompt=text_content,
                response_format=ArticleLocations,
            )

            # 3. Flatten results
            for loc in extracted.locations:
                all_results.append(
                    {
                        "source": source_id,
                        "specific_location": loc.specific_location,
                        "city": loc.city,
                        "department": loc.department,
                        "region": loc.region,
                        "country": loc.country,
                        "snippet": loc.context_snippet,
                    }
                )

        except Exception as e:
            print(f"Error processing {source_id}: {e}")

    df = pd.DataFrame(all_results)

    if not df.empty:
        df = geocode_locations(df)

    return df
