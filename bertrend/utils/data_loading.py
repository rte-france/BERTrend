#  Copyright (c) 2024, RTE (https://www.rte-france.com)
#  See AUTHORS.txt
#  SPDX-License-Identifier: MPL-2.0
#  This file is part of BERTrend.
import gzip
import re
from pathlib import Path
from typing import Dict, Tuple

import pandas as pd
from loguru import logger
from transformers import AutoTokenizer

from bertrend import DATA_PATH

# Define column names
TEXT_COLUMN = "text"
TIMESTAMP_COLUMN = "timestamp"
GROUPED_TIMESTAMP_COLUMN = "grouped_timestamp"
URL_COLUMN = "url"
TITLE_COLUMN = "title"
CITATION_COUNT_COL = "citation_count"


def find_compatible_files(path, extensions):
    return [
        (str(f.relative_to(path)), f.suffix[1:])
        for f in path.rglob("*")
        if f.suffix[1:] in extensions
    ]


# @st.cache_data
def load_and_preprocess_data(
    selected_file: Tuple[str, str],
    language: str,
    min_chars: int,
    split_by_paragraph: bool,
) -> pd.DataFrame:
    """
    Load and preprocess data from a selected file.

    Args:
        selected_file (tuple): A tuple containing the selected file name and extension.
        language (str): The language of the text data ('French' or 'English').
        min_chars (int): The minimum number of characters required for a text to be included.
        split_by_paragraph (bool): Whether to split the text data by paragraphs.

    Returns:
        pd.DataFrame: The loaded and preprocessed DataFrame.
    """
    file_name, file_ext = selected_file

    if file_ext == "csv":
        df = pd.read_csv(DATA_PATH / file_name)
    elif file_ext == "parquet":
        df = pd.read_parquet(DATA_PATH / file_name)
    elif file_ext == "json":
        df = pd.read_json(DATA_PATH / file_name)
    elif file_ext == "jsonl":
        df = pd.read_json(DATA_PATH / file_name, lines=True)

    # Convert timestamp column to datetime
    df[TIMESTAMP_COLUMN] = pd.to_datetime(df[TIMESTAMP_COLUMN], errors="coerce")

    # Drop rows with invalid timestamps
    df = df.dropna(subset=[TIMESTAMP_COLUMN])

    df = df.sort_values(by=TIMESTAMP_COLUMN, ascending=True).reset_index(drop=True)
    df["document_id"] = df.index

    if URL_COLUMN in df.columns:
        df["source"] = df[URL_COLUMN].apply(
            lambda x: x.split("/")[2] if pd.notna(x) else None
        )
    else:
        df["source"] = None
        df[URL_COLUMN] = None

    if language == "French":
        df[TEXT_COLUMN] = df[TEXT_COLUMN].apply(preprocess_french_text)

    if split_by_paragraph:
        new_rows = []
        for _, row in df.iterrows():
            # Attempt splitting by \n\n first
            paragraphs = re.split(r"\n\n", row[TEXT_COLUMN])
            if len(paragraphs) == 1:  # If no split occurred, attempt splitting by \n
                paragraphs = re.split(r"\n", row[TEXT_COLUMN])
            for paragraph in paragraphs:
                new_row = row.copy()
                new_row[TEXT_COLUMN] = paragraph
                new_row["source"] = row["source"]
                new_rows.append(new_row)
        df = pd.DataFrame(new_rows)

    if min_chars > 0:
        df = df[df[TEXT_COLUMN].str.len() >= min_chars]

    df = df[df[TEXT_COLUMN].str.strip() != ""].reset_index(drop=True)

    return df


def group_by_days(
    df: pd.DataFrame, day_granularity: int = 1
) -> Dict[pd.Timestamp, pd.DataFrame]:
    """
    Group a DataFrame by a specified number of days.

    Args:
        df (pd.DataFrame): The input DataFrame containing a TIMESTAMP_COLUMN column.
        day_granularity (int): The number of days to group by (default is 1).

    Returns:
        Dict[pd.Timestamp, pd.DataFrame]: A dictionary where each key is the timestamp group and the value is the corresponding DataFrame.
    """
    df[TIMESTAMP_COLUMN] = pd.to_datetime(df[TIMESTAMP_COLUMN])
    grouped = df.groupby(pd.Grouper(key=TIMESTAMP_COLUMN, freq=f"{day_granularity}D"))
    dict_of_dfs = {name: group for name, group in grouped}
    return dict_of_dfs


def load_data(full_data_name: Path) -> pd.DataFrame:
    """
    Load data from a file into a pandas DataFrame.

    Args:
        full_data_name (Path): The path to the data file.

    Returns:
        pd.DataFrame: Loaded data with timestamp column converted to datetime.
    """
    logger.info(f"Loading data from: {full_data_name}")
    df = file_to_pd(str(full_data_name), full_data_name.parent)
    df[TIMESTAMP_COLUMN] = pd.to_datetime(df[TIMESTAMP_COLUMN])
    return df.drop_duplicates(subset=["title"], keep="first")


def file_to_pd(file_name: str, base_dir: Path = None) -> pd.DataFrame:
    """
    Read data in various formats and convert it to a DataFrame.

    Args:
        file_name (str): The name of the file to read.
        base_dir (Path, optional): The base directory of the file.

    Returns:
        pd.DataFrame: The loaded data.
    """
    data_path = base_dir / file_name if base_dir else Path(file_name)
    data_path_str = str(data_path)

    if file_name.endswith(".csv"):
        return pd.read_csv(data_path_str)
    elif file_name.endswith(".jsonl") or file_name.endswith(".jsonlines"):
        return pd.read_json(data_path_str, lines=True)
    elif file_name.endswith(".jsonl.gz") or file_name.endswith(".jsonlines.gz"):
        with gzip.open(data_path_str, "rt") as f_in:
            return pd.read_json(f_in, lines=True)
    elif file_name.endswith(".parquet"):
        return pd.read_parquet(data_path_str)


def clean_dataset(dataset: pd.DataFrame, length_criteria: int) -> pd.DataFrame:
    """
    Clean the dataset by removing short texts.

    Args:
        dataset (pd.DataFrame): The dataset to clean.
        length_criteria (int): The minimum length of text to keep.

    Returns:
        pd.DataFrame: The cleaned dataset.
    """
    cleaned_dataset = dataset.loc[dataset[TEXT_COLUMN].str.len() >= length_criteria]
    return cleaned_dataset


def split_df_by_paragraphs(
    dataset: pd.DataFrame,
    enhanced: bool = False,
    tokenizer: AutoTokenizer = None,
    max_length: int = 512,
    min_length: int = 5,
) -> pd.DataFrame:
    """
    Split texts into multiple paragraphs and return a concatenation of all extracts as a new DataFrame.
    If enhanced is True, it also considers the embedding model's max sequence length.

    Args:
        dataset (pd.DataFrame): The dataset to split.
        enhanced (bool): Whether to use the enhanced splitting method.
        tokenizer (AutoTokenizer): The tokenizer to use for splitting (required if enhanced is True).
        max_length (int): The maximum length of tokens (used if enhanced is True).
        min_length (int): The minimum length of tokens (used if enhanced is True).

    Returns:
        pd.DataFrame: The dataset with texts split into paragraphs.
    """
    if not enhanced:
        df = dataset.copy()
        df[TEXT_COLUMN] = df[TEXT_COLUMN].str.split("\n")
        df = df.explode(TEXT_COLUMN)
        df = df[df[TEXT_COLUMN] != ""]
        return df

    if tokenizer is None:
        raise ValueError("Tokenizer is required for enhanced splitting.")

    df = dataset.copy()
    new_rows = []

    for _, row in df.iterrows():
        doc = row[TEXT_COLUMN]
        timestamp = row[TIMESTAMP_COLUMN]

        paragraphs = re.split(r"\n+", doc)

        for paragraph in paragraphs:
            sentences = re.split(r"(?<=[.!?])\s+", paragraph)

            current_doc = ""
            for sentence in sentences:
                sentence_ids = tokenizer.encode(
                    sentence, padding=False, truncation=False, add_special_tokens=False
                )

                if len(sentence_ids) > max_length:
                    sentence_chunks = [
                        sentence[i : i + max_length]
                        for i in range(0, len(sentence), max_length)
                    ]
                    for chunk in sentence_chunks:
                        new_row = row.copy()
                        new_row[TEXT_COLUMN] = chunk
                        new_row["is_split"] = True
                        new_row["num_tokens"] = len(
                            tokenizer.encode(
                                chunk,
                                padding=False,
                                truncation=False,
                                add_special_tokens=False,
                            )
                        )
                        new_rows.append(new_row)
                else:
                    ids = tokenizer.encode(
                        current_doc + " " + sentence,
                        padding=False,
                        truncation=False,
                        add_special_tokens=False,
                    )
                    num_tokens = len(ids)

                    if num_tokens <= max_length:
                        current_doc += " " + sentence
                    else:
                        if current_doc.strip():
                            new_row = row.copy()
                            new_row[TEXT_COLUMN] = current_doc.strip()
                            new_row["is_split"] = True
                            new_row["num_tokens"] = len(
                                tokenizer.encode(
                                    current_doc.strip(),
                                    padding=False,
                                    truncation=False,
                                    add_special_tokens=False,
                                )
                            )
                            new_rows.append(new_row)
                        current_doc = sentence

            if current_doc.strip():
                new_row = row.copy()
                new_row[TEXT_COLUMN] = current_doc.strip()
                new_row["is_split"] = True
                new_row["num_tokens"] = len(
                    tokenizer.encode(
                        current_doc.strip(),
                        padding=False,
                        truncation=False,
                        add_special_tokens=False,
                    )
                )
                new_rows.append(new_row)

    return pd.DataFrame(new_rows)


def preprocess_french_text(text: str) -> str:
    """
    Preprocess French text by normalizing apostrophes, replacing hyphens and similar characters with spaces,
    removing specific prefixes, removing unwanted punctuations (excluding apostrophes, periods, commas, and specific other punctuation),
    replacing special characters with a space (preserving accented characters, llm_utils Latin extensions, and newlines),
    normalizing superscripts and subscripts,
    splitting words containing capitals in the middle (while avoiding splitting fully capitalized words),
    and replacing multiple spaces with a single space.

    Args:
        text (str): The input French text to preprocess.

    Returns:
        str: The preprocessed French text.
    """
    # Normalize different apostrophe variations to a standard apostrophe
    text = text.replace("’", "'")

    # Replace hyphens and similar characters with spaces
    text = re.sub(r"\b(-|/|;|:)", " ", text)

    # Replace special characters with a space (preserving specified punctuation, accented characters, llm_utils Latin extensions, and newlines)
    text = re.sub(r"[^\w\s\nàâçéèêëîïôûùüÿñæœ.,\'\"\(\)\[\]]", " ", text)

    # Normalize superscripts and subscripts for both numbers and letters
    superscript_map = str.maketrans(
        "⁰¹²³⁴⁵⁶⁷⁸⁹ᵃᵇᶜᵈᵉᶠᵍʰⁱʲᵏˡᵐⁿᵒᵖᵠʳˢᵗᵘᵛʷˣʸᶻ", "0123456789abcdefghijklmnopqrstuvwxyz"
    )
    subscript_map = str.maketrans("₀₁₂₃₄₅₆₇₈₉ₐₑᵢₒᵣᵤᵥₓ", "0123456789aeioruvx")
    text = text.translate(superscript_map)
    text = text.translate(subscript_map)

    # Split words that contain capitals in the middle but avoid splitting fully capitalized words
    text = re.sub(r"(?<=[a-z])(?=[A-Z])", " ", text)

    # Replace multiple spaces with a single space
    text = re.sub(r"[ \t]+", " ", text)

    return text
