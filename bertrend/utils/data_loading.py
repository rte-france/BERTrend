#  Copyright (c) 2024, RTE (https://www.rte-france.com)
#  See AUTHORS.txt
#  SPDX-License-Identifier: MPL-2.0
#  This file is part of BERTrend.
import gzip
import re
from pathlib import Path
from typing import Literal

import pandas as pd
from loguru import logger
from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer

from bertrend import DATA_PATH
from bertrend.config.parameters import MIN_CHARS_DEFAULT

# Define column names
TEXT_COLUMN = "text"
TIMESTAMP_COLUMN = "timestamp"
GROUPED_TIMESTAMP_COLUMN = "grouped_timestamp"
URL_COLUMN = "url"
TITLE_COLUMN = "title"
DOCUMENT_ID_COLUMN = "document_id"
SOURCE_COLUMN = "source"
CITATION_COUNT_COL = "citation_count"


def find_compatible_files(path: Path, extensions: list[str]) -> list[Path]:
    return [f.relative_to(path) for f in path.rglob("*") if f.suffix[1:] in extensions]


def load_data(
    selected_file: Path,
    language: str = "French",
) -> pd.DataFrame | None:
    """
    Load and preprocess data (adds additional columns if needed) from a selected file.

    Args:
        selected_file (Path): A path to the selected file
        language (str): The language of the text data ('French' or 'English').

    Returns:
        pd.DataFrame: The loaded and preprocessed DataFrame.
    """
    df = _file_to_pd(selected_file)
    if df is None:
        return None

    # Convert timestamp column to datetime
    df[TIMESTAMP_COLUMN] = pd.to_datetime(df[TIMESTAMP_COLUMN], errors="coerce")

    # Drop rows with invalid timestamps
    df = df.dropna(subset=[TIMESTAMP_COLUMN])

    df = df.sort_values(by=TIMESTAMP_COLUMN, ascending=True).reset_index(drop=True)
    df[DOCUMENT_ID_COLUMN] = df.index

    if URL_COLUMN in df.columns:
        df[SOURCE_COLUMN] = df[URL_COLUMN].apply(
            lambda x: x.split("/")[2] if pd.notna(x) else None
        )
    else:
        df[SOURCE_COLUMN] = None
        df[URL_COLUMN] = None

    if language == "French":
        df[TEXT_COLUMN] = df[TEXT_COLUMN].apply(preprocess_french_text)

    if TITLE_COLUMN in df.columns:
        df.drop_duplicates(subset=[TITLE_COLUMN], keep="first", inplace=True)
    else:
        # Title colupmn used in topic explorations
        df[TITLE_COLUMN] = ""

    return df


def split_data(
    df: pd.DataFrame,
    min_chars: int = MIN_CHARS_DEFAULT,
    split_by_paragraph: Literal["no", "yes", "enhanced"] = "yes",
    embedding_model_name: str = None,
) -> pd.DataFrame | None:
    """
    Load and preprocess data from a selected file.

    Args:
        df (pd.DataFrame): A dataframe containing loaded documents
        min_chars (int): The minimum number of characters required for a text to be included.
        split_by_paragraph (Literal): Way to split the text data by paragraphs (no, yes, enhanced)
        embedding_model_name (str): Name of the embedding model (if done locally), only useful if split_by_paragraph is 'enhanced'.

    Returns:
        pd.DataFrame: The split DataFrame.
    """
    if split_by_paragraph == "yes":
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
    elif split_by_paragraph == "enhanced":
        # FIXME: requires access to the tokenizer
        df = enhanced_split_df_by_paragraphs(
            dataset=df, embedding_model_name=embedding_model_name
        )

    if min_chars > 0:
        df = df[df[TEXT_COLUMN].str.len() >= min_chars]

    df = df[df[TEXT_COLUMN].str.strip() != ""].reset_index(drop=True)

    return df


def group_by_days(
    df: pd.DataFrame, day_granularity: int = 1
) -> dict[pd.Timestamp, pd.DataFrame]:
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


def _file_to_pd(file_name: Path, base_dir: Path = None) -> pd.DataFrame | None:
    """
    Read data in various formats and convert it to a DataFrame.

    Args:
        file_name (str): The name of the file to read.
        base_dir (Path, optional): The base directory of the file.

    Returns:
        pd.DataFrame: The loaded data.
    """
    file_ext = file_name.suffix.lower()
    if file_ext == ".csv":
        return pd.read_csv(DATA_PATH / file_name)
    elif file_ext == ".parquet":
        return pd.read_parquet(DATA_PATH / file_name)
    elif file_ext == ".json":
        return pd.read_json(DATA_PATH / file_name)
    elif file_ext == ".jsonl" or file_ext == ".jsonlines":
        return pd.read_json(DATA_PATH / file_name, lines=True)
    elif file_ext == ".jsonl.gz" or file_ext == ".jsonlines.gz":
        with gzip.open(file_name, "rt") as f_in:
            return pd.read_json(f_in, lines=True)
    elif file_ext == ".xslsx":
        return pd.read_excel(DATA_PATH / file_name)
    else:
        return None


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


def enhanced_split_df_by_paragraphs(
    dataset: pd.DataFrame, embedding_model_name: str = None
) -> pd.DataFrame:
    """
    Split texts into multiple paragraphs and return a concatenation of all extracts as a new DataFrame.
    If enhanced is True, it also considers the embedding model's max sequence length.

    Args:
        dataset (pd.DataFrame): The dataset to split.
        embedding_model_name (str): The name of the embedding model to use (only required if enhanced is True)

    Returns:
        pd.DataFrame: The dataset with texts split into paragraphs.
    """
    if embedding_model_name is None:
        raise ValueError("Tokenizer is required for enhanced splitting.")

    tokenizer = AutoTokenizer.from_pretrained(embedding_model_name)
    max_length = SentenceTransformer(embedding_model_name).get_max_seq_length()
    # Correcting the max seq length anomaly in certain embedding models description
    if max_length == 514:
        max_length = 512

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
    replacing special characters with a space (preserving accented characters, utils Latin extensions, and newlines),
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

    # Replace special characters with a space (preserving specified punctuation, accented characters, utils Latin extensions, and newlines)
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
