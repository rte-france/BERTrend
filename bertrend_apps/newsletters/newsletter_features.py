#  Copyright (c) 2024, RTE (https://www.rte-france.com)
#  See AUTHORS.txt
#  SPDX-License-Identifier: MPL-2.0
#  This file is part of BERTrend.
import inspect
from pathlib import Path
import os
import locale
from typing import List, Tuple, Any

# from md2pdf.core import md2pdf
import markdown
import pandas as pd
import tldextract
from loguru import logger

from bertrend import LLM_CONFIG
from bertrend.llm_utils.openai_client import OpenAI_Client
from bertrend_apps.newsletters.prompts import (
    FR_USER_SUMMARY_MULTIPLE_DOCS,
    EN_USER_SUMMARY_MULTIPLE_DOCS,
    FR_USER_GENERATE_TOPIC_LABEL_SUMMARIES_V2,
    EN_USER_GENERATE_TOPIC_LABEL_SUMMARIES_V2,
)
from bertrend.services.summarizer import Summarizer
from bertrend.services.summary.abstractive_summarizer import AbstractiveSummarizer
from bertopic._bertopic import BERTopic
from tqdm import tqdm

# Ensures to write with +rw for both user and groups
os.umask(0o002)


def generate_newsletter(
    topic_model: BERTopic,
    df: pd.DataFrame,
    topics: List[int],
    df_split: pd.DataFrame = None,
    top_n_topics: int = 5,
    top_n_docs: int = 3,
    top_n_docs_mode: str = "cluster_probability",
    newsletter_title: str = "Newsletter",
    summarizer_class: Summarizer = AbstractiveSummarizer,
    summary_mode: str = "document",
    prompt_language: str = "fr",
    improve_topic_description: bool = False,
    openai_model_name: str = None,
    nb_sentences: int = 3,
) -> Tuple[str, Any, Any]:
    """Generates a newsletters based on a trained BERTopic model.

    Args:
        topic_model (BERTopic): trained BERTopic model
        df (pd.DataFrame): DataFrame containing documents
        topics (List[int]): list of length len(df) containing the topic number for every document
        df_split (pd.DataFrame, optional): DataFrame containing split documents
        top_n_topics (int, optional): Number of topics to use for newsletters
        top_n_docs (int, optional): Number of document to use to summarize each topic
        top_n_docs_mode (str, optional): algorithm used to recover top n documents (see `get_most_representative_docs` function)
        newsletter_title (str, optional): newsletters title
        summarizer_class (Summarizer, optional): type of summarizer to use (see `bertrend/summary`)
        summary_mode (str, optional): - `document` : for each topic, summarize top n documents independently
                                      - `topic`   : for each topic, use top n documents to generate a single topic summary
                                                    using OpenAI API
                                      - `none`    : do not perform any summarization
        prompt_language (str, optional): prompt language
        improve_topic_description (bool, optional): whether to use ChatGPT to transform topic keywords to a more readable description
        openai_model_name (str, optional): OpenAI model called using OpenAI_API, used to improve topic description and when summary_mode=topic
        nb_sentences (int, optional): Number of sentences used for topic description

    Returns:
        str: Newsletter in Markdown format
    """
    logger.debug("Generating newsletters...")
    openai_api = OpenAI_Client(
        api_key=LLM_CONFIG["api_key"],
        endpoint=LLM_CONFIG["endpoint"],
        model=LLM_CONFIG["model"],
    )

    # Adapt language for date
    current_local = locale.getlocale()
    if prompt_language == "en":
        locale.setlocale(locale.LC_TIME, "en_US.UTF-8")
    elif prompt_language == "fr":
        locale.setlocale(locale.LC_TIME, "fr_FR.UTF-8")

    # Instantiates summarizer
    summarizer = summarizer_class()

    # Ensure top_n_topics is smaller than number of topics
    topics_info = topic_model.get_topic_info()[1:]

    if top_n_topics is None or top_n_topics > len(topics_info):
        top_n_topics = len(topics_info)

    # Date range
    date_min = df.timestamp.min().strftime("%A %d %b %Y")
    date_max = df.timestamp.max().strftime("%A %d %b %Y")

    # Store each line in a list
    md_lines = [f"# {newsletter_title}"]
    if prompt_language == "fr":
        md_lines.append(f"<div class='date_range'>du {date_min} au {date_max}</div>")
    else:
        md_lines.append(f"<div class='date_range'>from {date_min} to {date_max}</div>")

    # Iterate over topics
    for i in tqdm(range(top_n_topics), desc="Processing topics..."):
        sub_df = get_most_representative_docs(
            topic_model,
            df,
            topics,
            mode=top_n_docs_mode,
            df_split=df_split,
            topic_number=i,
            top_n_docs=top_n_docs,
        )

        # Compute summary according to summary_mode
        if summary_mode == "document":
            # Generates summaries for articles
            texts = [doc.text for _, doc in sub_df.iterrows()]
            summaries = summarizer.summarize_batch(
                texts, prompt_language=prompt_language
            )
        elif summary_mode == "topic":
            article_list = ""
            for _, doc in sub_df.iterrows():
                article_list += f"Titre : {doc.title}\nContenu : {doc.text}\n\n"

            topic_summary = openai_api.generate(
                (
                    FR_USER_SUMMARY_MULTIPLE_DOCS
                    if prompt_language == "fr"
                    else EN_USER_SUMMARY_MULTIPLE_DOCS
                ).format(
                    keywords=", ".join(topics_info["Representation"].iloc[i]),
                    article_list=article_list,
                    nb_sentences=nb_sentences,
                ),
                model=openai_model_name,
            )
        elif summary_mode == "none":
            # No summarization is performed
            pass
        else:
            logger.error(
                f"{summary_mode} is not a valid parameter for argument summary_mode in function generate_newsletter"
            )
            exit()

        # Improve topic description
        if improve_topic_description:
            titles = [doc.title for _, doc in sub_df.iterrows()]

            improved_topic_description_v2 = openai_api.generate(
                (
                    FR_USER_GENERATE_TOPIC_LABEL_SUMMARIES_V2
                    if prompt_language == "fr"
                    else EN_USER_GENERATE_TOPIC_LABEL_SUMMARIES_V2
                ).format(
                    newsletter_title=newsletter_title,
                    title_list=(
                        " ; ".join(summaries)
                        if summary_mode == "document"
                        else topic_summary
                    ),
                ),
                model=openai_model_name,
            ).replace('"', "")

            if improved_topic_description_v2.endswith("."):
                improved_topic_description_v2 = improved_topic_description_v2[:-1]

            md_lines.append(f"## Sujet {i + 1} : {improved_topic_description_v2}")

            md_lines.append(
                f"### {' '.join(['#' + keyword for keyword in topics_info['Representation'].iloc[i]])}"
            )
        else:
            md_lines.append(
                f"## Sujet {i + 1} : {', '.join(topics_info['Representation'].iloc[i])}"
            )

        # Write summaries + documents
        if summary_mode == "topic":
            md_lines.append(topic_summary)
        i = 0
        for _, doc in sub_df.iterrows():
            # Write newsletters
            md_lines.append(f"### [*{doc.title}*]({doc.url})")
            try:
                domain = tldextract.extract(doc.url).domain
            except:
                logger.warning(f"Cannot extract URL for {doc}")
                domain = ""
            md_lines.append(
                f"<div class='timestamp'>{doc.timestamp.strftime('%A %d %b %Y')} | {domain}</div>"
            )
            if summary_mode == "document":
                md_lines.append(summaries[i])
            elif summary_mode == "none":
                md_lines.append(
                    doc.text
                )  # Add the full text when no summarization is performed
            i += 1

    # Write full file
    md_content = "\n\n".join(md_lines)

    # Reset locale
    locale.setlocale(locale.LC_TIME, ".".join(current_local))
    return md_content, date_min, date_max


def export_md_string(newsletter_md: str, path: Path, format="md"):
    path.parent.mkdir(parents=True, exist_ok=True)
    if format == "md":
        with open(path, "w") as f:
            f.write(newsletter_md)
    # elif format == "pdf":
    #    md2pdf(path, md_content=newsletter_md)
    elif format == "html":
        css_style = Path(inspect.getfile(generate_newsletter)).parent / "newsletter.css"
        result = md2html(newsletter_md, css_style)
        with open(path, "w") as f:
            f.write(result)


def md2html(md: str, css_style: Path = None) -> str:
    html_result = markdown.markdown(md)
    if not css_style:
        return html_result
    output = """<!DOCTYPE html>
    <html lang="en">
    <head>
        <meta charset="utf-8">
        <style type="text/css">
    """
    cssin = open(css_style)
    output += cssin.read()
    output += """
        </style>
    </head>
    <body>
    """
    output += html_result
    output += """</body>
    </html>
    """
    return output


def get_most_representative_docs(
    topic_model,
    df,
    topics,
    mode="cluster_probability",
    df_split=None,
    topic_number=0,
    top_n_docs=3,
):
    """
    Return most representative documents for a given topic.

    - If df_split is not None :
        Groups splited docs by title to count how many paragraphs of the initial document belong to the topic.
        Returns docs having the most occurences.

    - If df_split is None:
        Uses mode to determine the method used. Currently support :
            * cluster_probability : computes the probability for each docs to belong to the topic using the clustering model. Returns most likely docs.
            * ctfidf_representation : computes c-TF-IDF representation for each docs and compare it to topic c-TF-IDF vector using cosine similarity. Returns highest similarity scores docs.

    """
    # If df_split is not None :
    if isinstance(df_split, pd.DataFrame):
        # Filter docs belonging to the specific topic
        sub_df = df_split.loc[pd.Series(topics) == topic_number]
        # Most representative docs in a topic are those with the highest number of extracts in this topic
        sub_df = (
            sub_df.groupby(["title"])
            .size()
            .reset_index(name="counts")
            .sort_values("counts", ascending=False)
            .iloc[0:top_n_docs]
        )
        return df[df["title"].isin(sub_df["title"])]

    # If no df_split is None, use mode to determine how to return most representative docs :
    elif mode == "cluster_probability":
        docs_prob = topic_model.get_document_info(df["text"])["Probability"]
        df = df.assign(Probability=docs_prob)
        sub_df = df.loc[pd.Series(topics) == topic_number]
        sub_df = sub_df.sort_values("Probability", ascending=False).iloc[0:top_n_docs]
        return sub_df

    elif mode == "ctfidf_representation":
        # TODO : "get_representative_docs" currently returns maximum 3 docs as implemtented in BERTopic
        # We should modify the function to return more if needed
        docs = topic_model.get_representative_docs(topic=topic_number)
        sub_df = df[df["text"].isin(docs)].iloc[0:top_n_docs]
        return sub_df
