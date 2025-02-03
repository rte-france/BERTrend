#  Copyright (c) 2024, RTE (https://www.rte-france.com)
#  See AUTHORS.txt
#  SPDX-License-Identifier: MPL-2.0
#  This file is part of BERTrend.
import inspect
from pathlib import Path
import os
import locale

# from md2pdf.core import md2pdf
import markdown
import pandas as pd
import tldextract
from loguru import logger

from bertrend import LLM_CONFIG
from bertrend.llm_utils.openai_client import OpenAI_Client
from bertrend.services.summary.chatgpt_summarizer import GPTSummarizer
from bertrend.topic_analysis.representative_docs import get_most_representative_docs
from bertrend.llm_utils.prompts import (
    USER_SUMMARY_MULTIPLE_DOCS,
    USER_GENERATE_TOPIC_LABEL_SUMMARIES,
)
from bertrend.services.summarizer import Summarizer
from bertopic._bertopic import BERTopic
from tqdm import tqdm

# Ensures to write with +rw for both user and groups
os.umask(0o002)

DEFAULT_TOP_N_TOPICS = 5
DEFAULT_TOP_N_DOCS = 3
DEFAULT_TOP_N_DOCS_MODE = "cluster_probability"
DEFAULT_SUMMARY_MODE = "document"


def generate_newsletter(
    topic_model: BERTopic,
    df: pd.DataFrame,
    topics: list[int],
    df_split: pd.DataFrame = None,
    top_n_topics: int = DEFAULT_TOP_N_TOPICS,
    top_n_docs: int = DEFAULT_TOP_N_DOCS,
    top_n_docs_mode: str = DEFAULT_TOP_N_DOCS_MODE,
    newsletter_title: str = "Newsletter",
    summarizer_class: Summarizer = GPTSummarizer,
    summary_mode: str = DEFAULT_SUMMARY_MODE,
    prompt_language: str = "fr",
    improve_topic_description: bool = False,
    openai_model_name: str = None,
    nb_sentences: int = 3,
) -> tuple[str, str, str]:
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
    if not openai_model_name:
        openai_model_name = LLM_CONFIG["model"]
    openai_api = OpenAI_Client(
        api_key=LLM_CONFIG["api_key"],
        endpoint=LLM_CONFIG["endpoint"],
        model=openai_model_name,
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
                (USER_SUMMARY_MULTIPLE_DOCS[prompt_language]).format(
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
                (USER_GENERATE_TOPIC_LABEL_SUMMARIES[prompt_language]).format(
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


def export_md_string(newsletter_md: str, path: Path, output_format="md"):
    """Save a markdown string to a file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    if output_format == "md":
        with open(path, "w") as f:
            f.write(newsletter_md)
    # elif output_format == "pdf":
    #    md2pdf(path, md_content=newsletter_md)
    elif output_format == "html":
        css_style = Path(inspect.getfile(generate_newsletter)).parent / "newsletter.css"
        result = md2html(newsletter_md, css_style)
        with open(path, "w") as f:
            f.write(result)


def md2html(md: str, css_style: Path = None) -> str:
    """Convert a markdown string to HTML."""
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
