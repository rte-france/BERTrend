#  Copyright (c) 2024, RTE (https://www.rte-france.com)
#  See AUTHORS.txt
#  SPDX-License-Identifier: MPL-2.0
#  This file is part of BERTrend.
import inspect
import os
from pathlib import Path

import jinja2

# from md2pdf.core import md2pdf
import pandas as pd
import tldextract
from bertopic._bertopic import BERTopic
from loguru import logger
from tqdm import tqdm

from bertrend import LLM_CONFIG
from bertrend.llm_utils.newsletter_model import (
    STRONG_TOPIC_TYPE,
    Article,
    Newsletter,
    Topic,
)
from bertrend.llm_utils.openai_client import OpenAI_Client
from bertrend.llm_utils.prompts import (
    USER_GENERATE_TOPIC_LABEL_SUMMARIES,
    USER_SUMMARY_MULTIPLE_DOCS,
)
from bertrend.services.summarizer import Summarizer
from bertrend.services.summary.chatgpt_summarizer import GPTSummarizer
from bertrend.topic_analysis.representative_docs import get_most_representative_docs

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
) -> Newsletter:
    """Generates a newsletter based on a trained BERTopic model.

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
        Newsletter type object
    """
    logger.debug("Generating newsletters...")
    if not openai_model_name:
        openai_model_name = LLM_CONFIG["model"]
    openai_api = OpenAI_Client(
        api_key=LLM_CONFIG["api_key"],
        base_url=LLM_CONFIG["base_url"],
        model=openai_model_name,
    )

    # Instantiates summarizer
    summarizer = summarizer_class()

    # Ensure top_n_topics is smaller than number of topics
    topics_info = topic_model.get_topic_info()[1:]

    if top_n_topics is None or top_n_topics > len(topics_info):
        top_n_topics = len(topics_info)

    # Create the Newsletter object
    newsletter = Newsletter(
        title=newsletter_title,
        period_start_date=df.timestamp.min().date(),
        period_end_date=df.timestamp.max().date(),
        topics=[],
    )

    # Iterate over topics
    for i in tqdm(range(top_n_topics), desc="Processing topics..."):
        topic_keywords = topics_info["Representation"].iloc[i]

        sub_df = get_most_representative_docs(
            topic_model=topic_model,
            df=df,
            topics=topics,
            mode=top_n_docs_mode,
            df_split=df_split,
            topic_number=i,
            top_n_docs=top_n_docs,
        )

        topic_summary = None

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
                    keywords=", ".join(topic_keywords),
                    article_list=article_list,
                    nb_sentences=nb_sentences,
                ),
                model=openai_model_name,
            )

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

            improved_topic_description_v2 = improved_topic_description_v2.removesuffix(
                "."
            )

            topic_title = improved_topic_description_v2
        else:
            topic_title = ", ".join(topic_keywords)

        i = 0
        article_list = []
        for _, doc in sub_df.iterrows():
            try:
                domain = tldextract.extract(doc.url).domain
            except:
                logger.warning(f"Cannot extract URL for {doc}")
                domain = None
            # Add the full text when no summarization is performed
            article_summary = (
                summaries[i]
                if summary_mode == "document"
                else doc.text
                if summary_mode == "none"
                else None
            )
            article_list.append(
                Article(
                    title=doc.title,
                    url=doc.url,
                    summary=article_summary,
                    date=doc.timestamp.date(),
                    source=domain,
                )
            )
            i += 1

        # Create topic object
        topic = Topic(
            title=topic_title,
            hashtags=topic_keywords,
            summary=topic_summary,
            articles=article_list,
            topic_type=STRONG_TOPIC_TYPE,
        )

        # Update newsletter object
        newsletter.topics.append(topic)

    return newsletter


def render_newsletter(
    newsletter: Newsletter, path: Path, output_format="md", language="fr"
):
    """Save a Markdown string to a file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    if output_format == "md":
        with open(path, "w") as f:
            f.write(render_newsletter_md(newsletter))
    elif output_format == "html":
        css_style = Path(inspect.getfile(generate_newsletter)).parent / "newsletter.css"
        html_template = (
            Path(inspect.getfile(generate_newsletter)).parent
            / "newsletter_outlook_template.html"
        )
        result = render_newsletter_html(
            newsletter,
            html_template=html_template,
            custom_css=None,
            language=language,
        )
        with open(path, "w") as f:
            f.write(result)


def render_newsletter_html(
    newsletter: Newsletter,
    html_template: Path = None,
    custom_css: Path = None,
    language: str = "fr",
) -> str:
    # Load the template from an external file
    with open(html_template, "r") as file:
        template_string = file.read()
    if custom_css is None:
        custom_css_content = ""
    else:
        with open(custom_css, "r") as file:
            custom_css_content = file.read()

    # Create a Jinja2 environment and compile the template
    template = jinja2.Template(template_string)

    # Render the template with data
    rendered_html = template.render(
        newsletter=newsletter, language=language, custom_css=custom_css_content
    )

    return rendered_html


def render_newsletter_md(newsletter: Newsletter) -> str:
    """
    Renders a Newsletter object into a Markdown formatted string.

    Args:
        newsletter (Newsletter): The newsletter object to render.

    Returns:
        str: The Markdown formatted string.
    """
    # Start with the title and period
    md_lines = [f"# {newsletter.title}"]
    md_lines.append(
        f"Period: {newsletter.period_start_date.strftime('%B %d, %Y')} to {newsletter.period_end_date.strftime('%B %d, %Y')}"
    )

    # Iterate over topics
    for topic in newsletter.topics:
        # Topic Header
        md_lines.append(f"\n## {topic.title}")

        # Topic Hashtags
        if topic.hashtags:
            md_lines.append(
                f"Hashtags: {' '.join([f'#{tag}' for tag in topic.hashtags])}"
            )

        # Topic Summary
        if topic.summary:
            md_lines.append(f"\n{topic.summary}")

        # Articles
        if topic.articles:
            for article in topic.articles:
                md_lines.append(f"\n### {article.title}")

                # Link to article if URL exists
                if article.url:
                    md_lines.append(f"[Link to Article]({article.url})")

                # Article Date
                md_lines.append(f"Date: {article.date.strftime('%B %d, %Y')}")

                # Article Source
                if article.source:
                    md_lines.append(f"Source: {article.source}")

                # Article Summary
                if article.summary:
                    md_lines.append(f"\n{article.summary}")

        # Add extra spacing between topics
        md_lines.append("\n---")

    # Return the Markdown content as a string
    return "\n".join(md_lines)
