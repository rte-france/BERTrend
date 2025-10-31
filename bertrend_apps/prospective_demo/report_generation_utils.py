#  Copyright (c) 2024, RTE (https://www.rte-france.com)
#  See AUTHORS.txt
#  SPDX-License-Identifier: MPL-2.0
#  This file is part of BERTrend.
import inspect
import tempfile
from pathlib import Path

from jinja2 import Environment, FileSystemLoader

from bertrend.llm_utils.newsletter_features import generate_newsletter
from bertrend_apps.prospective_demo.data_model import DetailedNewsletter

MAXIMUM_NUMBER_OF_ARTICLES = 3


def render_html_report(
    newsletter: DetailedNewsletter,
    language: str = "fr",
) -> str:

    template_dirs = [
        Path(__file__).parent,  # Current directory
        Path(
            inspect.getfile(generate_newsletter)
        ).parent,  # Main template ("newsletter_outlook_template.html")
    ]

    # Set up the Jinja2 environment to look in both directories
    env = Environment(loader=FileSystemLoader(template_dirs))

    # Render the template with data
    template = env.get_template("detailed_report_template.html")
    rendered_html = template.render(
        newsletter=newsletter, language=language, custom_css=""
    )

    return rendered_html


def create_temp_report(html_content) -> Path:
    with tempfile.NamedTemporaryFile(
        mode="w", suffix=".html", delete=False
    ) as temp_file:
        temp_file.write(html_content)
        return Path(temp_file.name)
