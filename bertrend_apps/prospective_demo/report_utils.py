#  Copyright (c) 2024, RTE (https://www.rte-france.com)
#  See AUTHORS.txt
#  SPDX-License-Identifier: MPL-2.0
#  This file is part of BERTrend.
from pathlib import Path


def generate_html_report(
    template_path: Path | str, report_title: str, data_list: list[dict]
) -> str:
    """
    Generate HTML report with single title and multiple topic containers.

    Args:
        template_path (str): Path to the HTML template file
        report_title (str): Single title for the entire report
        data_list (list): List of dictionaries containing data for each topic
            Each dict should have:
            - topic_title (str)
            - topic_description (str)
            - links (list): List of links
            - color (str): 'orange' or 'green'
            - analysis (str): HTML content for detailed analysis

    Returns:
        str: Complete HTML document
    """
    # Read the template file
    with open(template_path, "r", encoding="utf-8") as file:
        template = file.read()

    # Find the container div in the template
    container_start = template.find('<div class="container">')
    container_end = template.find("</div>\n\n<script>")

    # Split template into parts
    header = template[:container_start]
    container_template = template[container_start:container_end]
    footer = template[container_end:]

    # Create the header with the single report title
    header_with_title = header + f'<div class="container">\n<h1>{report_title}</h1>\n'

    # Generate topic containers
    topic_containers = []
    for data in data_list:
        # Create container for each topic
        container = f"""
    <h2 class="{data['color']}-title">{data['topic_title']}</h2>
    <p>
    {data['topic_description']}
    </p>
    """
        # Add links
        for link in data["links"]:
            container += f'<a href="{link}" target="_blank">{link}</a><br>\n'

        # Add details section
        container += f"""
    <div class="details">
        <details class="dropdown">
            <summary>Analyse détaillée</summary>
            <div class="dropdown-content">
                <div class="iframe-container">
                    <iframe id="myIframe_{len(topic_containers)}"></iframe>
                </div>
            </div>
        </details>
    </div>
    """
        topic_containers.append(container)

    # Create script section for all iframes
    scripts = ["<script>"]
    for i, data in enumerate(data_list):
        scripts.append(
            f"""
    const htmlString_{i} = `{data['analysis']}`;
    const iframe_{i} = document.getElementById('myIframe_{i}');
    iframe_{i}.srcdoc = htmlString_{i};
    """
        )
    scripts.append("</script>")

    # Combine all parts
    full_html = (
        header_with_title
        + "\n".join(topic_containers)
        + "</div>"  # Close the main container
        + "\n".join(scripts)
    )

    return full_html


# Example usage
if __name__ == "__main__":
    # Example data
    report_title = "LLM Updates Report - Q1 2024"
    data_list = [
        {
            "topic_title": "New Developments",
            "topic_description": "Recent advances in language models...",
            "color": "orange",
            "links": ["https://example.com/article1", "https://example.com/article2"],
            "analysis": """
                <h2>Detailed Analysis</h2>
                <p>Key findings from this week include...</p>
            """,
        },
        {
            "topic_title": "Industry Trends",
            "topic_description": "Market trends in AI adoption...",
            "color": "green",
            "links": [
                "https://example.com/trend1",
                "https://example.com/trend2",
                "https://example.com/trend3",
            ],
            "analysis": """
                <h2>Market Analysis</h2>
                <p>The industry has shown significant growth...</p>
            """,
        },
    ]

    # Generate the HTML
    output_html = generate_html_report("report_template.html", report_title, data_list)

    # Save to file
    with open("generated_report.html", "w", encoding="utf-8") as file:
        file.write(output_html)
