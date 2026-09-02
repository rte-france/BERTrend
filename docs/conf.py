"""Sphinx configuration for the BERTrend documentation.

The documentation is written in Markdown (parsed by MyST) and lives directly in
this folder, so that it stays readable when browsing the sources on GitHub.
Built on Read the Docs, see /.readthedocs.yaml.
"""

import tomllib
from pathlib import Path

# -- Project information -----------------------------------------------------

_pyproject = Path(__file__).parent.parent / "pyproject.toml"
with _pyproject.open("rb") as f:
    _metadata = tomllib.load(f)["project"]

project = "BERTrend"
author = "RTE France"
copyright = "RTE France - see LICENSE.md"
version = release = _metadata["version"]

# -- General configuration ---------------------------------------------------

extensions = [
    "myst_parser",
    "sphinxcontrib.mermaid",
    "sphinx_copybutton",
]

# `docs/README.md` is the GitHub-facing index; it is included by `index.md`
# instead of being built as a page of its own (which would duplicate it).
exclude_patterns = [
    "_build",
    "README.md",
    "requirements.txt",
    "Thumbs.db",
    ".DS_Store",
]

root_doc = "index"

# -- MyST (Markdown) options -------------------------------------------------

myst_enable_extensions = [
    "attrs_inline",
    "attrs_block",
    "colon_fence",
    "deflist",
    "fieldlist",
    "tasklist",
]
# Turn ```mermaid fences into `.. mermaid::` directives.
myst_fence_as_directive = ["mermaid"]
# Allow linking to `## section` anchors of other Markdown pages.
myst_heading_anchors = 4

# -- HTML output -------------------------------------------------------------

html_theme = "furo"
html_title = f"BERTrend {version}"
html_static_path = ["_static"]
html_css_files = ["custom.css"]

# `data_architecture.html` is a *self-contained* page (it carries its own
# `<html>`, styles and CSS-only tabs). It must not go through the Sphinx
# templating, so it is copied verbatim next to the generated pages: that keeps
# every existing relative link such as `[...](data_architecture.html)` valid,
# both on the documentation site and when browsing the repository.
html_extra_path = ["data_architecture.html"]

html_theme_options = {
    "source_repository": "https://github.com/rte-france/BERTrend/",
    "source_branch": "main",
    "source_directory": "docs/",
}

# Mermaid renders client-side; keep the version pinned to a major.
mermaid_version = "11"
