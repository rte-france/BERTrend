"""
Test script to verify that the installation process works correctly with the new dependency management structure.
This script attempts to import key components from each dependency group.
"""

#  Copyright (c) 2024-2026, RTE (https://www.rte-france.com)
#  See AUTHORS.txt
#  SPDX-License-Identifier: MPL-2.0
#  This file is part of BERTrend.

import pytest


def test_core_dependencies():
    """Test that core dependencies for topic modeling and analysis are installed correctly."""
    try:
        import bertopic
        import gensim
        import hdbscan
        import numpy
        import pandas
        import scipy
        import sentence_transformers
        import sklearn
        import torch
        import umap

        print("✅ Core dependencies imported successfully")
    except ImportError as e:
        pytest.fail(f"❌ Failed to import core dependencies: {e}")


def test_nlp_dependencies():
    """Test that NLP and text processing dependencies are installed correctly."""
    try:
        import langdetect
        import lxml_html_clean
        import markdown
        import nltk
        import sentencepiece
        import thefuzz
        import tldextract
        import tokenizers

        print("✅ NLP dependencies imported successfully")
    except ImportError as e:
        pytest.fail(f"❌ Failed to import NLP dependencies: {e}")


def test_llm_dependencies():
    """Test that LLM integration dependencies are installed correctly."""
    try:
        import langchain_core
        import openai
        import tiktoken

        print("✅ LLM dependencies imported successfully")
    except ImportError as e:
        pytest.fail(f"❌ Failed to import LLM dependencies: {e}")


def test_visualization_dependencies():
    """Test that visualization and UI dependencies are installed correctly."""
    try:
        import datamapplot
        import plotly
        import plotly_resampler
        import pylabeladjust
        import pyqtree
        import seaborn
        import streamlit

        print("✅ Visualization dependencies imported successfully")
    except ImportError as e:
        pytest.fail(f"❌ Failed to import visualization dependencies: {e}")


def test_utility_dependencies():
    """Test that utility dependencies are installed correctly."""
    try:
        import black
        import cron_descriptor
        import dask
        import dateparser
        import dill
        import joblib
        import jsonlines
        import loguru
        import opentelemetry
        import requests
        import tqdm

        print("✅ Utility dependencies imported successfully")
    except ImportError as e:
        pytest.fail(f"❌ Failed to import utility dependencies: {e}")


def test_optional_apps_dependencies():
    """Test that optional apps dependencies are installed correctly."""
    try:
        # Only test a subset of apps dependencies to avoid errors if some are not installed
        import arxiv
        import feedparser
        import googleapiclient

        print("✅ Apps dependencies imported successfully")
    except ImportError as e:
        pytest.fail(f"❌ Failed to import apps dependencies: {e}")
