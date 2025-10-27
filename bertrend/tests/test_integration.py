#  Copyright (c) 2024, RTE (https://www.rte-france.com)
#  See AUTHORS.txt
#  SPDX-License-Identifier: MPL-2.0
#  This file is part of BERTrend.

import os
import pytest
import pandas as pd
from datetime import datetime, timedelta
from unittest.mock import Mock, patch

from bertrend.BERTrend import BERTrend
from bertrend.BERTopicModel import BERTopicModel
from bertrend.services.embedding_service import EmbeddingService
from bertrend.utils.data_loading import load_data, split_data, group_by_days
from bertrend.llm_utils.newsletter_features import (
    generate_newsletter,
    render_newsletter,
)


class TestEmbeddingServiceIntegration:
    """Integration tests for the EmbeddingService."""

    @pytest.mark.integration
    def test_embedding_service_with_local_model(self):
        """Test that EmbeddingService works with a local model."""
        # Use a small, fast model for testing
        model_name = "all-MiniLM-L6-v2"

        # Create a small test dataset
        texts = [
            "This is a test document about artificial intelligence.",
            "Machine learning is a subset of artificial intelligence.",
            "Natural language processing is used in many applications.",
        ]

        # Initialize the embedding service with local model
        embedding_service = EmbeddingService(local=True, model_name=model_name)

        # Generate embeddings
        embeddings, token_strings, token_embeddings = embedding_service.embed(texts)

        # Verify the results
        assert embeddings is not None
        assert embeddings.shape == (
            len(texts),
            384,
        )  # 384 is the dimension for all-MiniLM-L6-v2
        assert token_strings is not None
        assert token_embeddings is not None
        assert len(token_strings) == len(texts)
        assert len(token_embeddings) == len(texts)


class TestBERTopicModelIntegration:
    """Integration tests for the BERTopicModel."""

    @pytest.mark.integration
    def test_bertopic_model_with_embeddings(self):
        """Test that BERTopicModel works with pre-computed embeddings."""
        # Create a small test dataset
        docs = [
            "This is a document about artificial intelligence.",
            "Machine learning is a subset of artificial intelligence.",
            "Natural language processing is used in many applications.",
            "Deep learning models require large amounts of data.",
            "Neural networks are inspired by the human brain.",
            "Computer vision is used for image recognition.",
            "Reinforcement learning is used for decision making.",
            "Supervised learning requires labeled data.",
            "Unsupervised learning finds patterns in unlabeled data.",
            "Transfer learning uses knowledge from one task for another.",
        ] * 5  # Duplicate to have enough data for clustering

        # Use a small, fast model for testing
        model_name = "all-MiniLM-L6-v2"

        # Initialize the embedding service
        embedding_service = EmbeddingService(local=True, model_name=model_name)

        # Generate embeddings
        embeddings, _, _ = embedding_service.embed(docs)

        # Initialize the BERTopicModel with parameters suitable for small test datasets
        topic_model = BERTopicModel(
            {
                "vectorizer_model": {
                    "min_df": 1
                },  # Reduce min_df to handle small datasets
                "umap_model": {
                    "n_components": 2,
                    "n_neighbors": 3,
                },  # Reduce dimensions for small datasets
                "hdbscan_model": {
                    "min_cluster_size": 2,
                    "min_samples": 1,
                },  # Adjust clustering for small datasets
            }
        )

        # Fit the model
        output = topic_model.fit(
            docs=docs, embeddings=embeddings, embedding_model=model_name
        )

        # Verify the results
        assert output is not None
        assert output.topic_model is not None
        assert output.topics is not None
        assert output.probs is not None
        assert output.embeddings is None  # Not stored in output
        assert len(output.topics) == len(docs)


class TestBERTrendIntegration:
    """Integration tests for the BERTrend class."""

    @pytest.fixture
    def sample_data(self):
        """Create a sample dataset for testing."""
        # Create timestamps spanning 3 months
        start_date = datetime(2023, 1, 1)
        dates = [start_date + timedelta(days=i) for i in range(0, 90, 3)]

        # Create sample data with timestamps
        data = []
        for i, date in enumerate(dates):
            data.append(
                {
                    "text": f"Document {i} about artificial intelligence and machine learning.",
                    "timestamp": date,
                    "document_id": f"doc_{i}",
                    "source": f"source_{i % 3}",
                    "url": f"http://example.com/{i}",
                }
            )
            data.append(
                {
                    "text": f"Document {i} about natural language processing and neural networks.",
                    "timestamp": date,
                    "document_id": f"doc_{i+100}",
                    "source": f"source_{i % 3}",
                    "url": f"http://example.com/{i+100}",
                }
            )
            data.append(
                {
                    "text": f"Document {i} about computer vision and deep learning.",
                    "timestamp": date,
                    "document_id": f"doc_{i+200}",
                    "source": f"source_{i % 3}",
                    "url": f"http://example.com/{i+200}",
                }
            )

        return pd.DataFrame(data)

    @pytest.mark.integration
    def test_bertrend_end_to_end(self, sample_data, tmp_path):
        """Test the end-to-end workflow of BERTrend."""
        # Use a small, fast model for testing
        model_name = "all-MiniLM-L6-v2"

        # Initialize the embedding service
        embedding_service = EmbeddingService(local=True, model_name=model_name)

        # Generate embeddings for the entire dataset
        embeddings, _, _ = embedding_service.embed(sample_data["text"])

        # Initialize BERTopicModel with parameters suitable for small test datasets
        topic_model = BERTopicModel(
            {
                "vectorizer_model": {
                    "min_df": 1
                },  # Reduce min_df to handle small datasets
                "umap_model": {
                    "n_components": 2,
                    "n_neighbors": 3,
                },  # Reduce dimensions for small datasets
                "hdbscan_model": {
                    "min_cluster_size": 2,
                    "min_samples": 1,
                },  # Adjust clustering for small datasets
            }
        )

        # Initialize BERTrend
        bertrend = BERTrend(topic_model=topic_model)

        # Group data by days
        grouped_data = group_by_days(sample_data, day_granularity=7)

        # Train topic models
        try:
            bertrend.train_topic_models(
                grouped_data=grouped_data,
                embedding_model=model_name,
                embeddings=embeddings,
                bertrend_models_path=tmp_path,
                save_topic_models=True,
            )
            print("train_topic_models completed successfully")
        except Exception as e:
            print(f"Error in train_topic_models: {e}")
            raise

        # Calculate signal popularity
        try:
            bertrend.calculate_signal_popularity()
            print("calculate_signal_popularity completed successfully")
        except Exception as e:
            print(f"Error in calculate_signal_popularity: {e}")
            raise

        # Classify signals
        try:
            current_date = sample_data["timestamp"].max()
            print(f"Current date: {current_date}")
            result = bertrend.classify_signals(
                window_size=30, current_date=current_date
            )
            print(f"classify_signals returned: {result}")
            noise_df, weak_signal_df, strong_signal_df = result
        except Exception as e:
            print(f"Error in classify_signals: {e}")
            raise

        # Verify the results
        print("Checking _is_fitted:", bertrend._is_fitted)
        assert bertrend._is_fitted is True

        print("Checking last_topic_model:", bertrend.last_topic_model)
        assert bertrend.last_topic_model is not None

        print(
            "Checking last_topic_model_timestamp:", bertrend.last_topic_model_timestamp
        )
        assert bertrend.last_topic_model_timestamp is not None

        print("Checking doc_groups:", len(bertrend.doc_groups))
        assert len(bertrend.doc_groups) > 0

        print("Checking emb_groups:", len(bertrend.emb_groups))
        assert len(bertrend.emb_groups) > 0

        print("Checking merged_df:", bertrend.merged_df)
        assert bertrend.merged_df is not None

        print("Checking all_merge_histories_df:", bertrend.all_merge_histories_df)
        assert bertrend.all_merge_histories_df is not None

        print("Checking all_new_topics_df:", bertrend.all_new_topics_df)
        assert bertrend.all_new_topics_df is not None

        # Verify that we have classified signals
        print("Checking noise_df:", noise_df)
        assert isinstance(noise_df, pd.DataFrame)

        print("Checking weak_signal_df:", weak_signal_df)
        assert isinstance(weak_signal_df, pd.DataFrame)

        print("Checking strong_signal_df:", strong_signal_df)
        assert isinstance(strong_signal_df, pd.DataFrame)

        # Save and restore the model
        bertrend.save_model(models_path=tmp_path)
        restored_bertrend = BERTrend.restore_model(models_path=tmp_path)

        # Verify the restored model
        assert restored_bertrend._is_fitted is True
        assert len(restored_bertrend.doc_groups) == len(bertrend.doc_groups)
        assert len(restored_bertrend.emb_groups) == len(bertrend.emb_groups)


class TestDataPipelineIntegration:
    """Integration tests for the data pipeline."""

    @pytest.fixture
    def sample_csv_file(self, tmp_path):
        """Create a sample CSV file for testing."""
        file_path = tmp_path / "sample_data.csv"

        # Create sample data
        data = []
        for i in range(20):
            data.append(
                {
                    "text": f"Document {i} about artificial intelligence.",
                    "timestamp": f"2023-01-{i+1:02d}",
                    "document_id": f"doc_{i}",
                    "source": f"source_{i % 3}",
                    "url": f"http://example.com/{i}",
                }
            )

        # Create DataFrame and save to CSV
        df = pd.DataFrame(data)
        df.to_csv(file_path, index=False)

        return file_path

    @pytest.mark.integration
    def test_data_loading_and_processing(self, sample_csv_file):
        """Test loading data from a file and processing it."""
        # Load data from CSV file
        df = load_data(sample_csv_file)

        # Verify the loaded data
        assert isinstance(df, pd.DataFrame)
        assert len(df) == 20
        assert "text" in df.columns
        assert "timestamp" in df.columns
        assert "document_id" in df.columns
        assert "source" in df.columns
        assert "url" in df.columns

        # Split data by paragraphs
        split_df = split_data(df)

        # Verify the split data
        assert isinstance(split_df, pd.DataFrame)
        assert len(split_df) >= len(df)  # Should be at least the same size

        # Group data by days
        grouped_data = group_by_days(df, day_granularity=7)

        # Verify the grouped data
        assert isinstance(grouped_data, dict)
        assert all(isinstance(key, pd.Timestamp) for key in grouped_data.keys())
        assert all(isinstance(value, pd.DataFrame) for value in grouped_data.values())


class TestNewsletterIntegration:
    """Integration tests for the newsletter generation functionality."""

    @pytest.mark.integration
    @pytest.mark.skipif(
        not os.environ.get("OPENAI_API_KEY"),
        reason="OPENAI_API_KEY not set - required for integration test",
    )
    def test_newsletter_generation(self, tmp_path):
        """Test generating a newsletter from topic model results.

        This is a real integration test that exercises the full pipeline including
        the LLM API. It requires OPENAI_API_KEY to be set in the environment.
        """

        # Create a sample dataset
        docs = [
            "This is a document about artificial intelligence.",
            "Machine learning is a subset of artificial intelligence.",
            "Natural language processing is used in many applications.",
            "Deep learning models require large amounts of data.",
            "Neural networks are inspired by the human brain.",
            "Computer vision is used for image recognition.",
            "Reinforcement learning is used for decision making.",
            "Supervised learning requires labeled data.",
            "Unsupervised learning finds patterns in unlabeled data.",
            "Transfer learning uses knowledge from one task for another.",
        ] * 5  # Duplicate to have enough data for clustering

        # Create a DataFrame with the necessary columns
        df = pd.DataFrame(
            {
                "text": docs,
                "title": [f"Article {i} Title" for i in range(len(docs))],
                "timestamp": [
                    datetime(2023, 1, 1) + timedelta(days=i) for i in range(len(docs))
                ],
                "document_id": [f"doc_{i}" for i in range(len(docs))],
                "source": [f"source_{i % 3}" for i in range(len(docs))],
                "url": [f"http://example.com/{i}" for i in range(len(docs))],
            }
        )

        # Use a small, fast model for testing
        model_name = "all-MiniLM-L6-v2"

        # Initialize the embedding service
        embedding_service = EmbeddingService(local=True, model_name=model_name)

        # Generate embeddings
        embeddings, _, _ = embedding_service.embed(docs)

        # Initialize and fit the BERTopicModel with parameters suitable for small test datasets
        topic_model = BERTopicModel(
            {
                "vectorizer_model": {
                    "min_df": 1
                },  # Reduce min_df to handle small datasets
                "umap_model": {
                    "n_components": 2,
                    "n_neighbors": 3,
                },  # Reduce dimensions for small datasets
                "hdbscan_model": {
                    "min_cluster_size": 2,
                    "min_samples": 1,
                },  # Adjust clustering for small datasets
            }
        )
        output = topic_model.fit(
            docs=docs, embeddings=embeddings, embedding_model=model_name
        )

        # Generate the newsletter
        newsletter = generate_newsletter(
            topic_model=output.topic_model,
            topics=output.topics,
            df=df,
            top_n_topics=3,
            top_n_docs=2,
            newsletter_title="Test Newsletter",
            summary_mode="document",
        )

        # Export the newsletter to markdown
        md_path = tmp_path / "newsletter.md"
        render_newsletter(newsletter, md_path, output_format="md")

        # Verify the results
        assert newsletter is not None
        assert newsletter.title == "Test Newsletter"
        assert newsletter.period_start_date == df.timestamp.min().date()
        assert newsletter.period_end_date == df.timestamp.max().date()
        assert len(newsletter.topics) <= 3
        assert md_path.exists()

        # Verify newsletter structure
        for topic in newsletter.topics:
            assert topic.topic_type is not None
            assert len(topic.articles) > 0
            for article in topic.articles:
                assert article.title is not None
                assert article.url is not None
