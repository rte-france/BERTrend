#  Copyright (c) 2024, RTE (https://www.rte-france.com)
#  See AUTHORS.txt
#  SPDX-License-Identifier: MPL-2.0
#  This file is part of BERTrend.

import pytest
import os
import json
from unittest.mock import patch, MagicMock

from bertrend.services.embedding_client import (
    EmbeddingAPIClient,
    MAX_DOCS_PER_REQUEST_PER_WORKER,
)


# Mock response for API calls
class MockResponse:
    def __init__(self, json_data, status_code):
        self.json_data = json_data
        self.status_code = status_code

    def json(self):
        return self.json_data


@pytest.fixture
def mock_client_secret():
    os.environ["BERTREND_CLIENT_SECRET"] = "test_secret"
    yield
    del os.environ["BERTREND_CLIENT_SECRET"]


@pytest.fixture
def embedding_client(mock_client_secret):
    with patch("bertrend.services.embedding_client.requests.get") as mock_get:
        # Mock responses for get_api_model_name and get_num_workers
        mock_get.side_effect = [
            MockResponse("test_model", 200),  # For get_api_model_name
            MockResponse(4, 200),  # For get_num_workers
        ]

        # Mock _get_access_token to avoid actual API calls
        with patch(
            "bertrend.services.authentication.SecureAPIClient._get_access_token"
        ):
            client = EmbeddingAPIClient(
                url="https://test-api.com",
                client_id="test_client",
                client_secret="test_secret",
            )
            return client


def test_initialization(embedding_client):
    """Test client initialization"""
    assert embedding_client.url == "https://test-api.com"
    assert embedding_client.client_id == "test_client"
    assert embedding_client.client_secret == "test_secret"
    assert embedding_client.model_name == "test_model"
    assert embedding_client.num_workers == 4


def test_get_api_model_name():
    """Test get_api_model_name method"""
    with patch("bertrend.services.embedding_client.requests.get") as mock_get:
        mock_get.return_value = MockResponse("test_model", 200)

        # Mock _get_access_token to avoid actual API calls
        with patch(
            "bertrend.services.authentication.SecureAPIClient._get_access_token"
        ):
            client = EmbeddingAPIClient(
                url="https://test-api.com",
                client_id="test_client",
                client_secret="test_secret",
            )
            # Reset mock to clear the calls made during initialization
            mock_get.reset_mock()
            mock_get.return_value = MockResponse("test_model", 200)

            model_name = client.get_api_model_name()
            assert model_name == "test_model"
            mock_get.assert_called_once_with(
                "https://test-api.com/model_name", verify=False
            )


def test_get_api_model_name_error():
    """Test get_api_model_name method with error response"""
    with patch("bertrend.services.embedding_client.requests.get") as mock_get:
        mock_get.return_value = MockResponse("test_model", 200)

        # Mock _get_access_token to avoid actual API calls
        with patch(
            "bertrend.services.authentication.SecureAPIClient._get_access_token"
        ):
            client = EmbeddingAPIClient(
                url="https://test-api.com",
                client_id="test_client",
                client_secret="test_secret",
            )
            # Reset mock to clear the calls made during initialization
            mock_get.reset_mock()
            mock_get.return_value = MockResponse(None, 500)

            with pytest.raises(Exception) as excinfo:
                client.get_api_model_name()
            assert "Error: 500" in str(excinfo.value)


def test_get_num_workers():
    """Test get_num_workers method"""
    with patch("bertrend.services.embedding_client.requests.get") as mock_get:
        # First set up the mock for initialization
        mock_get.side_effect = [
            MockResponse("test_model", 200),  # For get_api_model_name during init
            MockResponse(4, 200),  # For get_num_workers during init
        ]

        # Mock _get_access_token to avoid actual API calls
        with patch(
            "bertrend.services.authentication.SecureAPIClient._get_access_token"
        ):
            client = EmbeddingAPIClient(
                url="https://test-api.com",
                client_id="test_client",
                client_secret="test_secret",
            )

            # Reset mock and set a new return value for the actual test
            mock_get.reset_mock()
            mock_get.side_effect = None  # Clear side_effect
            mock_get.return_value = MockResponse(8, 200)

            num_workers = client.get_num_workers()
            assert num_workers == 8
            mock_get.assert_called_once_with(
                "https://test-api.com/num_workers", verify=False
            )


def test_get_num_workers_error():
    """Test get_num_workers method with error response"""
    with patch("bertrend.services.embedding_client.requests.get") as mock_get:
        # First set up the mock for initialization
        mock_get.side_effect = [
            MockResponse("test_model", 200),  # For get_api_model_name during init
            MockResponse(4, 200),  # For get_num_workers during init
        ]

        # Mock _get_access_token to avoid actual API calls
        with patch(
            "bertrend.services.authentication.SecureAPIClient._get_access_token"
        ):
            client = EmbeddingAPIClient(
                url="https://test-api.com",
                client_id="test_client",
                client_secret="test_secret",
            )

            # Reset mock and set a new return value for the actual test
            mock_get.reset_mock()
            mock_get.side_effect = None  # Clear side_effect
            mock_get.return_value = MockResponse(None, 500)

            with pytest.raises(Exception) as excinfo:
                client.get_num_workers()
            assert "Error: 500" in str(excinfo.value)


def test_embed_query():
    """Test embed_query method with a single string"""
    with patch("bertrend.services.embedding_client.requests.get") as mock_get:
        mock_get.side_effect = [
            MockResponse("test_model", 200),  # For get_api_model_name
            MockResponse(4, 200),  # For get_num_workers
        ]

        # Mock _get_access_token and _get_headers to avoid actual API calls
        with patch(
            "bertrend.services.authentication.SecureAPIClient._get_access_token"
        ):
            with patch(
                "bertrend.services.authentication.SecureAPIClient._get_headers",
                return_value={"Authorization": "Bearer test_token"},
            ):
                with patch(
                    "bertrend.services.embedding_client.requests.post"
                ) as mock_post:
                    # Create a mock embedding vector
                    mock_embedding = [0.1, 0.2, 0.3, 0.4]
                    mock_post.return_value = MockResponse(
                        {"embeddings": [mock_embedding]}, 200
                    )

                    client = EmbeddingAPIClient(
                        url="https://test-api.com",
                        client_id="test_client",
                        client_secret="test_secret",
                    )
                    embedding = client.embed_query("test text")

                    assert embedding == mock_embedding
                    mock_post.assert_called_once_with(
                        "https://test-api.com/encode",
                        data=json.dumps(
                            {"text": ["test text"], "show_progress_bar": False}
                        ),
                        verify=False,
                        headers={"Authorization": "Bearer test_token"},
                    )


def test_embed_query_list():
    """Test embed_query method with a list of strings"""
    with patch("bertrend.services.embedding_client.requests.get") as mock_get:
        mock_get.side_effect = [
            MockResponse("test_model", 200),  # For get_api_model_name
            MockResponse(4, 200),  # For get_num_workers
        ]

        # Mock _get_access_token and _get_headers to avoid actual API calls
        with patch(
            "bertrend.services.authentication.SecureAPIClient._get_access_token"
        ):
            with patch(
                "bertrend.services.authentication.SecureAPIClient._get_headers",
                return_value={"Authorization": "Bearer test_token"},
            ):
                with patch(
                    "bertrend.services.embedding_client.requests.post"
                ) as mock_post:
                    # Create a mock embedding vector
                    mock_embedding = [0.1, 0.2, 0.3, 0.4]
                    mock_post.return_value = MockResponse(
                        {"embeddings": [mock_embedding]}, 200
                    )

                    client = EmbeddingAPIClient(
                        url="https://test-api.com",
                        client_id="test_client",
                        client_secret="test_secret",
                    )
                    embedding = client.embed_query(["test text"])

                    assert embedding == mock_embedding
                    mock_post.assert_called_once_with(
                        "https://test-api.com/encode",
                        data=json.dumps(
                            {"text": ["test text"], "show_progress_bar": False}
                        ),
                        verify=False,
                        headers={"Authorization": "Bearer test_token"},
                    )


def test_embed_query_error():
    """Test embed_query method with error response"""
    with patch("bertrend.services.embedding_client.requests.get") as mock_get:
        mock_get.side_effect = [
            MockResponse("test_model", 200),  # For get_api_model_name
            MockResponse(4, 200),  # For get_num_workers
        ]

        # Mock _get_access_token and _get_headers to avoid actual API calls
        with patch(
            "bertrend.services.authentication.SecureAPIClient._get_access_token"
        ):
            with patch(
                "bertrend.services.authentication.SecureAPIClient._get_headers",
                return_value={"Authorization": "Bearer test_token"},
            ):
                with patch(
                    "bertrend.services.embedding_client.requests.post"
                ) as mock_post:
                    mock_post.return_value = MockResponse(None, 500)

                    client = EmbeddingAPIClient(
                        url="https://test-api.com",
                        client_id="test_client",
                        client_secret="test_secret",
                    )
                    result = client.embed_query("test text")

                    assert result is None


def test_embed_batch():
    """Test embed_batch method"""
    with patch("bertrend.services.embedding_client.requests.get") as mock_get:
        mock_get.side_effect = [
            MockResponse("test_model", 200),  # For get_api_model_name
            MockResponse(4, 200),  # For get_num_workers
        ]

        # Mock _get_access_token and _get_headers to avoid actual API calls
        with patch(
            "bertrend.services.authentication.SecureAPIClient._get_access_token"
        ):
            with patch(
                "bertrend.services.authentication.SecureAPIClient._get_headers",
                return_value={"Authorization": "Bearer test_token"},
            ):
                with patch(
                    "bertrend.services.embedding_client.requests.post"
                ) as mock_post:
                    # Create mock embedding vectors
                    mock_embeddings = [[0.1, 0.2], [0.3, 0.4]]
                    mock_post.return_value = MockResponse(
                        {"embeddings": mock_embeddings}, 200
                    )

                    client = EmbeddingAPIClient(
                        url="https://test-api.com",
                        client_id="test_client",
                        client_secret="test_secret",
                    )
                    embeddings = client.embed_batch(["text1", "text2"])

                    assert embeddings == mock_embeddings
                    mock_post.assert_called_once_with(
                        "https://test-api.com/encode",
                        data=json.dumps(
                            {"text": ["text1", "text2"], "show_progress_bar": True}
                        ),
                        verify=False,
                        headers={"Authorization": "Bearer test_token"},
                    )


def test_embed_batch_error():
    """Test embed_batch method with error response"""
    with patch("bertrend.services.embedding_client.requests.get") as mock_get:
        mock_get.side_effect = [
            MockResponse("test_model", 200),  # For get_api_model_name
            MockResponse(4, 200),  # For get_num_workers
        ]

        # Mock _get_access_token and _get_headers to avoid actual API calls
        with patch(
            "bertrend.services.authentication.SecureAPIClient._get_access_token"
        ):
            with patch(
                "bertrend.services.authentication.SecureAPIClient._get_headers",
                return_value={"Authorization": "Bearer test_token"},
            ):
                with patch(
                    "bertrend.services.embedding_client.requests.post"
                ) as mock_post:
                    mock_post.return_value = MockResponse(None, 500)

                    client = EmbeddingAPIClient(
                        url="https://test-api.com",
                        client_id="test_client",
                        client_secret="test_secret",
                    )
                    result = client.embed_batch(["text1", "text2"])

                    assert result == []


def test_embed_documents():
    """Test embed_documents method"""
    with patch("bertrend.services.embedding_client.requests.get") as mock_get:
        mock_get.side_effect = [
            MockResponse("test_model", 200),  # For get_api_model_name
            MockResponse(4, 200),  # For get_num_workers
        ]

        # Mock _get_access_token and _get_headers to avoid actual API calls
        with patch(
            "bertrend.services.authentication.SecureAPIClient._get_access_token"
        ):
            with patch(
                "bertrend.services.authentication.SecureAPIClient._get_headers",
                return_value={"Authorization": "Bearer test_token"},
            ):
                # Create a mock for Parallel that returns a callable
                mock_parallel_instance = MagicMock()
                mock_embeddings1 = [[0.1, 0.2]]
                mock_embeddings2 = [[0.3, 0.4]]
                mock_parallel_instance.return_value = [
                    mock_embeddings1,
                    mock_embeddings2,
                ]

                mock_parallel = MagicMock(return_value=mock_parallel_instance)

                with patch(
                    "bertrend.services.embedding_client.Parallel",
                    return_value=mock_parallel_instance,
                ):
                    client = EmbeddingAPIClient(
                        url="https://test-api.com",
                        client_id="test_client",
                        client_secret="test_secret",
                    )
                    embeddings = client.embed_documents(
                        ["text1", "text2"], batch_size=1
                    )

                    assert embeddings == [mock_embeddings1[0], mock_embeddings2[0]]


def test_embed_documents_too_many():
    """Test embed_documents method with too many documents"""
    with patch("bertrend.services.embedding_client.requests.get") as mock_get:
        mock_get.side_effect = [
            MockResponse("test_model", 200),  # For get_api_model_name
            MockResponse(1, 200),  # For get_num_workers (set to 1 for this test)
        ]

        # Mock _get_access_token to avoid actual API calls
        with patch(
            "bertrend.services.authentication.SecureAPIClient._get_access_token"
        ):
            client = EmbeddingAPIClient(
                url="https://test-api.com",
                client_id="test_client",
                client_secret="test_secret",
            )

            # Create a list with more documents than allowed
            too_many_docs = ["text"] * (MAX_DOCS_PER_REQUEST_PER_WORKER + 1)

            with pytest.raises(ValueError) as excinfo:
                client.embed_documents(too_many_docs)
            assert "Too many documents to be embedded" in str(excinfo.value)


def test_embed_documents_batch_failure():
    """Test embed_documents method with a batch failure"""
    with patch("bertrend.services.embedding_client.requests.get") as mock_get:
        mock_get.side_effect = [
            MockResponse("test_model", 200),  # For get_api_model_name
            MockResponse(4, 200),  # For get_num_workers
        ]

        # Mock _get_access_token and _get_headers to avoid actual API calls
        with patch(
            "bertrend.services.authentication.SecureAPIClient._get_access_token"
        ):
            with patch(
                "bertrend.services.authentication.SecureAPIClient._get_headers",
                return_value={"Authorization": "Bearer test_token"},
            ):
                # Create a mock for Parallel that returns a callable
                mock_parallel_instance = MagicMock()
                # One batch succeeds, one fails
                mock_parallel_instance.return_value = [[[0.1, 0.2]], []]

                with patch(
                    "bertrend.services.embedding_client.Parallel",
                    return_value=mock_parallel_instance,
                ):
                    client = EmbeddingAPIClient(
                        url="https://test-api.com",
                        client_id="test_client",
                        client_secret="test_secret",
                    )

                    with pytest.raises(ValueError) as excinfo:
                        client.embed_documents(["text1", "text2"], batch_size=1)
                    assert "At least one batch processing failed" in str(excinfo.value)


def test_aembed_query():
    """Test aembed_query method"""
    with patch("bertrend.services.embedding_client.requests.get") as mock_get:
        mock_get.side_effect = [
            MockResponse("test_model", 200),  # For get_api_model_name
            MockResponse(4, 200),  # For get_num_workers
        ]

        # Mock _get_access_token and _get_headers to avoid actual API calls
        with patch(
            "bertrend.services.authentication.SecureAPIClient._get_access_token"
        ):
            with patch(
                "bertrend.services.authentication.SecureAPIClient._get_headers",
                return_value={"Authorization": "Bearer test_token"},
            ):
                with patch(
                    "bertrend.services.embedding_client.requests.post"
                ) as mock_post:
                    # Create a mock embedding vector
                    mock_embedding = [0.1, 0.2, 0.3, 0.4]
                    mock_post.return_value = MockResponse(
                        {"embeddings": [mock_embedding]}, 200
                    )

                    client = EmbeddingAPIClient(
                        url="https://test-api.com",
                        client_id="test_client",
                        client_secret="test_secret",
                    )

                    # Test the async method (which currently just calls the sync method)
                    import asyncio

                    embedding = asyncio.run(client.aembed_query("test text"))

                    assert embedding == mock_embedding


def test_aembed_documents():
    """Test aembed_documents method"""
    with patch("bertrend.services.embedding_client.requests.get") as mock_get:
        mock_get.side_effect = [
            MockResponse("test_model", 200),  # For get_api_model_name
            MockResponse(4, 200),  # For get_num_workers
        ]

        # Mock _get_access_token and _get_headers to avoid actual API calls
        with patch(
            "bertrend.services.authentication.SecureAPIClient._get_access_token"
        ):
            with patch(
                "bertrend.services.authentication.SecureAPIClient._get_headers",
                return_value={"Authorization": "Bearer test_token"},
            ):
                # Create a mock for Parallel that returns a callable
                mock_parallel_instance = MagicMock()
                # Create mock embedding vectors
                mock_embeddings = [[[0.1, 0.2]], [[0.3, 0.4]]]
                mock_parallel_instance.return_value = mock_embeddings

                with patch(
                    "bertrend.services.embedding_client.Parallel",
                    return_value=mock_parallel_instance,
                ):
                    client = EmbeddingAPIClient(
                        url="https://test-api.com",
                        client_id="test_client",
                        client_secret="test_secret",
                    )

                    # Test the async method (which currently just calls the sync method)
                    import asyncio

                    embeddings = asyncio.run(
                        client.aembed_documents(["text1", "text2"])
                    )

                    assert embeddings == [mock_embeddings[0][0], mock_embeddings[1][0]]
