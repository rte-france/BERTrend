# BERTrend Integration Tests

This directory contains integration tests for the BERTrend project. These tests verify that the different components of the system work together correctly in real-world scenarios.

## Overview

The integration tests are organized into several test classes, each focusing on a specific integration point:

1. **TestEmbeddingServiceIntegration**: Tests the integration of the EmbeddingService with local embedding models.
2. **TestBERTopicModelIntegration**: Tests the integration of BERTopicModel with pre-computed embeddings.
3. **TestBERTrendIntegration**: Tests the end-to-end workflow of BERTrend, including training topic models, calculating signal popularity, and classifying signals.
4. **TestDataPipelineIntegration**: Tests the data loading and processing pipeline.
5. **TestNewsletterIntegration**: Tests the newsletter generation functionality, which combines topic modeling with LLM-based summarization.

## Running the Tests

To run the integration tests, you need to have the BERTrend package installed with all its dependencies. You can run the tests using pytest with the following command:

```bash
pytest -m integration bertrend/tests/test_integration.py -v
```

This will run only the tests marked with the `integration` marker. The `-v` flag provides verbose output.

### Running Specific Tests

To run a specific test class, you can use:

```bash
pytest -m integration bertrend/tests/test_integration.py::TestBERTrendIntegration -v
```

To run a specific test method, you can use:

```bash
pytest -m integration bertrend/tests/test_integration.py::TestBERTrendIntegration::test_bertrend_end_to_end -v
```

## Test Requirements

The integration tests have the following requirements:

1. **Sentence Transformer Models**: The tests use the `all-MiniLM-L6-v2` model, which will be downloaded automatically if not already present.
2. **OpenAI API Key**: The newsletter generation test requires an OpenAI API key. If the key is not available, the test will be skipped.
3. **Temporary Directory**: The tests use pytest's `tmp_path` fixture to create temporary directories for saving models and files.

## Adding New Integration Tests

When adding new integration tests, please follow these guidelines:

1. Use the `@pytest.mark.integration` decorator to mark the test as an integration test.
2. Use small, fast models and synthetic data to ensure the tests run efficiently.
3. Include assertions that verify the correct integration between components.
4. Handle external dependencies gracefully, skipping tests if necessary.
5. Clean up any resources created during the test.

## Example

Here's an example of a simple integration test:

```python
@pytest.mark.integration
def test_embedding_service_with_local_model():
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
```