#  Copyright (c) 2024, RTE (https://www.rte-france.com)
#  See AUTHORS.txt
#  SPDX-License-Identifier: MPL-2.0
#  This file is part of BERTrend.
import json
from typing import List, Tuple

import numpy as np
import requests
import torch
from loguru import logger
from sentence_transformers import SentenceTransformer
from tqdm import tqdm

from bertrend import EMBEDDING_CONFIG
from bertrend.parameters import (
    EMBEDDING_DEVICE,
    EMBEDDING_BATCH_SIZE,
    EMBEDDING_MAX_SEQ_LENGTH,
)


class EmbeddingService:
    def __init__(self, local: bool = True, embedding_model_name: str = None):
        self.local = local
        if not self.local:
            self.port = EMBEDDING_CONFIG["port"]
            self.host = EMBEDDING_CONFIG["host"]
            self.url = f"http://{self.host}:{self.port}"
        self.embedding_model_name = embedding_model_name

    # TODO: harmonize interfaces for local / remote services

    def embed_documents(
        self,
        texts: List[str],
        embedding_model_name: str,
        embedding_dtype: str,
    ) -> Tuple[SentenceTransformer, np.ndarray]:
        """
        Embed a list of documents using a Sentence Transformer model.

        This function loads a specified Sentence Transformer model and uses it to create
        embeddings for a list of input texts. It processes the texts in batches to manage
        memory efficiently, especially for large datasets.

        Args:
            texts (List[str]): A list of text documents to be embedded.
            embedding_model_name (str): The name of the Sentence Transformer model to use.
            embedding_dtype (str): The data type to use for the embeddings ('float32', 'float16', or 'bfloat16').
            embedding_device (str, optional): The device to use for embedding ('cuda' or 'cpu').
                                              Defaults to 'cuda' if available, else 'cpu'.
            batch_size (int, optional): The number of texts to process in each batch. Defaults to 32.
            max_seq_length (int, optional): The maximum sequence length for the model. Defaults to 512.

        Returns:
            Tuple[SentenceTransformer, np.ndarray]: A tuple containing:
                - The loaded and configured Sentence Transformer model.
                - A numpy array of embeddings, where each row corresponds to a text in the input list.

        Raises:
            ValueError: If an invalid embedding_dtype is provided.
        """
        if self.local:
            return self._local_embed_documents(
                texts,
                embedding_model_name,
                embedding_dtype,
            )
        else:
            return self._remote_embed_documents(
                texts,
                embedding_model_name,
                embedding_dtype,
            )

    def _local_embed_documents(
        self,
        texts: List[str],
        embedding_model_name: str,
        embedding_dtype: str,
        embedding_device: str = EMBEDDING_DEVICE,
        batch_size: int = EMBEDDING_BATCH_SIZE,
        max_seq_length: int = EMBEDDING_MAX_SEQ_LENGTH,
    ) -> Tuple[SentenceTransformer, np.ndarray]:
        """
        Embed a list of documents using a Sentence Transformer model.

        This function loads a specified Sentence Transformer model and uses it to create
        embeddings for a list of input texts. It processes the texts in batches to manage
        memory efficiently, especially for large datasets.

        Args:
            texts (List[str]): A list of text documents to be embedded.
            embedding_model_name (str): The name of the Sentence Transformer model to use.
            embedding_dtype (str): The data type to use for the embeddings ('float32', 'float16', or 'bfloat16').
            embedding_device (str, optional): The device to use for embedding ('cuda' or 'cpu').
                                              Defaults to 'cuda' if available, else 'cpu'.
            batch_size (int, optional): The number of texts to process in each batch. Defaults to 32.
            max_seq_length (int, optional): The maximum sequence length for the model. Defaults to 512.

        Returns:
            Tuple[SentenceTransformer, np.ndarray]: A tuple containing:
                - The loaded and configured Sentence Transformer model.
                - A numpy array of embeddings, where each row corresponds to a text in the input list.

        Raises:
            ValueError: If an invalid embedding_dtype is provided.
        """
        # Configure model kwargs based on the specified dtype
        model_kwargs = {}
        if embedding_dtype == "float16":
            model_kwargs["torch_dtype"] = torch.float16
        elif embedding_dtype == "bfloat16":
            model_kwargs["torch_dtype"] = torch.bfloat16
        elif embedding_dtype != "float32":
            raise ValueError(
                "Invalid embedding_dtype. Must be 'float32', 'float16', or 'bfloat16'."
            )

        # Load the embedding model
        embedding_model = SentenceTransformer(
            embedding_model_name,
            device=embedding_device,
            trust_remote_code=True,
            model_kwargs=model_kwargs,
        )
        embedding_model.max_seq_length = max_seq_length

        # Calculate the number of batches
        num_batches = (len(texts) + batch_size - 1) // batch_size

        # Initialize an empty list to store embeddings
        embeddings = []

        # Process texts in batches
        for i in tqdm(range(num_batches), desc="Processing batches"):
            start_idx = i * batch_size
            end_idx = min(start_idx + batch_size, len(texts))
            batch_texts = texts[start_idx:end_idx]
            batch_embeddings = embedding_model.encode(
                batch_texts, show_progress_bar=False
            )
            embeddings.append(batch_embeddings)

        # Concatenate all batch embeddings
        embeddings = np.concatenate(embeddings, axis=0)

        return embedding_model, embeddings

    def _remote_embed_documents(
        self, texts: List[str], show_progress_bar: bool = True
    ) -> List[List[float]]:
        """
        Embed a list of documents using a Sentence Transformer model.

        This function loads a specified Sentence Transformer model and uses it to create
        embeddings for a list of input texts. It processes the texts in batches to manage
        memory efficiently, especially for large datasets.

        Args:
            texts (List[str]): A list of text documents to be embedded.
            embedding_model_name (str): The name of the Sentence Transformer model to use.
            embedding_dtype (str): The data type to use for the embeddings ('float32', 'float16', or 'bfloat16').
            embedding_device (str, optional): The device to use for embedding ('cuda' or 'cpu').
                                              Defaults to 'cuda' if available, else 'cpu'.
            batch_size (int, optional): The number of texts to process in each batch. Defaults to 32.
            max_seq_length (int, optional): The maximum sequence length for the model. Defaults to 512.

        Returns:
            Tuple[SentenceTransformer, np.ndarray]: A tuple containing:
                - The loaded and configured Sentence Transformer model.
                - A numpy array of embeddings, where each row corresponds to a text in the input list.

        Raises:
            ValueError: If an invalid embedding_dtype is provided.
        """
        # TODO: implement call to embedding service (cf wattElse)
        logger.debug(f"Computing embeddings...")
        response = requests.post(
            self.url + "/encode",
            data=json.dumps({"text": texts, "show_progress_bar": show_progress_bar}),
        )
        if response.status_code == 200:
            embeddings = np.array(response.json()["embeddings"])
            logger.debug(f"Computing embeddings done for batch")
            return embeddings.tolist()
        else:
            logger.error(f"Error: {response.status_code}")
            return []
