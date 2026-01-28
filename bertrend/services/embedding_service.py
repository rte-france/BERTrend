#  Copyright (c) 2024, RTE (https://www.rte-france.com)
#  See AUTHORS.txt
#  SPDX-License-Identifier: MPL-2.0
#  This file is part of BERTrend.
import json
import os
from typing import Literal

import numpy as np
import pandas as pd
import requests
import torch
from bertopic.backend import BaseEmbedder
from dotenv import load_dotenv
from loguru import logger
from sentence_transformers import SentenceTransformer
from tqdm import tqdm

from bertrend import EMBEDDING_CONFIG
from bertrend.config.parameters import (
    EMBEDDING_DEVICE,
    EMBEDDING_BATCH_SIZE,
    EMBEDDING_MAX_SEQ_LENGTH,
)
from bertrend.services.embedding_client import EmbeddingAPIClient

load_dotenv(override=True)


class EmbeddingService(BaseEmbedder):
    """
    Service for generating text embeddings using local or remote embedding models.

    This class provides functionality to embed text documents using either a local
    Sentence Transformer model or a remote embedding service. It supports batched
    processing for efficient memory usage and can return both document-level embeddings
    and token-level embeddings.

    The class inherits from BERTopic's BaseEmbedder, allowing it to be used directly
    with BERTopic models for topic modeling.
    """

    def __init__(
        self,
        local: bool = EMBEDDING_CONFIG.get("use_local", True),
        model_name: str = EMBEDDING_CONFIG.get("model_name", None),
        embedding_dtype: Literal[
            "float32", "float16", "bfloat16"
        ] = EMBEDDING_CONFIG.get("embedding_dtype", "float32"),
        url: str = EMBEDDING_CONFIG["url"],
        client_id: str = "bertrend",
        client_secret: str = os.getenv("BERTREND_CLIENT_SECRET", None),
    ):
        """
        Class implementing embedding service.

        Parameters
        ----------
        local : bool
            Indicates whether to use local or remote embedding service.
        model_name : str
            The name of the Sentence Transformer model to use.
        embedding_dtype : Literal['float32', 'float16', 'bfloat16']
            The data type to use for the embeddings.
        url : str
            The URL of the remote embedding service to use.
        client_id : str
            The client ID for authentication with the remote service.
        client_secret : str
            The client secret for authentication with the remote service.
        """
        super().__init__()
        self.local = local
        if not self.local:
            self.url = url
            self.secure_client = EmbeddingAPIClient(self.url, client_id, client_secret)

        self.embedding_model = None
        self.embedding_model_name = model_name
        self.embedding_dtype = embedding_dtype

    def embed(
        self, texts: list[str] | pd.Series, verbose: bool = False
    ) -> tuple[
        np.ndarray,
        list[list[str]] | None,
        list[np.ndarray] | None,
    ]:
        """
        Embed a list of documents using a Sentence Transformer model.

        This function loads a specified Sentence Transformer model and uses it to create
        embeddings for a list of input texts. It processes the texts in batches to manage
        memory efficiently, especially for large datasets.

        Parameters
        ----------
        texts : list[str] or pd.Series
            A list of text documents to be embedded.
        verbose : bool, default=False
            Level of output details.

        Returns
        -------
        tuple
            A tuple containing:
            - np.ndarray : A numpy array of embeddings, where each row corresponds to a text in the input list.
            - list[list[str]] or None : A list of grouped token strings
            - list[np.ndarray] or None : A list of grouped token embeddings
        """
        # Convert to list if input is a pandas Series
        if isinstance(texts, pd.Series):
            texts = texts.tolist()
        if self.local:
            return self._local_embed_documents(texts)
        else:
            return self._remote_embed_documents(texts)

    def _local_embed_documents(
        self,
        texts: list[str],
        embedding_device: str = EMBEDDING_DEVICE,
        batch_size: int = EMBEDDING_BATCH_SIZE,
        max_seq_length: int = EMBEDDING_MAX_SEQ_LENGTH,
    ) -> tuple[np.ndarray, list[list[str]], list[np.ndarray]]:
        """
        Embed a list of documents using a Sentence Transformer model.

        This function loads a specified Sentence Transformer model and uses it to create
        embeddings for a list of input texts. It processes the texts in batches to manage
        memory efficiently, especially for large datasets.

        Parameters
        ----------
        texts : list[str]
            A list of text documents to be embedded.
        embedding_device : str, optional
            The device to use for embedding ('cuda' or 'cpu').
            Defaults to 'cuda' if available, else 'cpu'.
        batch_size : int, optional
            The number of texts to process in each batch. Defaults to 32.
        max_seq_length : int, optional
            The maximum sequence length for the model. Defaults to 512.

        Returns
        -------
        tuple
            A tuple containing:
            - np.ndarray : A numpy array of embeddings, where each row corresponds to a text in the input list.
            - list[list[str]] : A list of grouped token strings
            - list[np.ndarray] : A list of grouped token embeddings

        Raises
        ------
        ValueError
            If an invalid embedding_dtype is provided.
        """
        # Configure model kwargs based on the specified dtype
        model_kwargs = {}
        if self.embedding_dtype == "float16":
            model_kwargs["torch_dtype"] = torch.float16
        elif self.embedding_dtype == "bfloat16":
            model_kwargs["torch_dtype"] = torch.bfloat16
        elif self.embedding_dtype != "float32":
            raise ValueError(
                "Invalid embedding_dtype. Must be 'float32', 'float16', or 'bfloat16'."
            )

        if self.embedding_model is None:
            logger.info(f"Loading embedding model: {self.embedding_model_name}...")
            # Load the embedding model
            self.embedding_model = SentenceTransformer(
                self.embedding_model_name,
                device=embedding_device,
                trust_remote_code=True,
                model_kwargs=model_kwargs,
            )
            self.embedding_model.max_seq_length = max_seq_length
            self.batch_size = batch_size
            logger.debug("Embedding model loaded")

        # Calculate the number of batches
        num_batches = (len(texts) + batch_size - 1) // batch_size

        # Initialize an empty list to store embeddings
        embeddings = []

        # Process texts in batches
        for i in tqdm(range(num_batches), desc="Processing batches"):
            start_idx = i * batch_size
            end_idx = min(start_idx + batch_size, len(texts))
            batch_texts = texts[start_idx:end_idx]
            batch_embeddings = self.embedding_model.encode(
                batch_texts,
                show_progress_bar=False,
                output_value=None,  # to get all output values, not only sentence embeddings
            )
            embeddings.append(batch_embeddings)

        # Concatenate all batch embeddings
        all_embeddings = np.concatenate(embeddings, axis=0)
        logger.success(f"Embedded {len(texts)} documents in {num_batches} batches")

        embeddings = [
            item["sentence_embedding"].detach().cpu() for item in all_embeddings
        ]
        embeddings = torch.stack(embeddings)
        embeddings = embeddings.numpy()

        token_embeddings = [
            item["token_embeddings"].detach().cpu() for item in all_embeddings
        ]
        token_ids = [item["input_ids"].detach().cpu() for item in all_embeddings]

        token_embeddings = _convert_to_numpy(token_embeddings)
        token_ids = _convert_to_numpy(token_ids, type="token_id")

        tokenizer = self.embedding_model._first_module().tokenizer

        token_strings, token_embeddings = _group_tokens(
            tokenizer, token_ids, token_embeddings, language="french"
        )

        return embeddings, token_strings, token_embeddings

    def _remote_embed_documents(
        self,
        texts: list[str],
        show_progress_bar: bool = True,
    ) -> tuple[np.ndarray, None, None]:
        """
        Embed a list of documents using a remote embedding service.
        The remote embedding service is assumed to have at least the endpoints /encode, /token and /model_name
        The service is assumed to require authentication using OAuth2 protocol. Credentials have to be
        provided for the "bertrend" app.

        Parameters
        ----------
        texts : list[str]
            A list of text documents to be embedded.
        show_progress_bar : bool, default=True
            Progress bar display on service side.

        Returns
        -------
        tuple
            A tuple containing:
            - np.ndarray : A numpy array of embeddings, where each row corresponds to a text in the input list.
            - None : Placeholder for token strings (not available in remote mode)
            - None : Placeholder for token embeddings (not available in remote mode)
        """
        logger.debug(f"Computing embeddings...")
        embeddings = self.secure_client.embed_documents(
            texts, show_progress_bar=show_progress_bar
        )
        return np.array(embeddings), None, None

    def _get_remote_model_name(self) -> str:
        """
        Return currently loaded model name in Embedding API.

        Returns
        -------
        str
            The name of the model currently loaded in the Embedding API.

        Raises
        ------
        Exception
            If the API request fails.
        """
        with requests.get(self.url + "/model_name", verify=False) as response:
            if response.status_code == 200:
                model_name = response.json()
                logger.debug(f"Remote model name: {model_name}")
                return model_name
            else:
                logger.error(f"Error: {response.status_code}")
                raise Exception(f"Error: {response.status_code}")


def _convert_to_numpy(obj, type=None):
    """
    Convert a torch.Tensor or list of torch.Tensors to numpy arrays.

    Parameters
    ----------
    obj : torch.Tensor or list
        The object to convert.
    type : str, optional
        The type of conversion (used for token ids).

    Returns
    -------
    np.ndarray or list
        Converted numpy array or list of numpy arrays.

    Raises
    ------
    TypeError
        If the object is not a list or torch.Tensor.
    """
    if isinstance(obj, torch.Tensor):
        return (
            obj.numpy().astype(np.int64)
            if type == "token_id"
            else obj.numpy().astype(np.float32)
        )
    elif isinstance(obj, list):
        return [_convert_to_numpy(item) for item in obj]
    else:
        raise TypeError("Object must be a list or torch.Tensor")


def _group_tokens(tokenizer, token_ids, token_embeddings, language="french"):
    """
    Group split tokens into whole words and average their embeddings.

    Parameters
    ----------
    tokenizer : object
        The tokenizer to use for converting ids to tokens.
    token_ids : list
        List of token ids.
    token_embeddings : list
        List of token embeddings.
    language : str, default="french"
        The language of the tokens.

    Returns
    -------
    tuple
        A tuple containing:
        - list : List of grouped tokens
        - list : List of corresponding averaged embeddings
    """
    grouped_token_lists = []
    grouped_embedding_lists = []

    special_tokens = {
        "english": ["[CLS]", "[SEP]", "[PAD]"],
        "french": ["<s>", "</s>", "<pad>"],
    }
    subword_prefix = {"english": "##", "french": "▁"}

    for token_id, token_embedding in tqdm(
        zip(token_ids, token_embeddings), desc="Grouping split tokens into whole words"
    ):
        tokens = tokenizer.convert_ids_to_tokens(token_id)

        grouped_tokens = []
        grouped_embeddings = []
        current_word = ""
        current_embedding = []

        for token, embedding in zip(tokens, token_embedding):
            if token in special_tokens[language]:
                continue

            if language == "french" and token.startswith(subword_prefix[language]):
                if current_word:
                    grouped_tokens.append(current_word)
                    grouped_embeddings.append(np.mean(current_embedding, axis=0))
                current_word = token[1:]
                current_embedding = [embedding]
            elif language == "english" and not token.startswith(
                subword_prefix[language]
            ):
                if current_word:
                    grouped_tokens.append(current_word)
                    grouped_embeddings.append(np.mean(current_embedding, axis=0))
                current_word = token
                current_embedding = [embedding]
            else:
                current_word += token.lstrip(subword_prefix[language])
                current_embedding.append(embedding)

        if current_word:
            grouped_tokens.append(current_word)
            grouped_embeddings.append(np.mean(current_embedding, axis=0))

        grouped_token_lists.append(grouped_tokens)
        grouped_embedding_lists.append(np.array(grouped_embeddings))

    return grouped_token_lists, grouped_embedding_lists
