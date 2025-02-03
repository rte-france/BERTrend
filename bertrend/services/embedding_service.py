#  Copyright (c) 2024, RTE (https://www.rte-france.com)
#  See AUTHORS.txt
#  SPDX-License-Identifier: MPL-2.0
#  This file is part of BERTrend.
import json
from typing import Literal

import numpy as np
import pandas as pd
import requests
import torch
from bertopic.backend import BaseEmbedder
from loguru import logger
from sentence_transformers import SentenceTransformer
from tqdm import tqdm

from bertrend import EMBEDDING_CONFIG
from bertrend.config.parameters import (
    EMBEDDING_DEVICE,
    EMBEDDING_BATCH_SIZE,
    EMBEDDING_MAX_SEQ_LENGTH,
)


class EmbeddingService(BaseEmbedder):
    """Class implementing embedding service."""

    def __init__(
        self,
        local: bool = EMBEDDING_CONFIG.get("use_local", True),
        model_name: str = EMBEDDING_CONFIG.get("model_name", None),
        embedding_dtype: Literal[
            "float32", "float16", "bfloat16"
        ] = EMBEDDING_CONFIG.get("embedding_dtype", "float32"),
        host: str = EMBEDDING_CONFIG["host"],
        port: str = EMBEDDING_CONFIG["port"],
    ):
        """
        Class implementing embedding service.
        :param local (bool): indicates whether to use local or remote embeddings service.
        :param model_name (str): The name of the Sentence Transformer model to use.
        :param embedding_dtype: (Literal): The data type to use for the embeddings ('float32', 'float16', or 'bfloat16').
        :param host (str): The host address of the remote embedding service to use.
        :param port (str): The port of the remote embedding service to use.
        """
        super().__init__()
        self.local = local
        if not self.local:
            self.url = f"http://{host}:{port}"

        self.embedding_model = None
        self.embedding_model_name = model_name
        self.embedding_dtype = embedding_dtype

    def embed(self, texts: list[str] | pd.Series, verbose: bool = False) -> tuple[
        np.ndarray,
        list[list[str]] | None,
        list[np.ndarray] | None,
    ]:
        """
        Embed a list of documents using a Sentence Transformer model.

        This function loads a specified Sentence Transformer model and uses it to create
        embeddings for a list of input texts. It processes the texts in batches to manage
        memory efficiently, especially for large datasets.

        Args:
            texts (Union[List[str], pd.Series]): A list of text documents to be embedded.
            verbose (bool): Level of output details. Defaults to False.

        Returns:
            Tuple[SentenceTransformer, np.ndarray]: A tuple containing:
                - A numpy array of embeddings, where each row corresponds to a text in the input list.
                - A list of grouped token strings
                - A list of grouped token embeddings
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

        Args:
            texts (List[str]): A list of text documents to be embedded.
            embedding_device (str, optional): The device to use for embedding ('cuda' or 'cpu').
                                              Defaults to 'cuda' if available, else 'cpu'.
            batch_size (int, optional): The number of texts to process in each batch. Defaults to 32.
            max_seq_length (int, optional): The maximum sequence length for the model. Defaults to 512.

        Returns:
            Tuple[SentenceTransformer, np.ndarray]: A tuple containing:
                - A numpy array of embeddings, where each row corresponds to a text in the input list.
                - A list of grouped token strings
                - A list of grouped token embeddings

        Raises:
            ValueError: If an invalid embedding_dtype is provided.
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
        self, texts: list[str], show_progress_bar: bool = True
    ) -> tuple[np.ndarray, None, None]:
        """
        Embed a list of documents using a Sentence Transformer model.

        This function loads a specified Sentence Transformer model and uses it to create
        embeddings for a list of input texts. It processes the texts in batches to manage
        memory efficiently, especially for large datasets.

        Args:
            texts (List[str]): A list of text documents to be embedded.
            show_progress_bar (bool): Progress bar display on service side. Defaults to True.

        Returns:
            Tuple[SentenceTransformer, np.ndarray]: A tuple containing:
                - A numpy array of embeddings, where each row corresponds to a text in the input list.
                - A list of grouped token strings
                - A list of grouped token embeddings
        """
        logger.debug(f"Computing embeddings...")
        response = requests.post(
            self.url + "/encode",
            data=json.dumps({"text": texts, "show_progress_bar": show_progress_bar}),
        )
        if response.status_code == 200:
            embeddings = np.array(response.json()["embeddings"])
            logger.debug(f"Computing embeddings done for batch")
            self.embedding_model = self._get_remote_model_name()
            return embeddings, None, None
        else:
            logger.error(f"Error: {response.status_code}")
            raise Exception(f"Error: {response.status_code}")

    def _get_remote_model_name(self) -> str:
        """
        Return currently loaded model name in Embedding API.
        """
        response = requests.get(
            self.url + "/model_name",
        )
        if response.status_code == 200:
            model_name = response.json()
            logger.debug(f"Model name: {model_name}")
            return model_name
        else:
            logger.error(f"Error: {response.status_code}")
            raise Exception(f"Error: {response.status_code}")


def _convert_to_numpy(obj, type=None):
    """
    Convert a torch.Tensor or list of torch.Tensors to numpy arrays.
    Args:
        obj: The object to convert (torch.Tensor or list).
        type: The type of conversion (optional, used for token ids).
    Returns:
        np.ndarray or list of np.ndarray.
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
    Args:
        tokenizer: The tokenizer to use for converting ids to tokens.
        token_ids: List of token ids.
        token_embeddings: List of token embeddings.
        language: The language of the tokens (default is "french").
    Returns:
        List of grouped tokens and their corresponding embeddings.
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
