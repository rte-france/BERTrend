#  Copyright (c) 2024, RTE (https://www.rte-france.com)
#  See AUTHORS.txt
#  SPDX-License-Identifier: MPL-2.0
#  This file is part of BERTrend.

import os

import requests
import json

from joblib import Parallel, delayed
from langchain_core.embeddings import Embeddings
from loguru import logger

import numpy as np

from bertrend.services.authentication import SecureAPIClient

MAX_N_JOBS = 4
BATCH_DOCUMENT_SIZE = 1000
MAX_DOCS_PER_REQUEST_PER_WORKER = 20000


class EmbeddingAPIClient(SecureAPIClient, Embeddings):
    """
    Custom Embedding API client, can integrate seamlessly with langchain
    """

    def __init__(
        self,
        url: str,
        client_id: str = "bertrend",
        client_secret: str = os.getenv("BERTREND_CLIENT_SECRET", None),
    ):
        super().__init__(url, client_id, client_secret)
        self.model_name = self.get_api_model_name()
        self.num_workers = self.get_num_workers()

    def get_api_model_name(self) -> str:
        """
        Return currently loaded model name in Embedding API.
        """
        response = requests.get(
            self.url + "/model_name",
            verify=False,
        )
        if response.status_code == 200:
            model_name = response.json()
            logger.debug(f"Model name: {model_name}")
            return model_name
        else:
            logger.error(f"Error: {response.status_code}")
            raise Exception(f"Error: {response.status_code}")

    def get_num_workers(self) -> int:
        """
        Return currently loaded number of workers in Embedding API.
        """
        response = requests.get(
            self.url + "/num_workers",
            verify=False,
        )
        if response.status_code == 200:
            num_workers = response.json()
            logger.debug(f"Number of workers: {num_workers}")
            return num_workers
        else:
            logger.error(f"Error: {response.status_code}")
            raise Exception(f"Error: {response.status_code}")

    def embed_query(
        self, text: str | list[str], show_progress_bar: bool = False
    ) -> list[float]:
        if type(text) == str:
            text = [text]
        logger.debug(f"Calling EmbeddingAPI using model: {self.model_name}")
        logger.debug(f"Computing embeddings...")
        response = requests.post(
            self.url + "/encode",
            data=json.dumps({"text": text, "show_progress_bar": show_progress_bar}),
            verify=False,
            headers=self._get_headers(),
        )
        if response.status_code == 200:
            embeddings = np.array(response.json()["embeddings"])
            logger.debug(f"Computing embeddings done")
            return embeddings.tolist()[0]
        else:
            logger.error(f"Error: {response.status_code}")

    def embed_batch(
        self, texts: list[str], show_progress_bar: bool = True
    ) -> list[list[float]]:
        logger.debug(f"Computing embeddings...")
        response = requests.post(
            self.url + "/encode",
            data=json.dumps({"text": texts, "show_progress_bar": show_progress_bar}),
            verify=False,
            headers=self._get_headers(),
        )
        if response.status_code == 200:
            embeddings = np.array(response.json()["embeddings"])
            logger.debug(f"Computing embeddings done for batch")
            return embeddings.tolist()
        else:
            logger.error(f"Error: {response.status_code}")
            return []

    def embed_documents(
        self,
        texts: list[str],
        show_progress_bar: bool = True,
        batch_size: int = BATCH_DOCUMENT_SIZE,
    ) -> list[list[float]]:
        if len(texts) > MAX_DOCS_PER_REQUEST_PER_WORKER * self.num_workers:
            # Too many documents to embed in one request, refuse it
            logger.error(
                f"Error: Too many documents to be embedded ({len(texts)} chunks, max {MAX_DOCS_PER_REQUEST_PER_WORKER * self.num_workers})"
            )
            raise ValueError(
                f"Error: Too many documents to be embedded ({len(texts)} chunks, max {MAX_DOCS_PER_REQUEST_PER_WORKER * self.num_workers})"
            )

        logger.debug(f"Calling EmbeddingAPI using model: {self.model_name}")

        # Split texts into chunks
        batches = [texts[i : i + batch_size] for i in range(0, len(texts), batch_size)]
        logger.debug(
            f"Computing embeddings on {len(texts)} documents using ({len(batches)}) batches..."
        )

        # Parallel request
        results = Parallel(n_jobs=MAX_N_JOBS)(
            delayed(self.embed_batch)(batch, show_progress_bar) for batch in batches
        )

        # Check results
        if any(result == [] for result in results):
            raise ValueError(
                "At least one batch processing failed. Documents are not embedded."
            )

        # Compile results
        embeddings = [embedding for result in results for embedding in result]
        assert len(embeddings) == len(texts)
        return embeddings

    async def aembed_query(self, text: str) -> list[float]:
        # FIXME!
        return self.embed_query(text)

    async def aembed_documents(self, texts: list[str]) -> list[list[float]]:
        # FIXME!
        return self.embed_documents(texts)
