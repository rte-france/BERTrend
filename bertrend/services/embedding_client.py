#  Copyright (c) 2024-2026, RTE (https://www.rte-france.com)
#  See AUTHORS.txt
#  SPDX-License-Identifier: MPL-2.0
#  This file is part of BERTrend.

import json
import os

import numpy as np
import requests
from dotenv import load_dotenv
from joblib import Parallel, delayed
from langchain_core.embeddings import Embeddings
from loguru import logger

from bertrend.services.authentication import SecureAPIClient

MAX_N_JOBS = 4
BATCH_DOCUMENT_SIZE = 1000
MAX_DOCS_PER_REQUEST_PER_WORKER = 20000

load_dotenv(override=True)


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
        with requests.get(
            self.url + "/model_name",
            verify=False,
        ) as response:
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
        with requests.get(
            self.url + "/num_workers",
            verify=False,
        ) as response:
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
        if isinstance(text, str):
            text = [text]
        logger.debug(f"Calling EmbeddingAPI using model: {self.model_name}")
        logger.debug("Computing embeddings...")
        with requests.post(
            self.url + "/encode",
            data=json.dumps({"text": text, "show_progress_bar": show_progress_bar}),
            verify=False,
            headers=self._get_headers(),
        ) as response:
            if response.status_code == 200:
                embeddings = np.array(response.json()["embeddings"])
                logger.debug("Computing embeddings done")
                return embeddings.tolist()[0]
            else:
                logger.error(f"Error: {response.status_code}")

    def embed_batch(
        self, texts: list[str], show_progress_bar: bool = True
    ) -> list[list[float]]:
        logger.debug("Computing embeddings...")
        with requests.post(
            self.url + "/encode",
            data=json.dumps({"text": texts, "show_progress_bar": show_progress_bar}),
            verify=False,
            headers=self._get_headers(),
        ) as response:
            if response.status_code == 200:
                embeddings = np.array(response.json()["embeddings"])
                logger.debug("Computing embeddings done for batch")
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
        # The embedding service can only handle a bounded number of documents at a
        # time (MAX_DOCS_PER_REQUEST_PER_WORKER per worker). Rather than refusing
        # larger inputs — which previously raised a ValueError and blocked the whole
        # embedding step (see issue #42) — process them in sequential chunks that
        # each stay within the limit and concatenate the results. This keeps the
        # 1:1 mapping between input documents and returned embeddings that callers
        # rely on, so no document is dropped.
        max_docs_per_call = MAX_DOCS_PER_REQUEST_PER_WORKER * max(self.num_workers, 1)
        if len(texts) > max_docs_per_call:
            logger.warning(
                f"{len(texts)} documents exceed the maximum handled in a single call "
                f"({max_docs_per_call}); processing them in sequential chunks to avoid "
                f"overloading the embedding service."
            )
            embeddings: list[list[float]] = []
            for start in range(0, len(texts), max_docs_per_call):
                embeddings.extend(
                    self.embed_documents(
                        texts[start : start + max_docs_per_call],
                        show_progress_bar=show_progress_bar,
                        batch_size=batch_size,
                    )
                )
            assert len(embeddings) == len(texts)
            return embeddings

        logger.debug(f"Calling EmbeddingAPI using model: {self.model_name}")

        # Split texts into chunks
        batches = [texts[i : i + batch_size] for i in range(0, len(texts), batch_size)]
        logger.info(
            f"Computing embeddings on {len(texts)} documents using ({len(batches)}) batches..."
        )

        # Parallel request
        #
        # We explicitly use a *thread-based* backend here. Using the default
        # (process-based) backend would require pickling ``self`` and the
        # underlying SecureAPIClient state (e.g. auth/session objects), which
        # is fragile and can easily break. Threads share the same context and
        # avoid these issues while still allowing concurrent HTTP requests.
        results = Parallel(n_jobs=MAX_N_JOBS, prefer="threads")(
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
