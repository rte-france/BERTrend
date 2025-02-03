#  Copyright (c) 2024, RTE (https://www.rte-france.com)
#  See AUTHORS.txt
#  SPDX-License-Identifier: MPL-2.0
#  This file is part of BERTrend.

import hashlib
import pickle
from pathlib import Path
from typing import Any

import os

# Ensures to write with +rw for both user and groups
os.umask(0o002)


def load_embeddings(cache_path: Path):
    """Loads embeddings as pickle"""
    with open(cache_path, "rb") as f_in:
        return pickle.load(f_in)


def save_embeddings(embeddings: list, cache_path: Path):
    """Save embeddings as pickle"""
    with open(cache_path, "wb") as f_out:
        pickle.dump(embeddings, f_out)


def get_hash(data: Any) -> str:
    """Returns a *stable* hash(persistent between different Python session) for any object. NB. The default hash() function does not guarantee this."""
    return hashlib.md5(repr(data).encode("utf-8")).hexdigest()
