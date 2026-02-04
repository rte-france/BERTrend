#  Copyright (c) 2024, RTE (https://www.rte-france.com)
#  See AUTHORS.txt
#  SPDX-License-Identifier: MPL-2.0
#  This file is part of BERTrend.

import os
import ssl

import nltk
from nltk.corpus import stopwords

# Ensures files are written with +rw permissions for both user and groups
os.umask(0o002)

def ensure_stopwords():
    """Check if NLTK stopwords are available locally before downloading."""
    try:
        # Try to access stopwords to check if they're already downloaded
        stopwords.words("english")
    except LookupError:
        nltk.download("stopwords")


# Workaround for downloading nltk data in some environments
try:
    _create_unverified_https_context = ssl._create_unverified_context
except AttributeError:
    pass
else:
    ssl._create_default_https_context = _create_unverified_https_context
ensure_stopwords()
