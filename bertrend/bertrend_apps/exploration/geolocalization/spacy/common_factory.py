#  Copyright (c) 2024-2026, RTE (https://www.rte-france.com)
#  See AUTHORS.txt
#  SPDX-License-Identifier: MPL-2.0
#  This file is part of BERTrend.

import re
import unicodedata
from typing import List, Union

from loguru import logger
from spacy import Language
from spacy.tokens import Token

from bertrend.bertrend_apps.exploration.geolocalization import DBPediaEntityLinker

RE_NORM_APOSTROPHE_DASH = re.compile(r"(-|l'|d')")


@Language.factory("normalizer_component")
def create_normalizer_component(nlp, name):
    return NormalizerComponent()


@Language.factory("tokenizer_component")
def create_tokenizer_component(nlp, name):
    return TokenizerComponent()


@Language.factory(
    "DBpedia_spotlight",
    default_config={
        "language_code": "fr",
        "dbpedia_rest_endpoint": None,
        "process": "annotate",
        "confidence": 0.5,
        "support": "",
        "types": None,
        "sparql": "",
        "policy": "",
        "span_group": "dbpedia_spotlight",
        "overwrite_ents": True,
    },
)
def dbpedia_spotlight_factory(
    nlp: Language,
    name: str,
    language_code: str,
    dbpedia_rest_endpoint: str,
    process: str,
    confidence: float,
    support: str,
    types: List[str],
    sparql: str,
    policy: str,
    span_group: str,
    overwrite_ents: bool,
):
    """Factory of the pipeline stage `dbpedia_spotlight`.
    Parameters:
    - `language_code`: which language to use for entity linking. Possible values are listed in EntityLinker.supported_languages. If the parameter is left as None, the language code is matched with the nlp object currently used.
    - `dbpedia_rest_endpoint`: this needs to be configured if you want to use a different REST endpoint from the default `EntityLinker.base_url`. Example: `http://localhost:2222/rest` for a localhost server
    - `process`: (REST API path) which of the processes to use from DBpedia Spotlight (see https://www.dbpedia-spotlight.org/api). The value can be 'annotate', 'spot' or 'candidates'
    - `confidence`: (REST API parameter) confidence score for disambiguation / linking
    - `support`: (REST API parameter) how prominent is this entity in Lucene Model, i.e. number of inlinks in Wikipedia
    - `types`: (REST API parameter) types filter (Eg.DBpedia:Place)
    - `sparql`: (REST API parameter) SPARQL filtering
    - `policy`: (REST API parameter) (whitelist) select all entities that have the same type; (blacklist) - select all entities that have not the same type.
    - `span_group`: which span group to write the entities to. By default the value is `dbpedia_spotlight` which writes to `doc.spans['dbpedia_spotlight']`
    - `overwrite_ents`: if set to False, it won't overwrite `doc.ents` in cases of overlapping spans with current entities, and only produce the results in `doc.spans[span_group]. If it is True, it will move the entities from doc.ents into `doc.spans['ents_original']`
    """
    logger.debug(
        f"dbpedia_spotlight_factory: {nlp}, language_code {language_code}, "
        f"dbpedia_rest_endpoint {dbpedia_rest_endpoint}, process {process}, "
        f"confidence {confidence}, support {support}, types {types}, "
        f"sparql {sparql}, policy {policy}, overwrite_ents {overwrite_ents}"
    )
    # take the language code from the nlp object
    nlp_lang_code = nlp.meta["lang"]
    # language_code can override the language code from the nlp object
    if not language_code:
        language_code = nlp_lang_code
    return DBPediaEntityLinker(
        language_code,
        dbpedia_rest_endpoint,
        process,
        confidence,
        support,
        types,
        sparql,
        policy,
        span_group,
        overwrite_ents,
    )


class NormalizerComponent:
    """
    Do some custom normalization steps
    WARNING: This DOES NOT modify utils length (applied to .norm_ token attribute)
    """

    def __call__(self, doc):
        for token in doc:
            norm = normalize(token)
            token.norm_ = norm
        return doc


class TokenizerComponent:
    """Split tokens by apostrophes and dashes"""

    def __call__(self, doc):
        with doc.retokenize() as retokenizer:
            for token in doc:
                text = token.text
                if len(text) > 1 and RE_NORM_APOSTROPHE_DASH.search(text):
                    elts = RE_NORM_APOSTROPHE_DASH.split(text)
                    elts = list(filter(bool, elts))
                    heads = [(token, i) for i in range(len(elts))]
                    retokenizer.split(token, elts, heads=heads)
        return doc


def strip_accents(s: str) -> str:
    # From https://stackoverflow.com/questions/517923/what-is-the-best-way-to-remove-accents-normalize-in-a-python-unicode-string
    return "".join(
        c for c in unicodedata.normalize("NFD", s) if unicodedata.category(c) != "Mn"
    )


def normalize(s: Union[str, Token]) -> str:
    if isinstance(s, Token):
        norm = s.lower_
    elif isinstance(s, str):
        norm = s.lower()
    norm = strip_accents(norm)
    norm = norm.strip()
    if norm == "saint":
        norm = "st"
    if norm == "sainte":
        norm = "ste"
    return norm
