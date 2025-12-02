#  Copyright (c) 2024, RTE (https://www.rte-france.com)
#  See AUTHORS.txt
#  SPDX-License-Identifier: MPL-2.0
#  This file is part of Wattelse, a NLP application suite.

# NB. Largely inspired from https://github.com/MartinoMensio/spacy-dbpedia-spotlight (MIT license)

import spacy
import requests
from loguru import logger
from requests import HTTPError

from spacy.tokens import Span, Doc

DBPEDIA_SPOTLIGHT_DEFAULT_ENDPOINT = "https://api.dbpedia-spotlight.org"
DBPEDIA_ENT = "DBPedia"

# span extension attribute for raw json
Span.set_extension("dbpedia_raw_result", default=None, force=True)
Doc.set_extension("dbpedia_raw_result", default=None, force=True)


class DBPediaEntityLinker(object):
    """This class manages the querying of DBpedia and attaches the found entities to the document"""

    # default location of the service
    base_url = DBPEDIA_SPOTLIGHT_DEFAULT_ENDPOINT
    # list of supported languages
    supported_languages = ["en", "de", "es", "fr", "it", "nl", "pt", "ru"]
    # list of supported processes
    supported_processes = ["annotate", "spot", "candidates"]

    def __init__(
        self,
        language_code="en",
        dbpedia_rest_endpoint=None,
        process="annotate",
        confidence=None,
        support=None,
        types=None,
        sparql=None,
        policy=None,
        span_group=DBPEDIA_ENT,
        overwrite_ents=True,
    ):
        # constructor of the pipeline stage
        if language_code not in self.supported_languages:
            raise ValueError(
                f"Linker not available in {language_code}. Choose one of {self.supported_languages}"
            )
        self.language_code = language_code
        if process not in self.supported_processes:
            raise ValueError(
                f"The process {process} is not supported. Choose one of {self.supported_processes}"
            )
        self.process = process
        self.confidence = confidence
        self.support = support
        self.types = types
        self.sparql = sparql
        self.policy = policy
        self.span_group = span_group
        self.overwrite_ents = overwrite_ents
        self.dbpedia_rest_endpoint = dbpedia_rest_endpoint

    def get_uri(self, el):
        # fields have different names depending on the process
        if self.process == "annotate":
            return el["@URI"]
        elif self.process == "candidates":
            return f"http://dbpedia.org/resource/{el['resource']['@uri']}"
        return None

    def get_ents_list(self, json):
        # fields have different names depending on the process
        if self.process == "annotate":
            return json.get("Resources", [])
        return json.get("annotation", {}).get("surfaceForm", [])

    def __call__(self, doc):
        # called in the pipeline
        if self.dbpedia_rest_endpoint:
            # override the default endpoint, e.g., 'http://localhost:2222/rest'
            endpoint = self.dbpedia_rest_endpoint
        else:
            # use the default endpoint for the language selected
            endpoint = f"{self.base_url}/{self.language_code}"

        params = {"text": doc.text}
        if self.confidence:
            params["confidence"] = self.confidence
        if self.support:
            params["support"] = self.support
        if self.types:
            params["types"] = ",".join(
                self.types
            )  # JP: for proper formatting in the API
        if self.sparql:
            params["sparql"] = self.sparql
        if self.policy:
            params["policy"] = self.policy

        # TODO: application/ld+json would be more detailed? https://github.com/digitalbazaar/pyld
        try:
            with requests.post(
                f"{endpoint}/{self.process}",
                headers={"accept": "application/json"},
                data=params,
            ) as response:
                response.raise_for_status()
                data = response.json()
        except HTTPError as e:
            # due to too many requests to the endpoint - this happens sometimes with the default public endpoint
            logger.warn(
                f"Bad response from server {endpoint}, probably too many requests. Consider using your own endpoint. Document not updated."
            )
            logger.trace(str(e))
            return doc
        except Exception as e:  # other errors
            logger.error(
                f"Endpoint {endpoint} unreachable, please check your connection. Document not updated."
            )
            logger.trace(str(e))
            return doc
        logger.trace(f"Received data: {data}")

        doc._.dbpedia_raw_result = data

        ents_data = []

        # fields have different names depending on the process
        text_key = "@name"
        # get_offset
        if self.process == "annotate":
            text_key = "@surfaceForm"

        for ent in self.get_ents_list(data):
            start_ch = int(ent["@offset"])
            end_ch = int(start_ch + len(ent[text_key]))
            ent_kb_id = self.get_uri(ent)
            # TODO look at '@types' and choose most relevant?
            # JP: adaptation to indicate the type instead of a generic span type when possible.
            # In case of a span matching several types, we retain the first one (the assumption is that types
            # are ordered by decreasing preference)
            if not self.types:
                # set a default span type
                span_type = DBPEDIA_ENT
            else:
                # takes the first one that matches
                for t in self.types:
                    if t in ent["@types"]:
                        span_type = t
                        break

            # if ent_kb_id:
            #    span = doc.char_span(start_ch, end_ch, span_type, ent_kb_id)
            span = doc.char_span(start_ch, end_ch, span_type)

            if not span:
                # something strange like "something@bbc.co.uk" where the match is only part of a SpaCy token
                # 1. find the token to split
                # tokens also wider than start_ch, end_ch
                tokens_to_split = [
                    t for t in doc if t.idx >= start_ch or t.idx + len(t) <= end_ch
                ]
                span = doc.char_span(
                    min(t.idx for t in tokens_to_split),
                    max(t.idx + len(t) for t in tokens_to_split),
                )

            span._.dbpedia_raw_result = ent

            ents_data.append(span)

        # try to add results to doc.ents
        try:
            # NB. Order is important here for precedence of infra entities over other entities
            doc.ents = list(doc.ents) + ents_data
        except Exception as e:
            logger.trace(str(e))
            if self.overwrite_ents:
                # overwrite ok
                doc.spans["ents_original"] = doc.ents
                try:
                    doc.ents = ents_data
                except (
                    ValueError
                ):  # if there are overlapping spans in the dbpedia_spotlight entities
                    doc.ents = spacy.util.filter_spans(ents_data)
            # else don't overwrite

        doc.spans[self.span_group] = ents_data

        return doc


def create(language_code, nlp=None):
    """Creates an instance of a Language with the DBpedia EntityLinker pipeline stage.
    If the parameter `nlp` is None, it will return a blank language with the EntityLinker.
    If the parameter `nlp` is an existing Language, it simply adds the EntityLinker pipeline stage (equivalent to `nlp.add`)
    """
    if not nlp:
        nlp = spacy.blank(language_code)
    nlp.add_pipe("dbpedia_spotlight")
    return nlp
