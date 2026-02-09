#  Copyright (c) 2024-2026, RTE (https://www.rte-france.com)
#  See AUTHORS.txt
#  SPDX-License-Identifier: MPL-2.0
#  This file is part of BERTrend.

import base64
import functools
import re
import time
from urllib.parse import urlsplit

import requests
from googlenewsdecoder import new_decoderv1
from loguru import logger

# Ref: https://stackoverflow.com/a/59023463/

_ENCODED_URL_PREFIX = "https://news.google.com/rss/articles/"
_ENCODED_URL_RE = re.compile(
    rf"^{re.escape(_ENCODED_URL_PREFIX)}(?P<encoded_url>[^?]+)"
)
_DECODED_URL_RE = re.compile(rb'^\x08\x13".+?(?P<primary_url>http[^\xd2]+)\xd2\x01')


def wait(secs):
    """wait decorator"""

    def decorator(func):
        def wrapper(*args, **kwargs):
            time.sleep(secs)
            return func(*args, **kwargs)

        return wrapper

    return decorator


def wait_if_seen_url(secs):
    """wait decorator based on URL cache: only waits max to secs for websites already seen"""

    def decorator(func):
        cache = {}

        def wrapper(*args, **kwargs):
            url = kwargs.get("url")
            if url is None:
                return func(*args, **kwargs)
            else:
                base_url = urlsplit(url).netloc
                last_call = cache.get(base_url)
                current_call = round(time.time() * 1000)
                if last_call is not None:
                    # sleep if recent call
                    delta = (current_call - last_call) / 1000
                    if delta < secs:
                        time.sleep(secs - delta)
                # update cache
                cache[base_url] = current_call
                return func(*args, **kwargs)

        return wrapper

    return decorator


def _fetch_decoded_batch_execute(id):
    s = (
        '[[["Fbv4je","[\\"garturlreq\\",[[\\"en-US\\",\\"US\\",[\\"FINANCE_TOP_INDICES\\",\\"WEB_TEST_1_0_0\\"],'
        'null,null,1,1,\\"US:en\\",null,180,null,null,null,null,null,0,null,null,[1608992183,723341000]],'
        '\\"en-US\\",\\"US\\",1,[2,3,4,8],1,0,\\"655000234\\",0,0,null,0],\\"'
        + id
        + '\\"]",null,"generic"]]]'
    )

    headers = {
        "Content-Type": "application/x-www-form-urlencoded;charset=utf-8",
        "Referer": "https://news.google.com/",
    }

    with requests.post(
        "https://news.google.com/_/DotsSplashUi/data/batchexecute?rpcids=Fbv4je",
        headers=headers,
        data={"f.req": s},
    ) as response:
        if response.status_code != 200:
            logger.error("Failed to fetch data from Google.")
            raise Exception("Failed to fetch data from Google.")

        text = response.text
    header = '[\\"garturlres\\",\\"'
    footer = '\\",'
    if header not in text:
        raise Exception(f"Header not found in response: {text}")
    start = text.split(header, 1)[1]
    if footer not in start:
        raise Exception("Footer not found in response.")
    url = start.split(footer, 1)[0]
    return url


@functools.lru_cache(2048)
def _decode_google_news_url_v2(source_url):
    """New way of decoding Gnews URL (from August 2024), the URL are not simply encoded in Base64, need to obtain
    the redirection URL from server-side"""
    # NB. Seems not to work since mid-september 2024, shall be replaced by new_decoderv1 for more recent URLs
    url = requests.utils.urlparse(source_url)
    path = url.path.split("/")
    if url.hostname == "news.google.com" and len(path) > 1 and path[-2] == "articles":
        base64_str = path[-1]
        decoded_bytes = base64.urlsafe_b64decode(base64_str + "==")
        decoded_str = decoded_bytes.decode("latin1")

        prefix = b"\x08\x13\x22".decode("latin1")
        if decoded_str.startswith(prefix):
            decoded_str = decoded_str[len(prefix) :]

        suffix = b"\xd2\x01\x00".decode("latin1")
        if decoded_str.endswith(suffix):
            decoded_str = decoded_str[: -len(suffix)]

        bytes_array = bytearray(decoded_str, "latin1")
        length = bytes_array[0]
        if length >= 0x80:
            decoded_str = decoded_str[2 : length + 1]
        else:
            decoded_str = decoded_str[1 : length + 1]

        if decoded_str.startswith("AU_yqL"):
            return _fetch_decoded_batch_execute(base64_str)

        return decoded_str
    else:
        return source_url


@functools.lru_cache(2048)
def _decode_google_news_url(url: str) -> str:
    """Decode encoded Google News entry URLs. (until August 2024)"""
    match = _ENCODED_URL_RE.match(url)
    encoded_text = match.groupdict()["encoded_url"]  # type: ignore
    encoded_text += (
        "==="  # Fix incorrect padding. Ref: https://stackoverflow.com/a/49459036/
    )
    decoded_text = base64.urlsafe_b64decode(encoded_text)

    match = _DECODED_URL_RE.match(decoded_text)
    primary_url = match.groupdict()["primary_url"]  # type: ignore
    primary_url = primary_url.decode()
    return primary_url


def decode_google_news_url(
    url: str,
) -> str:  # Not cached because not all Google News URLs are encoded.
    """Return Google News entry URLs after decoding their encoding as applicable."""
    time.sleep(0.5)  # to avoid being banned

    decoded_content = new_decoderv1(url) if url.startswith(_ENCODED_URL_PREFIX) else url
    if decoded_content["status"]:
        # decoding was ok
        return decoded_content["decoded_url"]
    else:
        logger.error(f"Error decoding Google News URL: {url}. {decoded_content}")
        raise Exception(f"Error decoding Google News URL: {url}")
