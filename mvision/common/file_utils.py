"""
This file is borrowed from allennlp with a few of modification
"""

from typing import Tuple
import os
import base64
import logging

import requests

from mvision.common.tqdm import Tqdm

logger = logging.getLogger(__name__)

CACHE_ROOT = os.getenv('MVISION_CACHE_ROOT', os.path.expanduser(os.path.join('~', '.mvision')))
DASET_CACHE = os.path.join(CACHE_ROOT, 'datasets')


def url_to_filename(url: str, etag: str = None) -> str:
    """
    convert a url into a filename in a  reversiable way
    :param url:
    :param etag:
    :return:
    """
    url_bytes = url.encode('utf-8')
    b64_bytes = base64.b64encode(url_bytes)
    decoded = b64_bytes.decode('utf-8')

    if etag:
        etag = etag.replace('"', '')
        return f"{decoded}.{etag}" # usage like format
    else:
        return decoded


def filename_to_url(filename: str) -> Tuple[str, str]:
    """
    recover the url from given encoded filename
    :param filename:
    :return:
    """
    try:
        decoded, etag = filename.split('.', 1)
    except ValueError:
        decoded, etag = decoded, None

    filename_bytes = decoded.encode('utf-8')
    url_bytes = base64.b16decode(filename_bytes)
    url = url_bytes.decode('utf-8')
    return url, etag







