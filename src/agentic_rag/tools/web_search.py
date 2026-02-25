"""Serper web-search tool wrapper.

Usage example::

    search = build_serper(serper_api_key="your_key")
    results = run_search(search, "latest FDA approval for insulin pumps")
"""
from __future__ import annotations

import logging

from langchain_community.utilities import GoogleSerperAPIWrapper

from ..exceptions import SearchError

logger = logging.getLogger(__name__)


def build_serper(serper_api_key: str) -> GoogleSerperAPIWrapper:
    """Instantiate a ``GoogleSerperAPIWrapper`` with an explicit API key.

    The key is passed explicitly rather than relying on ``os.environ``,
    because pydantic-settings reads ``.env`` into the Settings model only
    and does NOT populate ``os.environ``.

    Parameters
    ----------
    serper_api_key:
        Your Google Serper API key (from ``Settings.SERPER_API_KEY``).

    Returns
    -------
    GoogleSerperAPIWrapper
    """
    logger.debug("Initialising Serper search wrapper.")
    return GoogleSerperAPIWrapper(serper_api_key=serper_api_key)


def run_search(search: GoogleSerperAPIWrapper, query: str) -> str:
    """Execute a web search and return the result as a plain string.

    Parameters
    ----------
    search:
        A ``GoogleSerperAPIWrapper`` instance (from ``build_serper()``).
    query:
        The search query string.

    Returns
    -------
    str
        Raw search-result text returned by the Serper API.
    """
    logger.debug("Serper search | query=%r", query)
    try:
        return search.run(query=query)
    except Exception as exc:
        raise SearchError(f"Serper search failed | query={query!r}") from exc
