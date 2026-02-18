from __future__ import annotations

from langchain_community.utilities import GoogleSerperAPIWrapper


def build_serper() -> GoogleSerperAPIWrapper:
    # Uses SERPER_API_KEY from env automatically.
    return GoogleSerperAPIWrapper()


def run_search(search: GoogleSerperAPIWrapper, query: str) -> str:
    return search.run(query=query)
