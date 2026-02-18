from __future__ import annotations

from typing import TypedDict, Optional, Literal


Route = Literal["Retrieve_QnA", "Retrieve_Device", "Web_Search"]


class GraphState(TypedDict, total=False):
    # Input
    query: str

    # Working fields
    route: Route
    context: str
    source: str  # human readable
    is_relevant: Literal["Yes", "No"]
    iteration_count: int

    # Output
    prompt: str
    response: str
