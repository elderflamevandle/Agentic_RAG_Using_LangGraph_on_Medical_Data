from __future__ import annotations

from pydantic import BaseModel, Field
from typing import Literal

class RouteDecision(BaseModel):
    route: Literal["Retrieve_QnA", "Retrieve_Device", "Web_Search"] = Field(
        ..., description="Where to retrieve information from."
    )

class RelevanceDecision(BaseModel):
    is_relevant: Literal["Yes", "No"] = Field(
        ..., description="Whether the provided context is relevant to the query."
    )
