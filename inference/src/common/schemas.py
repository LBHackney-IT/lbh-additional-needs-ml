"""
Pydantic v2 schemas for the output JSON.
"""

from __future__ import annotations

from datetime import datetime
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field


class Span(BaseModel):
    """A character-index entity mention (need or person)."""

    id: str
    text: str
    start: int
    end: int
    label: str
    confidence: float


class SpanPredictions(BaseModel):
    """Span predictions grouped by their role in a note."""

    needs: list[Span] = Field(default_factory=list)
    persons: list[Span] = Field(default_factory=list)


class Relation(BaseModel):
    """Need → person relation."""

    model_config = ConfigDict(populate_by_name=True)

    from_id: str = Field(alias="from")
    to: str
    confidence: float


class Link(BaseModel):
    """Resolved attribution of a need to a household person or a tenure."""

    need_id: str
    target_id: str
    target_type: Literal["person", "tenure"]
    confidence: float
    method: str


class HouseholdMember(BaseModel):
    """Maps to MMH's tenure householdMembers field."""

    # https://github.com/LBHackney-IT/tenure-shared/blob/main/Hackney.Shared.Tenure/Domain/HouseholdMembers.cs

    id: str
    householdMembersType: str | None = None
    fullName: str | None = None
    personTenureType: str | None = None
    isResponsible: bool | None = None
    dateOfBirth: str | None = None


class EnrichedNote(BaseModel):
    """Fully linked per-note output."""

    model_config = ConfigDict(populate_by_name=True)

    id: str
    text: str
    date: str | None = None
    model: str
    needs: list[Span] = Field(default_factory=list)
    persons: list[Span] = Field(default_factory=list)
    relations: list[Relation] = Field(default_factory=list)
    links: list[Link] = Field(default_factory=list)
    tenure_ids: list[str] = Field(default_factory=list)
    household_members: list[HouseholdMember] = Field(default_factory=list)
    pipeline_run_at: datetime | None = None

    def model_dump_for_json(self) -> dict[str, Any]:
        payload = self.model_dump(mode="json", by_alias=True)
        if self.pipeline_run_at:
            payload["pipeline_run_at"] = self.pipeline_run_at.isoformat()
        return payload


def validate_enriched_payload(payload: dict[str, Any]) -> EnrichedNote:
    """Validate a single enriched note dict against the schema."""
    return EnrichedNote.model_validate(payload)
