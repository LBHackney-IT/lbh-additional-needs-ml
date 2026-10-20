"""Pydantic v2 schemas for the three note layers.

- GatewayNote is one database row.
- DomainNote is the in-pipeline note, with the category prefix and the expanded string the classifiers read.
- EnrichedNote is the JSON output, whose ``text`` is the category-prefixed note.
"""

from __future__ import annotations

from datetime import datetime
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator

# ==========================================
# SHARED SCHEMAS
# ==========================================


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


class TenureMembership(BaseModel):
    """A person's role in one tenancy."""

    tenure_id: str
    personTenureType: str | None = None
    isResponsible: bool | None = None


class HouseholdMember(BaseModel):
    """A household person, with one membership per matched tenancy.

    Person fields map to MMH's household member identity. Role fields live on
    TenureMembership because they describe the person in a specific tenancy.
    https://github.com/LBHackney-IT/tenure-shared/blob/main/Hackney.Shared.Tenure/Domain/HouseholdMembers.cs
    """

    id: str
    type: str | None = None
    fullName: str | None = None
    dateOfBirth: str | None = None
    memberships: list[TenureMembership] = Field(default_factory=list)


class AcronymReplacement(BaseModel):
    """One abbreviation substitution, in prefixed-text and expanded-text coordinates."""

    prefixed_start: int
    prefixed_end: int
    expanded_start: int
    expanded_end: int


# ==========================================
# LAYER 1: DATA ACCESS (GATEWAY)
# ==========================================


class GatewayNote(BaseModel):
    """One note-tenure row as read from the database."""

    note_id: str
    text: str = ""
    note_category: str | None = None
    note_date: str | None = None
    tenure_id: str | None = None
    household_members: list[dict[str, Any]] = Field(default_factory=list)

    @field_validator("text", mode="before")
    @classmethod
    def _blank_text(cls, value: Any) -> str:
        return value or ""

    @field_validator("household_members", mode="before")
    @classmethod
    def _blank_members(cls, value: Any) -> list[dict[str, Any]]:
        return value or []


# ==========================================
# LAYER 2: BUSINESS LOGIC (DOMAIN)
# ==========================================


class DomainNote(BaseModel):
    """In-pipeline note.

    ``prefixed_text`` is the note after the category prefix, with abbreviations
    still as the officer wrote them. ``expanded_text`` is what the classifiers
    see (with acronym expansions).
    """

    id: str
    source: str = "mmh"
    prefixed_text: str
    expanded_text: str
    replacements: list[AcronymReplacement] = Field(default_factory=list)
    date: str | None = None
    tenure_ids: list[str] = Field(default_factory=list)
    household_members: list[HouseholdMember] = Field(default_factory=list)
    # ML Outputs - Populated progressively during the pipeline execution
    needs: list[Span] = Field(default_factory=list)
    persons: list[Span] = Field(default_factory=list)
    relations: list[Relation] = Field(default_factory=list)
    links: list[Link] = Field(default_factory=list)

    def to_enriched(
        self,
        *,
        model: str,
        pipeline_run_at: datetime | None,
    ) -> EnrichedNote:
        """Return the output note, with spans projected onto ``prefixed_text``."""
        return EnrichedNote(
            id=self.id,
            source=self.source,
            text=self.prefixed_text,
            date=self.date,
            model=model,
            # Project spans back to the officer's original wording (before acronym expansion)
            needs=[self._project_span(span) for span in self.needs],
            persons=[self._project_span(span) for span in self.persons],
            relations=self.relations,
            links=self.links,
            tenure_ids=self.tenure_ids,
            household_members=self.household_members,
            pipeline_run_at=pipeline_run_at,
        )

    def _project_span(self, span: Span) -> Span:
        """Return a span whose offsets and text are a slice of ``prefixed_text``."""
        if not self.replacements:
            return span
        start = self._map_offset(span.start, is_end=False)
        end = self._map_offset(span.end, is_end=True)

        # Clamp boundaries safely within the string length
        start = min(max(start, 0), len(self.prefixed_text))
        end = min(max(end, start), len(self.prefixed_text))

        return span.model_copy(
            update={"start": start, "end": end, "text": self.prefixed_text[start:end]}
        )

    def _map_offset(self, offset: int, *, is_end: bool) -> int:
        """Map an expanded-text offset back to prefixed_text coordinates.

        Because replacements are strictly ordered left-to-right, we calculate how
        much the text "grew" before reaching our target offset, and subtract it.
        """
        shift = 0
        for replacement in self.replacements:
            # Ignore replacements after this specific span. Won't affect position.
            if offset < replacement.expanded_start:
                break

            # Handle partial word highlights - expand to full acronym
            if offset < replacement.expanded_end:
                return replacement.prefixed_end if is_end else replacement.prefixed_start

            # Add each 'growth' from preceding spans
            growth = (replacement.expanded_end - replacement.expanded_start) - (
                replacement.prefixed_end - replacement.prefixed_start
            )
            shift += growth

        # Subtract shift from every preceding span
        return offset - shift


# ==========================================
# LAYER 3: ENRICHED OUTPUT
# ==========================================


class EnrichedNote(BaseModel):
    """Fully linked per-note output."""

    model_config = ConfigDict(populate_by_name=True)

    id: str
    source: str = "mmh"
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
