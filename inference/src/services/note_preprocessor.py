"""Preprocesses notes.

- Flattening: Turns Athena note x tenure rows into one record per note. (Flattens upstream ETL fan-out)
- Abbreviation expansion: ``expanded_text`` is for the classifiers. ``prefixed_text`` keeps the category prefix and the officer's abbreviations.
"""

from __future__ import annotations

import re
from typing import Any

from src.common.schemas import (
    AcronymReplacement,
    DomainNote,
    GatewayHouseholdMember,
    GatewayNote,
    HouseholdMember,
    TenureMembership,
)

# ==========================================
# ACRONYM EXPANSION (REGEX MATCHING)
# ==========================================

# Matching is case-insensitive.
# Longer patterns are tried first so "tnts" is not read as "tnt".
abbreviation_mapping: dict[str, str] = {
    # Hackney terms
    "hsc": "Hackney Service Centre",
    "mmh": "Manage My Home",
    "lbh": "London Borough of Hackney",
    "rcc": "Repairs Contact Centre",
    # Housing terms
    "tnts": "tenants",
    "tnt": "tenant",
    "clt": "client",
    "l/h": "leaseholder",
    "h/o": "housing officer",
    "tmo": "Tenant Management Organisation",
    "nho": "neighbourhood housing officer",
    "ahm": "area housing manager",
    "hb": "housing benefit",
    "nosp": "Notice of Seeking Possession",
    "ntq": "Notice to Quit",
    "rtb": "Right to Buy",
    "uc": "Universal Credit",
    "asb": "anti-social behaviour",
    "dv": "domestic violence",
    "da": "domestic abuse",
    # General terms
    "tel": "telephone",
    "docs": "documents",
    "appt": "appointment",
    "acct": "account",
    "bacs": "bank transfer",
    "info": "information",
    "org": "organisation",
    "sms": "text message",
    "msg": "message",
    "cctv": "camera",
    "gp": "general practitioner",
    "pls": "please",
    "recd": "received",
    "asap": "as soon as possible",
    "cert": "certificate",
    "vm": "voicemail",
}

# 1. Sort by key length descending so "tnts" matches before "tnt"
_SORTED_TERMS = sorted(abbreviation_mapping.keys(), key=len, reverse=True)

# 2. Build the regex using re.escape() and word boundaries \b dynamically
_ACRONYM_PATTERN = re.compile(
    r"\b(" + "|".join(re.escape(term) for term in _SORTED_TERMS) + r")\b",
    re.IGNORECASE,
)


def expand_acronyms_with_offsets(text: str) -> tuple[str, list[AcronymReplacement]]:
    """Expand housing acronyms and record how each replacement shifts later characters.

    Matching is case-insensitive and stops at word boundaries, so an acronym is
    not rewritten when it sits inside a longer word.

    Args:
        text: Note text, including any unexpanded acronyms.

    Returns:
        Expanded text, and one replacement per acronym in left-to-right order.
    """
    if not text:
        return text, []

    parts: list[str] = []
    replacements: list[AcronymReplacement] = []
    cursor = 0
    expanded_cursor = 0

    for match in _ACRONYM_PATTERN.finditer(text):
        start, end = match.span()
        expansion = abbreviation_mapping[match.group(0).lower()]
        unchanged = text[cursor:start]
        parts.append(unchanged)
        expanded_cursor += len(unchanged)
        expanded_start = expanded_cursor
        parts.append(expansion)
        expanded_cursor += len(expansion)
        replacements.append(
            AcronymReplacement(
                prefixed_start=start,
                prefixed_end=end,
                expanded_start=expanded_start,
                expanded_end=expanded_cursor,
            )
        )
        cursor = end

    parts.append(text[cursor:])
    return "".join(parts), replacements


# ==========================================
# NOTES PREPROCESSOR
# ==========================================


class NotePreprocessor:
    """Group note-tenure rows by note id and normalise note text for the models."""

    def __init__(self, source: str = "mmh"):
        self.source = source

    def _prefix_category(self, note: GatewayNote) -> str:
        """Build the note text, prefixing the category the officer chose."""
        content = note.text or ""
        category = str(note.note_category or "").strip()
        if category and not content.startswith("[Category:"):
            content = f"[Category: {category}] {content}"
        return content

    @staticmethod
    def _upsert_person(
        people: dict[str, HouseholdMember], member: GatewayHouseholdMember
    ) -> HouseholdMember:
        """Create a new person or backfill missing fields on an existing one."""
        if member.id not in people:
            people[member.id] = HouseholdMember(
                id=member.id,
                type=member.type,
                fullName=member.fullName,
                dateOfBirth=member.dateOfBirth,
                memberships=[],
            )
        else:
            existing = people[member.id]
            existing.type = existing.type or member.type
            existing.fullName = existing.fullName or member.fullName
            existing.dateOfBirth = existing.dateOfBirth or member.dateOfBirth

        return people[member.id]

    @staticmethod
    def _add_membership(
        person: HouseholdMember,
        tenure_id: str | None,
        member: GatewayHouseholdMember,
        seen_tenures: set[str],
    ) -> None:
        """Add a tenancy membership if it hasn't been added yet."""
        if not tenure_id or tenure_id in seen_tenures:
            return

        seen_tenures.add(tenure_id)
        person.memberships.append(
            TenureMembership(
                tenure_id=tenure_id,
                personTenureType=member.personTenureType,
                isResponsible=member.isResponsible,
            )
        )

    @classmethod
    def _merge_household_members(
        cls,
        rows: list[tuple[str | None, list[GatewayHouseholdMember]]],
    ) -> list[HouseholdMember]:
        """Collapse a note's tenures (grouped) into a list of strongly-typed HouseholdMembers."""
        people: dict[str, HouseholdMember] = {}
        seen_tenures: dict[str, set[str]] = {}

        # Get tenure x person combos and consolidate into person: tenures[].
        for tenure_id, members in rows:
            for member in members:
                if not member.id:
                    continue

                # 1. Get or update the person
                person = cls._upsert_person(people, member)

                # 2. Add the membership record
                seen_tenures.setdefault(member.id, set())
                cls._add_membership(person, tenure_id, member, seen_tenures[member.id])

        return list(people.values())

    def preprocess_notes(self, rows: list[GatewayNote | dict[str, Any]]) -> list[DomainNote]:
        """Return one domain note per note id, with household members formatted.

        ``prefixed_text`` is the note after the category prefix and before abbreviation expansion.
        ``expanded_text`` is that string with abbreviations expanded, which is what the classifiers read.
        Input (Rows):
        [
          {"note_id": "n1", "tenure_id": "t1", "household_members": [alice]},
          {"note_id": "n1", "tenure_id": "t2", "household_members": [alice, bob]},
        ]

        Output:
        {
            "note_id":  "n1",
            ...
            "tenure_ids": ["t1", "t2"],
            "household_members":         [
                {
                    "id": "alice",
                    "memberships": [
                        {"tenure_id": "t1", "personTenureType": "Tenant", ...},
                        {"tenure_id": "t2", "personTenureType": "HouseholdMember", ...},
                    ],
                },
                {"id": "bob", "memberships": [{"tenure_id": "t2", "personTenureType": "Tenant", ...}]},
            ]
        }
        """

        # 1. Group all GatewayNote rows by note_id
        grouped_rows: dict[str, list[GatewayNote]] = {}
        for raw_row in rows:
            row = (
                raw_row if isinstance(raw_row, GatewayNote) else GatewayNote.model_validate(raw_row)
            )
            if row.note_id:
                grouped_rows.setdefault(row.note_id, []).append(row)

        domain_notes: list[DomainNote] = []

        # 2. Reduce grouped rows into DomainNotes
        for note_id, note_group in grouped_rows.items():
            base_note = note_group[0]
            # Acroynm expansion
            prefixed_text = self._prefix_category(base_note)
            expanded_text, replacements = expand_acronyms_with_offsets(prefixed_text)

            # Grouping tenures and household members per note
            tenure_ids: list[str] = []
            household_data: list[tuple[str | None, list[GatewayHouseholdMember]]] = []
            for row in note_group:
                if row.tenure_id and row.tenure_id not in tenure_ids:
                    tenure_ids.append(row.tenure_id)
                household_data.append((row.tenure_id, row.household_members))

            # Convert + reshape & combine household members
            domain_notes.append(
                DomainNote(
                    id=note_id,
                    source=self.source,
                    prefixed_text=prefixed_text,
                    expanded_text=expanded_text,
                    replacements=replacements,
                    date=base_note.note_date,
                    tenure_ids=tenure_ids,
                    household_members=self._merge_household_members(household_data),
                )
            )

        return domain_notes
