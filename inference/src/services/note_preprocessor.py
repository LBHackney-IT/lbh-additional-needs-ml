"""Preprocesses notes.

- Flattening: Turns Athena note x tenure rows into one record per note. (Flattens upstream ETL fan-out)
- Abbreviation expansion: ``expanded_text`` is for the classifiers. ``prefixed_text`` keeps the category prefix and the officer's abbreviations.
"""

from __future__ import annotations

from src.common.schemas import (
    DomainNote,
    GatewayHouseholdMember,
    GatewayNote,
    HouseholdMember,
    TenureMembership,
)


class NotePreprocessor:
    """Group note-tenure rows by note id and normalise note text for the models."""

    def __init__(self, source: str = "mmh"):
        self.source = source

    @staticmethod
    def expand_acronyms(text: str) -> str:
        """Expand acronyms to the form used in model training.

        This currently returns the note unchanged.
        """
        return text

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

    def preprocess_notes(self, rows: list[GatewayNote]) -> list[DomainNote]:
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
        for row in rows:
            if row.note_id:
                grouped_rows.setdefault(row.note_id, []).append(row)

        domain_notes: list[DomainNote] = []

        # 2. Reduce grouped rows into DomainNotes
        for note_id, note_group in grouped_rows.items():
            base_note = note_group[0]
            # Acroynm expansion
            prefixed_text = self._prefix_category(base_note)
            expanded_text, replacements = self.expand_acronyms(prefixed_text)

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
