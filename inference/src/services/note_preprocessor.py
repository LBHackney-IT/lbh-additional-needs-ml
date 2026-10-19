"""Preprocesses notes.

- Flattening: Turns Athena note x tenure rows into one record per note. (Flattens upstream ETL fan-out)
- Acronym expansion (Not implemented yet): Models were trained on acronym expanded notes so we need to replace acronyms accordingly.
"""

from __future__ import annotations

from typing import Any


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

    def _compose_text(self, row: dict[str, Any]) -> str:
        """Build the model text, prefixing the category the officer chose."""
        content = row.get("note_content") or ""
        category = str(row.get("note_category") or "").strip()
        if category and not content.startswith("[Category:"):
            content = f"[Category: {category}] {content}"
        return self.expand_acronyms(content)

    @staticmethod
    def _new_person(member: dict[str, Any]) -> dict[str, Any]:
        """Seed a person record from the first household row we see for them."""
        return {
            "id": member.get("id"),
            "type": member.get("type"),
            "fullName": member.get("fullName"),
            "dateOfBirth": member.get("dateOfBirth"),
            "memberships": [],
        }

    @staticmethod
    def _fill_missing_fields(person: dict[str, Any], member: dict[str, Any]) -> None:
        """Backfill any person fields that were blank on an earlier row."""
        for field in ("type", "fullName", "dateOfBirth"):
            if not person.get(field) and member.get(field):
                person[field] = member[field]

    @staticmethod
    def _add_membership(
        person: dict[str, Any],
        tenure_id: str | None,
        member: dict[str, Any],
        seen_tenures: set[str],
    ) -> None:
        """Record a tenancy membership once per (person, tenure) pair."""
        if not tenure_id or tenure_id in seen_tenures:
            return
        seen_tenures.add(tenure_id)
        person["memberships"].append(
            {
                "tenure_id": tenure_id,
                "personTenureType": member.get("personTenureType"),
                "isResponsible": member.get("isResponsible"),
            }
        )

    @classmethod
    def _merge_household_members(
        cls,
        rows: list[tuple[str | None, list[dict[str, Any]]]],
    ) -> list[dict[str, Any]]:
        """Collapse a note's tenures (grouped) into list of persons with a list of memberships."""
        people: dict[str, dict[str, Any]] = {}
        seen_tenures: dict[str, set[str]] = {}

        # Get tenure x person/householdmember combos and consolidate into hms[]: person: tenures[].
        for tenure_id, members in rows:
            for member in members:
                person_id = member.get("id")
                if not person_id:
                    continue

                if person_id not in people:
                    people[person_id] = cls._new_person(member)
                    seen_tenures[person_id] = set()
                else:
                    cls._fill_missing_fields(people[person_id], member)

                cls._add_membership(people[person_id], tenure_id, member, seen_tenures[person_id])

        return list(people.values())

    def preprocess_notes(self, rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """Return one pipeline record per note with household members formatted and acronyms replaced.
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

        note_records: dict[str, dict[str, Any]] = {}
        household_rows: dict[str, list[tuple[str | None, list[dict[str, Any]]]]] = {}

        for row in rows:
            note_id = row.get("note_id")
            if not note_id:
                continue

            # Initialize dict for note ID
            note_records.setdefault(
                note_id,
                {
                    "id": note_id,
                    "source": self.source,
                    "text": self._compose_text(row),
                    "date": row.get("note_date"),
                    "tenure_ids": [],
                    "household_members": [],
                },
            )

            # Collect unique tenure IDs and householdmember dicts per note and append to relevant groups
            tenure_id = row.get("tenure_id")
            if tenure_id and tenure_id not in note_records[note_id]["tenure_ids"]:
                note_records[note_id]["tenure_ids"].append(tenure_id)
            household_rows.setdefault(note_id, []).append(
                (tenure_id, row.get("household_members") or [])
            )

        # Merge/transform household memberships across tenures
        for note_id, record in note_records.items():
            record["household_members"] = self._merge_household_members(household_rows[note_id])

        return list(note_records.values())
