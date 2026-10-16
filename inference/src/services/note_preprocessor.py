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
    def _merge_household_members(
        rows: list[tuple[str | None, list[dict[str, Any]]]],
    ) -> list[dict[str, Any]]:
        """Collapse a note's tenures (grouped) into list of persons with a list of memberships."""
        people: dict[str, dict[str, Any]] = {}
        seen_tenures: dict[str, set[str]] = {}

        for tenure_id, members in rows:
            for member in members:
                person_id = member.get("id")
                if not person_id:
                    continue

                person = people.get(person_id)
                if person is None:
                    person = {
                        "id": person_id,
                        "type": member.get("type"),
                        "fullName": member.get("fullName"),
                        "dateOfBirth": member.get("dateOfBirth"),
                        "memberships": [],
                    }
                    people[person_id] = person
                    seen_tenures[person_id] = set()
                else:
                    for field in ("type", "fullName", "dateOfBirth"):
                        if not person.get(field) and member.get(field):
                            person[field] = member[field]

                if tenure_id and tenure_id not in seen_tenures[person_id]:
                    seen_tenures[person_id].add(tenure_id)
                    person["memberships"].append(
                        {
                            "tenure_id": tenure_id,
                            "personTenureType": member.get("personTenureType"),
                            "isResponsible": member.get("isResponsible"),
                        }
                    )
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
