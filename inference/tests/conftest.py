"""Shared fixtures: a household with one tenant + one leaseholder, and a note
that already has a need, a person_name span, and a relation between them.
"""

import pytest

from src.common.schemas import EnrichedNote


@pytest.fixture
def sample_household():
    return [
        {
            "id": "person-1",
            "fullName": "Jane Doe",
            "dateOfBirth": "1985-03-15",
            "memberships": [
                {"tenure_id": "tenure-1", "personTenureType": "tenant", "isResponsible": True}
            ],
        },
        {
            "id": "person-2",
            "fullName": "John Smith",
            "dateOfBirth": "1990-07-01",
            "memberships": [
                {
                    "tenure_id": "tenure-1",
                    "personTenureType": "leaseholder",
                    "isResponsible": False,
                }
            ],
        },
    ]


@pytest.fixture
def sample_note_payload(sample_household):
    return {
        "id": "note-1",
        "source": "mmh",
        "text": "[Category: Repairs] Jane Doe at the HSC has mobility needs.",
        "date": "2024-01-01",
        "model": "test_span_test_relation",
        "needs": [
            {
                "id": "need-1",
                "text": "mobility needs",
                "start": 13,
                "end": 28,
                "label": "physical_health",
                "confidence": 0.95,
            }
        ],
        "persons": [
            {
                "id": "person-span-1",
                "text": "Jane Doe",
                "start": 0,
                "end": 8,
                "label": "person_name",
                "confidence": 0.99,
            }
        ],
        "relations": [{"from": "need-1", "to": "person-span-1", "confidence": 0.88}],
        "links": [],
        "tenure_ids": ["tenure-1"],
        "household_members": sample_household,
    }


@pytest.fixture
def enriched_note(sample_note_payload):
    return EnrichedNote.model_validate(sample_note_payload)
