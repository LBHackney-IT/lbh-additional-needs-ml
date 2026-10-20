from src.common.schemas import GatewayNote
from src.services.note_preprocessor import NotePreprocessor


# -- Helpers ---
def _people(record):
    return [member.model_dump() for member in record.household_members]


# -- Category prefixing ---
def test_prepare_prefixes_the_note_category():
    records = NotePreprocessor().preprocess_notes(
        [
            GatewayNote(
                note_id="note-1",
                note_category="Repairs",
                text="Leak reported",
                tenure_id="tenure-1",
                household_members=[],
            )
        ]
    )

    assert records[0].prefixed_text == "[Category: Repairs] Leak reported"


def test_prepare_skips_an_empty_category_prefix():
    records = NotePreprocessor().preprocess_notes(
        [
            GatewayNote(
                note_id="note-1",
                note_category="  ",
                text="Leak reported",
                tenure_id="tenure-1",
                household_members=[],
            )
        ]
    )

    assert records[0].prefixed_text == "Leak reported"


# -- Acronym Expansion ---
def test_prepare_keeps_original_text():
    # Note - placeholde until acronym expansion is implemented
    records = NotePreprocessor().preprocess_notes(
        [
            {
                "note_id": "note-1",
                "note_content": "ASB reported",
                "tenure_id": None,
                "household_members": None,
            }
        ]
    )

    assert records[0]["text"] == "ASB reported"
    assert records[0]["tenure_ids"] == []
    assert records[0]["household_members"] == []
    assert NotePreprocessor.expand_acronyms("ASB reported") == "ASB reported"


# -- Source Dataset ---
def test_prepare_adds_the_configured_source():
    records = NotePreprocessor(source="repairs_hub").preprocess_notes(
        [
            GatewayNote(
                note_id="note-1",
                text="Repair note",
                tenure_id="tenure-1",
                household_members=[],
            )
        ]
    )

    assert records[0].source == "repairs_hub"


def test_prepare_groups_note_tenure_rows_into_memberships():
    rows = [
        GatewayNote(
            note_id="note-1",
            text="Note text",
            note_date="2026-10-01 09:30:00",
            tenure_id="tenure-1",
            household_members=[
                {
                    "id": "person-1",
                    "type": "person",
                    "fullName": "Ada",
                    "dateOfBirth": "1980-01-01",
                    "personTenureType": "Tenant",
                    "isResponsible": True,
                }
            ],
        ),
        GatewayNote(
            note_id="note-1",
            text="Note text",
            note_date="2026-10-01 09:30:00",
            tenure_id="tenure-2",
            household_members=[
                {
                    "id": "person-1",
                    "personTenureType": "HouseholdMember",
                    "isResponsible": False,
                },
                {"id": "person-2", "fullName": "Bea"},
            ],
        ),
        GatewayNote(
            note_id="note-2",
            text="Second note",
            note_date="2026-10-02 10:00:00",
            tenure_id="tenure-3",
            household_members=[],
        ),
    ]

    records = NotePreprocessor().preprocess_notes(rows)

    assert [record.id for record in records] == ["note-1", "note-2"]
    assert records[0].prefixed_text == "Note text"
    assert records[0].expanded_text == "Note text"
    assert records[0].source == "mmh"
    assert records[0].date == "2026-10-01 09:30:00"
    assert records[0].tenure_ids == ["tenure-1", "tenure-2"]
    assert _people(records[0]) == [
        {
            "id": "person-1",
            "type": "person",
            "fullName": "Ada",
            "dateOfBirth": "1980-01-01",
            "memberships": [
                {
                    "tenure_id": "tenure-1",
                    "personTenureType": "Tenant",
                    "isResponsible": True,
                },
                {
                    "tenure_id": "tenure-2",
                    "personTenureType": "HouseholdMember",
                    "isResponsible": False,
                },
            ],
        },
        {
            "id": "person-2",
            "type": None,
            "fullName": "Bea",
            "dateOfBirth": None,
            "memberships": [
                {
                    "tenure_id": "tenure-2",
                    "personTenureType": None,
                    "isResponsible": None,
                }
            ],
        },
    ]
    assert _people(records[1]) == []


def test_prepare_skips_rows_without_a_note_id():
    records = NotePreprocessor().preprocess_notes(
        [
            GatewayNote(note_id="", text="missing", tenure_id="tenure-1"),
            GatewayNote(note_id="", text="also missing", tenure_id="tenure-2"),
            GatewayNote(
                note_id="note-1",
                text="kept",
                note_date=None,
                tenure_id="tenure-3",
                household_members=[],
            ),
        ]
    )

    assert [record.id for record in records] == ["note-1"]
    assert records[0].prefixed_text == "kept"


def test_prepare_does_not_repeat_a_tenancy_or_its_membership():
    member = {
        "id": "person-1",
        "fullName": "Ada",
        "personTenureType": "Tenant",
        "isResponsible": True,
    }
    records = NotePreprocessor().preprocess_notes(
        [
            GatewayNote(
                note_id="note-1",
                text="Note text",
                tenure_id="tenure-1",
                household_members=[member],
            ),
            GatewayNote(
                note_id="note-1",
                text="Note text",
                tenure_id="tenure-1",
                household_members=[member, {"personTenureType": "Tenant"}],
            ),
        ]
    )

    assert records[0].tenure_ids == ["tenure-1"]
    assert _people(records[0]) == [
        {
            "id": "person-1",
            "type": None,
            "fullName": "Ada",
            "dateOfBirth": None,
            "memberships": [
                {
                    "tenure_id": "tenure-1",
                    "personTenureType": "Tenant",
                    "isResponsible": True,
                }
            ],
        }
    ]


def test_prepare_fills_blank_person_fields_from_a_later_tenancy():
    records = NotePreprocessor().preprocess_notes(
        [
            GatewayNote(
                note_id="note-1",
                text="Note text",
                tenure_id="tenure-1",
                household_members=[{"id": "person-1", "personTenureType": "Tenant"}],
            ),
            GatewayNote(
                note_id="note-1",
                text="Note text",
                tenure_id="tenure-2",
                household_members=[
                    {
                        "id": "person-1",
                        "type": "person",
                        "fullName": "Ada",
                        "dateOfBirth": "1980-01-01",
                        "personTenureType": "HouseholdMember",
                    }
                ],
            ),
        ]
    )

    person = records[0].household_members[0]
    assert person.fullName == "Ada"
    assert person.type == "person"
    assert person.dateOfBirth == "1980-01-01"
    assert [membership.tenure_id for membership in person.memberships] == [
        "tenure-1",
        "tenure-2",
    ]
