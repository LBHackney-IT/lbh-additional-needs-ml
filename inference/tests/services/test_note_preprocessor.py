from src.services.note_preprocessor import NotePreprocessor


def test_prepare_groups_note_tenure_rows_into_memberships():
    rows = [
        {
            "note_id": "note-1",
            "note_content": "Note text",
            "note_date": "2026-10-01 09:30:00",
            "tenure_id": "tenure-1",
            "household_members": [
                {
                    "id": "person-1",
                    "type": "person",
                    "fullName": "Ada",
                    "dateOfBirth": "1980-01-01",
                    "personTenureType": "Tenant",
                    "isResponsible": True,
                }
            ],
        },
        {
            "note_id": "note-1",
            "note_content": "Note text",
            "note_date": "2026-10-01 09:30:00",
            "tenure_id": "tenure-2",
            "household_members": [
                {
                    "id": "person-1",
                    "personTenureType": "HouseholdMember",
                    "isResponsible": False,
                },
                {"id": "person-2", "fullName": "Bea"},
            ],
        },
        {
            "note_id": "note-2",
            "note_content": "Second note",
            "note_date": "2026-10-02 10:00:00",
            "tenure_id": "tenure-3",
            "household_members": [],
        },
    ]

    records = NotePreprocessor().prepare(rows)

    assert [record["id"] for record in records] == ["note-1", "note-2"]
    assert records[0]["text"] == "Note text"
    assert records[0]["date"] == "2026-10-01 09:30:00"
    assert records[0]["tenure_ids"] == ["tenure-1", "tenure-2"]
    assert records[0]["household_members"] == [
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
    assert records[1]["household_members"] == []
