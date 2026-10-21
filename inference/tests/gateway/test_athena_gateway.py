import json

import pytest

from src.gateway.athena_gateway import AthenaRecordSource


# --- Fixtures / Helpers ---
class FakePaginator:
    def __init__(self, pages):
        self.pages = pages

    def paginate(self, **_kwargs):
        return self.pages


class FakeAthenaClient:
    def __init__(self, pages, state="SUCCEEDED"):
        self.pages = pages
        self.state = state
        self.started = None

    def start_query_execution(self, **kwargs):
        self.started = kwargs
        return {"QueryExecutionId": "query-1"}

    def get_query_execution(self, **_kwargs):
        return {
            "QueryExecution": {
                "Status": {
                    "State": self.state,
                    "StateChangeReason": "test failure" if self.state == "FAILED" else "",
                }
            }
        }

    def get_paginator(self, name):
        assert name == "get_query_results"
        return FakePaginator(self.pages)


def query_page(rows):
    return {
        "ResultSet": {
            "ResultSetMetadata": {
                "ColumnInfo": [
                    {"Name": "note_id"},
                    {"Name": "tenure_id"},
                    {"Name": "note_content"},
                    {"Name": "note_category"},
                    {"Name": "note_date"},
                    {"Name": "household_members"},
                ]
            },
            "Rows": rows,
        }
    }


def cell_row(*values):
    return {"Data": [{"VarCharValue": value} if value is not None else {} for value in values]}


# --- Athena Client Handling ---
def test_fetch_records_returns_data_in_expected_format():
    pages = [
        query_page(
            [
                cell_row(
                    "note_id",
                    "tenure_id",
                    "note_content",
                    "note_category",
                    "note_date",
                    "household_members",
                ),
                cell_row(
                    "note-1",
                    "tenure-1",
                    "Note text",
                    "tenureManagement",
                    "2026-10-01 09:30:00",
                    json.dumps(
                        [
                            {
                                "id": "person-1",
                                "type": "person",
                                "fullName": "Ada",
                                "dateOfBirth": "1980-01-01",
                                "personTenureType": "Tenant",
                                "isResponsible": True,
                            }
                        ]
                    ),
                ),
                cell_row(
                    "note-1",
                    "tenure-2",
                    "Note text",
                    "tenureManagement",
                    "2026-10-01 09:30:00",
                    json.dumps(
                        [
                            {
                                "id": "person-1",
                                "personTenureType": "HouseholdMember",
                                "isResponsible": False,
                            },
                            {"id": "person-2", "fullName": "Bea"},
                        ]
                    ),
                ),
            ]
        ),
        query_page(
            [
                cell_row(
                    "note-2",
                    "tenure-3",
                    "Second note",
                    "tenureManagement",
                    "2026-10-02 10:00:00",
                    None,
                ),
            ]
        ),
    ]
    client = FakeAthenaClient(pages)
    class_under_test = AthenaRecordSource(
        client=client,
        poll_interval_seconds=0,
        timeout_seconds=1,
    )

    records = class_under_test.fetch_records()

    assert client.started["WorkGroup"] == "housing"
    assert client.started["QueryExecutionContext"] == {"Database": "prod__housing-refined-zone"}

    assert [record.note_id for record in records] == ["note-1", "note-1", "note-2"]
    assert records[0].text == "Note text"
    assert records[0].note_date == "2026-10-01 09:30:00"
    assert [record.tenure_id for record in records] == ["tenure-1", "tenure-2", "tenure-3"]
    assert [member.model_dump() for member in records[0].household_members] == [
        {
            "id": "person-1",
            "type": "person",
            "fullName": "Ada",
            "dateOfBirth": "1980-01-01",
            "personTenureType": "Tenant",
            "isResponsible": True,
        }
    ]
    assert records[1].household_members[1].fullName == "Bea"
    assert records[2].household_members == []


def test_fetch_records_raises_for_failed_query():
    client = FakeAthenaClient([], state="FAILED")
    source = AthenaRecordSource(client=client, poll_interval_seconds=0, timeout_seconds=1)

    with pytest.raises(RuntimeError, match="Athena query failed"):
        source.fetch_records()


# --- Database / Table Params ---
def test_query_quotes_configured_identifiers():
    source = AthenaRecordSource(
        database="housing-refined-zone",
        table="v_repairs_jds_reshaped",
        client=object(),
    )

    assert 'FROM "housing-refined-zone"."v_repairs_jds_reshaped"' in source.query


@pytest.mark.parametrize(
    ("database", "table"),
    [
        ("housing;DROP TABLE notes", "notes_table"),
        ("housing-refined-zone", 'notes_table" WHERE TRUE; --'),
    ],
)
def test_query_rejects_unsafe_identifiers(database, table):
    source = AthenaRecordSource(database=database, table=table, client=object())

    with pytest.raises(ValueError, match="Invalid Athena identifier"):
        _ = source.query


# --- Where Param ---
def test_query_uses_the_default_mmh_filter():
    source = AthenaRecordSource(client=object())

    for flag in (
        "flag_no_matching_target",
        "flag_note_date_parse_failed",
        "flag_is_organisation",
        "flag_inactive",
    ):
        assert f'"{flag}" = FALSE' in source.query


def test_query_uses_the_provided_where_clause():
    source = AthenaRecordSource(
        where="note_date >= DATE '2024-01-01' AND note_date < DATE '2025-01-01'",
        client=object(),
    )

    assert "WHERE note_date >= DATE '2024-01-01' AND note_date < DATE '2025-01-01'" in source.query
    assert "flag_no_matching_target" not in source.query


def test_query_accepts_a_where_keyword():
    source = AthenaRecordSource(where="WHERE flag_inactive = FALSE", client=object())

    assert "WHERE flag_inactive = FALSE" in source.query
    assert source.query.count("WHERE") == 1


def test_query_omits_where_when_the_clause_is_empty():
    source = AthenaRecordSource(where="", client=object())

    assert "WHERE" not in source.query


def test_query_rejects_a_second_statement():
    source = AthenaRecordSource(where="flag_inactive = FALSE; DROP TABLE notes", client=object())

    with pytest.raises(ValueError, match="WHERE clause"):
        _ = source.query
