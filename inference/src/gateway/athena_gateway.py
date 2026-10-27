"""Athena source for production Additional Needs records."""

from __future__ import annotations

import json
import logging
import re
import time
from collections.abc import Iterator
from typing import Any

import boto3

from src.common.schemas import GatewayNote

# --- Constants ---
DEFAULT_DATABASE = "prod__housing-refined-zone"
DEFAULT_TABLE = "additional_needs_notes_reshaped"
DEFAULT_WORKGROUP = "housing"
DEFAULT_REGION = "eu-west-2"
DEFAULT_WHERE = (
    '"flag_no_matching_target" = FALSE'
    ' AND "flag_note_date_parse_failed" = FALSE'
    ' AND "flag_is_organisation" = FALSE'
    ' AND "flag_inactive" = FALSE'
)
ATHENA_IDENTIFIER_PATTERN = re.compile(r"^[A-Za-z0-9_-]+$")


# --- Helpers ---
def _parse_household_members(value: Any) -> list[dict[str, Any]]:
    """Convert Athena's JSON/string household members value into member dictionaries.

    Unreadable JSON is treated as an empty household so one bad cell does not
    discard the rest of the note.
    """
    if value is None or value == "":
        return []
    try:
        members = json.loads(value) if isinstance(value, str) else value
    except json.JSONDecodeError:
        logging.getLogger(__name__).warning("Skipping unreadable household_members value")
        return []
    if not isinstance(members, list):
        raise ValueError("household_members must be a JSON array")
    return [member for member in members if isinstance(member, dict)]


def _validate_identifier(value: str, label: str) -> str:
    """Return a safe Athena identifier or raise before constructing SQL."""
    if not ATHENA_IDENTIFIER_PATTERN.fullmatch(value):
        raise ValueError(f"Invalid Athena identifier ({label}): {value!r}")
    return value


def _normalize_where(clause: str) -> str:
    """Return a single SQL predicate, or an empty string when no filter is set."""
    predicate = " ".join(clause.split())
    if re.match(r"(?i)where\b", predicate):
        predicate = re.sub(r"(?i)^where\s*", "", predicate, count=1).strip()
    if ";" in predicate:
        raise ValueError("WHERE clause must not contain ';'")
    return predicate


class AthenaRecordSource:
    """Fetch production note-tenure rows through Athena."""

    def __init__(
        self,
        database: str = DEFAULT_DATABASE,
        table: str = DEFAULT_TABLE,
        workgroup: str = DEFAULT_WORKGROUP,
        region: str = DEFAULT_REGION,
        poll_interval_seconds: float = 1.0,
        timeout_seconds: int = 3600,
        where: str = DEFAULT_WHERE,
        logger: logging.Logger | None = None,
        client: Any | None = None,
    ):
        self.database = database
        self.table = table
        self.workgroup = workgroup
        self.region = region
        self.where = where
        self.poll_interval_seconds = poll_interval_seconds
        self.timeout_seconds = timeout_seconds
        self.logger = logger
        self.client = (
            client if client is not None else boto3.client("athena", region_name=self.region)
        )

    # --- Conversion Helpers ---
    @staticmethod
    def _rows_to_dicts(
        rows: list[dict[str, Any]], column_names: list[str]
    ) -> Iterator[dict[str, Any]]:
        for row in rows:
            values = [cell.get("VarCharValue") for cell in row.get("Data", [])]
            values.extend([None] * (len(column_names) - len(values)))
            yield dict(zip(column_names, values, strict=True))

    @staticmethod
    def _parse_row(row: dict[str, Any]) -> GatewayNote | None:
        """Return one note-tenure row, with household_members decoded from JSON."""
        note_id = row.get("note_id")
        if not note_id:
            return None
        category = row.get("note_category")
        return GatewayNote(
            note_id=note_id,
            text=row.get("note_content") or "",
            note_category=category or None,
            note_date=row.get("note_date"),
            tenure_id=row.get("tenure_id"),
            household_members=_parse_household_members(row.get("household_members")),
        )

    # --- Query Helpers ---
    @property
    def query(self) -> str:
        """Return the Athena query, with rows ordered by note_date for a date resume."""
        # Load params
        database = _validate_identifier(self.database, "database")
        table = _validate_identifier(self.table, "table")
        predicate = _normalize_where(self.where)
        where_clause = f"\n            WHERE {predicate}" if predicate else ""

        return f"""
            SELECT
                note_id,
                note_category,
                note_content,
                note_date,
                tenure_id,
                household_members
            FROM "{database}"."{table}"{where_clause}
            ORDER BY note_date ASC NULLS FIRST, note_id;
        """.strip()

    def _start_query(self) -> str:
        """Starts the athena query"""
        response = self.client.start_query_execution(
            QueryString=self.query,
            QueryExecutionContext={"Database": self.database},
            WorkGroup=self.workgroup,
        )
        execution_id = response["QueryExecutionId"]
        if self.logger:
            self.logger.info(
                "Started Athena query %s in workgroup %s", execution_id, self.workgroup
            )
        return execution_id

    def _wait_for_query(self, execution_id: str) -> None:
        """Polls athena until it succeeds, fails/cancels, or times out."""
        deadline = time.monotonic() + self.timeout_seconds
        while True:
            status = self.client.get_query_execution(QueryExecutionId=execution_id)[
                "QueryExecution"
            ]["Status"]
            state = status["State"]
            if state == "SUCCEEDED":
                return
            if state in {"FAILED", "CANCELLED"}:
                reason = status.get("StateChangeReason", "unknown reason")
                raise RuntimeError(f"Athena query {state.lower()}: {reason}")
            if time.monotonic() >= deadline:
                raise TimeoutError(f"Athena query timed out: {execution_id}")
            time.sleep(self.poll_interval_seconds)

    def _read_rows(self, execution_id: str) -> list[dict[str, Any]]:
        """Pages through each page and converts to dictionaries"""
        paginator = self.client.get_paginator("get_query_results")
        pages = paginator.paginate(QueryExecutionId=execution_id)
        rows: list[dict[str, Any]] = []
        column_names: list[str] | None = None
        for page in pages:
            if column_names is None:
                column_names = [
                    column["Name"]
                    for column in page["ResultSet"]["ResultSetMetadata"]["ColumnInfo"]
                ]
                page_rows = page["ResultSet"]["Rows"][1:]
            else:
                page_rows = page["ResultSet"]["Rows"]
            rows.extend(self._rows_to_dicts(page_rows, column_names))
        return rows

    # --- Public ---
    def fetch_records(self) -> list[GatewayNote]:
        """Run the query and return data as a list of dictionaries."""
        execution_id = self._start_query()
        self._wait_for_query(execution_id)
        records = [
            record
            for row in self._read_rows(execution_id)
            if (record := self._parse_row(row)) is not None
        ]
        if self.logger:
            self.logger.info("Athena returned %d note-tenure rows", len(records))
        return records
