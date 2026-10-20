from datetime import UTC, datetime

from src.common.schemas import (
    DomainNote,
    EnrichedNote,
    GatewayNote,
    Link,
    Relation,
    Span,
    validate_enriched_payload,
)
from src.services.note_preprocessor import NotePreprocessor, expand_acronyms_with_offsets

# ==========================================
# TEST HELPERS
# ==========================================


def _span(span_id: str, text: str, start: int, end: int, label: str) -> Span:
    """Helper to create a Span with default confidence."""
    return Span(id=span_id, text=text, start=start, end=end, label=label, confidence=0.9)


def _cover(expanded: str, phrase: str, span_id: str, label: str) -> Span:
    """Helper to create a Span covering a specific phrase inside an expanded string."""
    start = expanded.index(phrase)
    return _span(span_id, phrase, start, start + len(phrase), label)


def _project(prefixed: str, needs: list[Span], persons: list[Span]) -> EnrichedNote:
    """Helper to run the full DomainNote -> EnrichedNote projection pipeline."""
    expanded, replacements = expand_acronyms_with_offsets(prefixed)
    domain = DomainNote(
        id="note-1",
        prefixed_text=prefixed,
        expanded_text=expanded,
        replacements=replacements,
        needs=needs,
        persons=persons,
    )
    return domain.to_enriched(model="test", pipeline_run_at=None)


def _overlaps(left: Span, right: Span) -> bool:
    """True if two spans share overlapping character indices."""
    return max(left.start, right.start) < min(left.end, right.end)


def _contains(outer: Span, inner: Span) -> bool:
    """True if the inner span is completely contained within the outer span."""
    return outer.start <= inner.start and inner.end <= outer.end


# ==========================================
# CONVERSION TESTS
# ==========================================


def test_gateway_note_converts_through_domain_to_output_text():
    gateway = GatewayNote(
        note_id="note-1",
        text="tnt visited HSc",
        note_category="Repairs",
        note_date="2026-01-01",
        tenure_id="tenure-1",
    )
    domain = NotePreprocessor().preprocess_notes([gateway])[0]

    assert domain.id == "note-1"
    assert domain.prefixed_text == "[Category: Repairs] tnt visited HSc"
    assert domain.expanded_text == "[Category: Repairs] tenant visited Hackney Service Centre"
    assert [
        domain.prefixed_text[item.prefixed_start : item.prefixed_end]
        for item in domain.replacements
    ] == ["tnt", "HSc"]

    tenant_at = domain.expanded_text.index("tenant")
    domain.persons = [
        Span(
            id="person-1",
            text="tenant",
            start=tenant_at,
            end=tenant_at + len("tenant"),
            label="person_role",
            confidence=0.9,
        )
    ]
    enriched = domain.to_enriched(
        model="span_relation",
        pipeline_run_at=datetime(2026, 1, 1, tzinfo=UTC),
    )
    payload = enriched.model_dump_for_json()
    person = payload["persons"][0]

    assert payload["text"] == domain.prefixed_text
    assert (person["start"], person["end"], person["text"]) == (tenant_at, tenant_at + 3, "tnt")
    assert payload["text"][person["start"] : person["end"]] == "tnt"
    assert payload["date"] == "2026-01-01"
    assert payload["tenure_ids"] == ["tenure-1"]

    for hidden in ("raw_text", "prefixed_text", "expanded_text", "replacements"):
        assert hidden not in payload


# ==========================================
# SPAN PROJECTION & OFFSET TESTS
# ==========================================


def test_to_enriched_projects_expanded_spans_onto_prefixed_text():
    prefixed = "tnt at HSc today"
    expanded = "tenant at Hackney Service Centre today"
    hackney_at = expanded.index("Hackney")
    note = _project(
        prefixed,
        [_span("need-1", "today", expanded.index("today"), len(expanded), "physical_health")],
        [
            _span("person-1", "tenant", 0, len("tenant"), "person_role"),
            _span("person-2", "Hackney", hackney_at, hackney_at + len("Hackney"), "person_role"),
        ],
    )

    tenant = note.persons[0]
    partial = note.persons[1]
    today = note.needs[0]

    assert (tenant.start, tenant.end, tenant.text) == (0, 3, "tnt")
    assert prefixed[tenant.start : tenant.end] == tenant.text
    assert (partial.start, partial.end, partial.text) == (7, 10, "HSc")
    assert (today.start, today.end, today.text) == (11, 16, "today")
    assert prefixed[today.start : today.end] == today.text
    assert note.text == prefixed


def test_to_enriched_collapses_overlapping_pieces_of_one_acronym():
    prefixed = "visit HSc today"
    expanded = "visit Hackney Service Centre today"
    note = _project(
        prefixed,
        [
            _cover(expanded, "Hackney", "need-1", "physical_health"),
            _cover(expanded, "Service", "need-2", "physical_health"),
        ],
        [_cover(expanded, "Centre today", "person-1", "person_role")],
    )

    projected = [*note.needs, *note.persons]
    assert [(span.start, span.end, span.text) for span in projected] == [
        (6, 9, "HSc"),
        (6, 9, "HSc"),
        (6, len(prefixed), "HSc today"),
    ]
    assert _overlaps(note.needs[0], note.needs[1])
    assert _contains(note.persons[0], note.needs[0])
    assert [span.id for span in projected] == ["need-1", "need-2", "person-1"]
    assert prefixed[note.persons[0].start : note.persons[0].end] == "HSc today"


def test_to_enriched_leaves_spans_unchanged_when_nothing_was_expanded():
    span = _span("need-1", "mobility needs", 0, 14, "physical_health")
    note = _project("Jane Doe mobility needs", [span], [])

    assert note.needs[0].text == "mobility needs"
    assert note.needs[0].start == 0
    assert note.needs[0].end == 14


def test_to_enriched_leaves_spans_unchanged_when_there_are_no_replacements():
    span = _span("need-1", "tenant", 0, 6, "person_role")
    note = _project("tenant", [span], [])

    assert note.needs[0].text == "tenant"
    assert note.needs[0].end == 6


# ==========================================
# PAYLOAD SERIALIZATION TESTS
# ==========================================


def test_enriched_note_to_json(sample_note_payload):
    note = validate_enriched_payload(sample_note_payload)
    dumped = note.model_dump_for_json()

    assert dumped["id"] == "note-1"
    assert dumped["source"] == "mmh"
    assert len(dumped["needs"]) == 1
    assert dumped["needs"][0]["label"] == "physical_health"
    assert dumped["relations"][0]["from"] == "need-1"
    assert dumped["tenure_ids"] == ["tenure-1"]

    member = dumped["household_members"][0]
    assert member["fullName"] == "Jane Doe"
    assert "personTenureType" not in member
    assert member["memberships"][0]["personTenureType"] == "tenant"


def test_enriched_note_with_links_serializes_correctly():
    note = EnrichedNote(
        id="n1",
        text="text",
        model="m1",
        needs=[
            Span(id="need-1", text="x", start=0, end=1, label="physical_health", confidence=0.9)
        ],
        persons=[],
        relations=[Relation.model_validate({"from": "need-1", "to": "p1", "confidence": 0.8})],
        links=[
            Link(
                need_id="need-1",
                target_id="person-1",
                target_type="person",
                confidence=0.95,
                method="fuzzy_name",
            )
        ],
        tenure_ids=["t1"],
        household_members=[],
        pipeline_run_at=datetime(2024, 1, 1, tzinfo=UTC),
    )

    payload = note.model_dump_for_json()
    assert payload["links"][0]["method"] == "fuzzy_name"
    assert payload["relations"][0]["from"] == "need-1"
    assert payload["text"] == "text"
    assert "raw_text" not in payload
    assert "prefixed_text" not in payload
    assert "expanded_text" not in payload

    restored = EnrichedNote.model_validate(payload)
    assert restored.links[0].target_type == "person"
    assert restored.text == "text"
