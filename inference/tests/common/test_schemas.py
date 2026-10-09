from datetime import UTC, datetime

from src.common.schemas import EnrichedNote, Link, Relation, Span, validate_enriched_payload


def test_enriched_note_to_json(sample_note_payload):
    note = validate_enriched_payload(sample_note_payload)
    dumped = note.model_dump_for_json()

    assert dumped["id"] == "note-1"
    assert len(dumped["needs"]) == 1
    assert dumped["needs"][0]["label"] == "physical_health"
    assert dumped["relations"][0]["from"] == "need-1"
    assert dumped["tenure_ids"] == ["tenure-1"]


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

    restored = EnrichedNote.model_validate(payload)
    assert restored.links[0].target_type == "person"
