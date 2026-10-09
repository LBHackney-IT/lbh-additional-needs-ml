from datetime import date, timedelta

from src.classifiers.entity_linker import DeterministicEntityLinker
from src.common.schemas import EnrichedNote, Relation, Span


def make_note(person_text, person_label, household_members, tenure_ids=("tenure-1",)):
    return EnrichedNote(
        id="test-note",
        text=f"{person_text} needs support.",
        model="test",
        needs=[
            Span(
                id="need-1", text="support", start=0, end=7, label="physical_health", confidence=0.9
            )
        ],
        persons=[
            Span(
                id="person-span",
                text=person_text,
                start=0,
                end=len(person_text),
                label=person_label,
                confidence=0.9,
            )
        ],
        relations=[
            Relation.model_validate({"from": "need-1", "to": "person-span", "confidence": 0.8})
        ],
        tenure_ids=list(tenure_ids),
        household_members=household_members,
    )


def test_fuzzy_name_match(enriched_note):
    linker = DeterministicEntityLinker(fuzzy_similarity=0.7)
    enriched_note.links = linker.link(enriched_note)

    assert len(enriched_note.links) == 1
    assert enriched_note.links[0].target_id == "person-1"
    assert enriched_note.links[0].target_type == "person"
    assert enriched_note.links[0].method == "fuzzy_name"
    assert enriched_note.links[0].confidence >= 0.7


def test_tenure_fallback_when_unlinked():
    note = make_note("Unknown person", "person_name", [{"id": "person-1", "fullName": "Jane Doe"}])

    links = DeterministicEntityLinker(fuzzy_similarity=0.95).link(note)

    assert len(links) == 1
    assert links[0].target_id == "tenure-1"
    assert links[0].target_type == "tenure"
    assert links[0].method == "tenure_fallback"


def test_role_tenant_single_match():
    note = make_note(
        "tenant",
        "person_role",
        [{"id": "person-tenant", "fullName": "Jane", "personTenureType": "tenant"}],
    )

    links = DeterministicEntityLinker().link(note)

    assert links[0].target_id == "person-tenant"
    assert links[0].method == "role_tenant"


def test_role_leaseholder_single_match():
    note = make_note(
        "leaseholder",
        "person_role",
        [{"id": "person-leaseholder", "personTenureType": "leaseholder"}],
    )

    links = DeterministicEntityLinker().link(note)

    assert links[0].target_id == "person-leaseholder"
    assert links[0].method == "role_leaseholder"


def test_ambiguous_tenant_role_falls_back_to_tenure():
    note = make_note(
        "tenant",
        "person_role",
        [
            {"id": "tenant-1", "personTenureType": "tenant"},
            {"id": "tenant-2", "personTenureType": "tenant"},
        ],
    )

    links = DeterministicEntityLinker().link(note)

    assert links[0].target_type == "tenure"
    assert links[0].method == "tenure_fallback"


def test_minor_person_is_excluded_and_falls_back_to_tenure():
    note = make_note(
        "Jamie",
        "person_name",
        [
            {
                "id": "person-child",
                "fullName": "Jamie",
                "dateOfBirth": (date.today() - timedelta(days=365 * 10)).isoformat(),
            }
        ],
    )

    links = DeterministicEntityLinker().link(note)

    assert links[0].target_type == "tenure"
    assert links[0].method == "tenure_fallback"


def test_missing_birth_date_does_not_block_person_link():
    note = make_note("Jamie", "person_name", [{"id": "person-adult", "fullName": "Jamie"}])

    links = DeterministicEntityLinker().link(note)

    assert links[0].target_id == "person-adult"
    assert links[0].method == "fuzzy_name"


def test_unknown_relation_ids_are_skipped():
    note = EnrichedNote(
        id="note-4",
        text="text",
        model="test",
        needs=[
            Span(id="need-1", text="x", start=0, end=1, label="physical_health", confidence=0.9)
        ],
        persons=[],
        relations=[
            Relation.model_validate({"from": "missing", "to": "also-missing", "confidence": 0.5})
        ],
        tenure_ids=["tenure-1"],
        household_members=[],
    )

    links = DeterministicEntityLinker().link(note)

    assert len(links) == 1
    assert links[0].target_type == "tenure"


def test_one_need_can_link_to_multiple_people():
    note = EnrichedNote(
        id="note-9",
        text="Jamie and Alex need support.",
        model="test",
        needs=[
            Span(
                id="need-1",
                text="support",
                start=21,
                end=28,
                label="physical_health",
                confidence=0.9,
            )
        ],
        persons=[
            Span(id="p1", text="Jamie", start=0, end=5, label="person_name", confidence=0.9),
            Span(id="p2", text="Alex", start=10, end=14, label="person_name", confidence=0.9),
        ],
        relations=[
            Relation.model_validate({"from": "need-1", "to": "p1", "confidence": 0.8}),
            Relation.model_validate({"from": "need-1", "to": "p2", "confidence": 0.7}),
        ],
        tenure_ids=["tenure-1"],
        household_members=[
            {"id": "person-jamie", "fullName": "Jamie"},
            {"id": "person-alex", "fullName": "Alex"},
        ],
    )

    links = DeterministicEntityLinker().link(note)

    assert {link.target_id for link in links} == {"person-jamie", "person-alex"}
