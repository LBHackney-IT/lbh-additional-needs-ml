"""Run the real gateway, preprocessing, model pipelines, linker, and JSON writer."""

import pytest

NOTE_TEXT = "Jane Doe has mobility needs. Alex Kim was mentioned. the tenant was informed."
NOTE_DATE = "2026-10-01 09:30:00"
PERSON_NAME = "person_name"
PERSON_ROLE = "person_role"
PHYSICAL_HEALTH = "physical_health"
MOBILITY = "mobility_mobility_physical"
HOARDING = "housing_conditions_hoarding"
PROPERTY_DISREPAIR = "property_level_disrepair_damp_mould"
TENURE_FALLBACK = [
    ("tenure-1", "tenure", "tenure_fallback"),
    ("tenure-2", "tenure", "tenure_fallback"),
]


# ==========================================
# Helpers & Data Factories
# ==========================================
def person(
    person_id,
    full_name,
    role="Tenant",
    is_responsible=True,
    date_of_birth="1985-03-15",
):
    return {
        "id": person_id,
        "type": "person",
        "fullName": full_name,
        "dateOfBirth": date_of_birth,
        "personTenureType": role,
        "isResponsible": is_responsible,
    }


def jane(role="Tenant", is_responsible=True, date_of_birth="1985-03-15"):
    return person("person-jane", "Jane Doe", role, is_responsible, date_of_birth)


def row(note_id, tenure_id, household_members, text=NOTE_TEXT):
    return {
        "note_id": note_id,
        "note_category": "General",
        "original_text": text,
        "note_date": NOTE_DATE,
        "tenure_id": tenure_id,
        "household_members": household_members,
    }


def two_tenure_rows(text=NOTE_TEXT, date_of_birth="1985-03-15"):
    """One note matched to two tenures, with Jane on both."""
    return [
        row("note-1", "tenure-1", [jane(date_of_birth=date_of_birth)], text),
        row("note-1", "tenure-2", [jane("HouseholdMember", False, date_of_birth)], text),
    ]


def span_predict(text, label, confidence=0.99):
    return (text, label, confidence)


def link_keys(note):
    return [(link.target_id, link.target_type, link.method) for link in note.links]


# ==========================================
# Test Suite
# ==========================================


class TestOrchestration:
    """Tests for Athena, JSON parsing, and batch file writing."""

    def test_handles_empty_athena_result_set(self, run_pipeline):
        run = run_pipeline([])
        assert run.parts == []
        assert f'"{run.config.athena_table}"' in run.query
        assert run.config.athena_where in run.query

    def test_handles_malformed_household_json_gracefully(self, run_pipeline):
        rows = [row("note-1", "tenure-1", "{not json"), row("note-1", "tenure-2", "{not json")]
        spans = [
            span_predict("mobility needs", PHYSICAL_HEALTH),
            span_predict("Jane Doe", PERSON_NAME),
        ]

        note = run_pipeline(rows, spans, relations={("mobility needs", "Jane Doe")}).note

        assert note.household_members == []
        assert note.tenure_ids == ["tenure-1", "tenure-2"]
        assert link_keys(note) == TENURE_FALLBACK

    def test_flushes_multiple_part_files_for_batch_threshold(self, run_pipeline):
        rows = [row("note-1", "tenure-1", [jane()]), row("note-2", "tenure-2", [jane()])]
        spans = [
            span_predict("mobility needs", PHYSICAL_HEALTH),
            span_predict("Jane Doe", PERSON_NAME),
        ]

        run = run_pipeline(
            rows,
            spans,
            relations={("mobility needs", "Jane Doe")},
            batch_size=1,
            flush_every_n_batches=1,
        )

        assert [[note.id for note in part] for part in run.parts] == [["note-1"], ["note-2"]]
        assert all(note.links[0].method == "fuzzy_name" for note in run.notes)


class TestExtractionAndRelations:
    """Tests for span identification, thresholds, and relation extraction.
    We inject fixed tokenisers and fixed huggingface model outputs while still testing our code."""

    def test_emits_exact_spans_relations_and_person_link(self, run_pipeline):
        text = "Jane Doe uses a wheelchair."
        spans = [
            span_predict("Jane Doe", PERSON_NAME, 0.95),
            span_predict("uses a wheelchair", MOBILITY, 0.90),
        ]

        run = run_pipeline(
            two_tenure_rows(text), spans, relations={("uses a wheelchair", "Jane Doe")}
        )
        note = run.note

        assert note.id == "note-1"
        assert note.source == "mmh"
        assert note.text == f"[Category: General] {text}"
        assert note.date == NOTE_DATE
        assert note.model == f"{run.config.span_model_name}_{run.config.relation_model_name}"
        assert note.pipeline_run_at is not None
        assert note.tenure_ids == ["tenure-1", "tenure-2"]
        assert all(
            note.text[span.start : span.end] == span.text for span in note.needs + note.persons
        )
        assert [(span.text, span.label) for span in note.needs] == [("uses a wheelchair", MOBILITY)]
        assert [(span.text, span.label) for span in note.persons] == [("Jane Doe", PERSON_NAME)]
        assert note.needs[0].confidence == pytest.approx(0.90)
        assert note.persons[0].confidence == pytest.approx(0.95)
        assert note.relations[0].from_id == note.needs[0].id
        assert note.relations[0].to == note.persons[0].id
        assert link_keys(note) == [("person-jane", "person", "fuzzy_name")]

        # Check Pydantic serialization
        assert "from" in run.raw_note["relations"][0]
        assert "from_id" not in run.raw_note["relations"][0]

        (household_person,) = note.household_members
        assert household_person.id == "person-jane"
        assert {(m.tenure_id, m.personTenureType) for m in household_person.memberships} == {
            ("tenure-1", "Tenant"),
            ("tenure-2", "HouseholdMember"),
        }

    def test_links_two_needs_to_their_respective_people(self, run_pipeline):
        text = "Jane Doe has limited mobility. John Smith has hoarding concerns."
        household = [jane(), person("person-john", "John Smith", "HouseholdMember", False)]
        spans = [
            span_predict("Jane Doe", PERSON_NAME),
            span_predict("limited mobility", MOBILITY),
            span_predict("John Smith", PERSON_NAME),
            span_predict("hoarding concerns", HOARDING),
        ]

        note = run_pipeline(
            [row("note-1", "tenure-1", household, text)],
            spans,
            relations={("limited mobility", "Jane Doe"), ("hoarding concerns", "John Smith")},
        ).note

        needs = {span.text: span.id for span in note.needs}
        people = {span.text: span.id for span in note.persons}
        relation_ids = {(relation.from_id, relation.to) for relation in note.relations}

        assert relation_ids == {
            (needs["limited mobility"], people["Jane Doe"]),
            (needs["hoarding concerns"], people["John Smith"]),
        }
        assert {(link.need_id, link.target_id, link.method) for link in note.links} == {
            (needs["limited mobility"], "person-jane", "fuzzy_name"),
            (needs["hoarding concerns"], "person-john", "fuzzy_name"),
        }

    def test_links_one_shared_need_to_two_people(self, run_pipeline):
        text = "Jane Doe and John Smith need home support."
        household = [jane(), person("person-john", "John Smith", "HouseholdMember", False)]
        spans = [
            span_predict("Jane Doe", PERSON_NAME),
            span_predict("John Smith", PERSON_NAME),
            span_predict("home support", PHYSICAL_HEALTH),
        ]

        note = run_pipeline(
            [row("note-1", "tenure-1", household, text)],
            spans,
            relations={("home support", "Jane Doe"), ("home support", "John Smith")},
        ).note

        assert len(note.needs) == 1
        assert len(note.persons) == 2
        assert len(note.relations) == 2
        assert {link.target_id for link in note.links} == {"person-jane", "person-john"}

    def test_falls_back_when_relation_is_rejected(self, run_pipeline):
        spans = [span_predict("Jane Doe", PERSON_NAME), span_predict("limited mobility", MOBILITY)]
        note = run_pipeline(
            two_tenure_rows("Jane Doe has limited mobility."), spans, relations=set()
        ).note

        assert [span.text for span in note.persons] == ["Jane Doe"]
        assert [span.text for span in note.needs] == ["limited mobility"]
        assert note.relations == []
        assert link_keys(note) == TENURE_FALLBACK


class TestEntityLinking:
    """Tests for business logic, minor exclusion, and tenure fallbacks."""

    def test_links_role_mention_when_unique(self, run_pipeline):
        spans = [
            span_predict("mobility needs", PHYSICAL_HEALTH),
            span_predict("the tenant", PERSON_ROLE),
        ]
        note = run_pipeline(
            two_tenure_rows(), spans, relations={("mobility needs", "the tenant")}
        ).note
        assert link_keys(note) == [("person-jane", "person", "role_tenant")]

    def test_falls_back_when_role_mention_is_ambiguous(self, run_pipeline):
        rows = [
            row(
                "note-1",
                "tenure-1",
                [person("person-jane", "Jane Doe")],
                "The tenant has mobility needs.",
            ),
            row(
                "note-1",
                "tenure-2",
                [person("person-john", "John Smith")],
                "The tenant has mobility needs.",
            ),
        ]
        spans = [
            span_predict("mobility needs", PHYSICAL_HEALTH),
            span_predict("The tenant", PERSON_ROLE),
        ]

        note = run_pipeline(rows, spans, relations={("mobility needs", "The tenant")}).note
        assert link_keys(note) == TENURE_FALLBACK

    def test_falls_back_when_name_is_unknown(self, run_pipeline):
        spans = [
            span_predict("mobility needs", PHYSICAL_HEALTH),
            span_predict("Alex Kim", PERSON_NAME),
        ]
        note = run_pipeline(
            two_tenure_rows(), spans, relations={("mobility needs", "Alex Kim")}
        ).note
        assert link_keys(note) == TENURE_FALLBACK

    def test_excludes_minor_and_falls_back_to_tenure(self, run_pipeline):
        spans = [
            span_predict("mobility needs", PHYSICAL_HEALTH),
            span_predict("Jane Doe", PERSON_NAME),
        ]
        note = run_pipeline(
            two_tenure_rows(date_of_birth="2018-06-01"),
            spans,
            relations={("mobility needs", "Jane Doe")},
        ).note

        assert note.household_members[0].dateOfBirth == "2018-06-01"
        assert link_keys(note) == TENURE_FALLBACK

    def test_links_property_level_need_to_tenures_only(self, run_pipeline):
        spans = [
            span_predict("Jane Doe", PERSON_NAME),
            span_predict("damp and mould", PROPERTY_DISREPAIR),
        ]
        note = run_pipeline(
            two_tenure_rows("Jane Doe reported damp and mould."),
            spans,
            relations={("damp and mould", "Jane Doe")},
        ).note
        assert link_keys(note) == TENURE_FALLBACK
