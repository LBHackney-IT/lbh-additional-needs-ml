"""
Post-processing step that maps pipeline-predicted (need, person) relations
to real database IDs from the household members list using simple heuristics.

HEURISTIC RULES:
    1. If the need is linked to a 'person_name', fuzzy match the household member's name to the extracted span.
    2. If it is linked to a 'person_role' and the extracted span is 'tenant'/'leaseholder' AND there is only one of those person types in the household list, link them.
    3. If the linked person is under 18, default to the tenancy/household (drop relation).
"""

import sys
import difflib
from dataclasses import dataclass
from datetime import date, datetime, timezone
from pathlib import Path

import pandas as pd

from common.json_helpers import load_json
from common.logging import setup_logger
from common.paths import PREDICTIONS


@dataclass
class Config:
    input_path_str:  str
    fuzzy_similarity: float = 0.7
    tenant_role:      str   = "tenant"
    leaseholder_role: str   = "leaseholder"
    logger = setup_logger("inference.entity_linking", "entity_linking.log")

    def __post_init__(self):
        self.input_path = Path(self.input_path_str)
        self.output_path = PREDICTIONS / f"e2e_{self.input_path.stem}.csv"


def _fuzzy(a: str, b: str) -> float:
    return difflib.SequenceMatcher(None, a.lower(), b.lower()).ratio()

def _resolve_span(extracted: str, entity_type: str, household: list[dict], cfg: Config) -> tuple[str | None, float]:
    """Matches an extracted entity to a household member ID based on simple rules."""
    text_lower = extracted.lower()

    # --- Match Exact Name (case-insensitive) ---
    if entity_type == "person_name":
        matches = [(_fuzzy(extracted, p.get("fullName", "")), p["id"]) for p in household]
        if matches:
            best_score, best_id = max(matches, key=lambda x: x[0])
            if best_score >= cfg.fuzzy_similarity:
                return best_id, best_score

    # --- Match by role ---
    elif entity_type == "person_role":
        if cfg.tenant_role in text_lower:
            tenants = [p for p in household if str(p.get("personTenureType", "")).lower() == cfg.tenant_role]
            if len(tenants) == 1:
                return tenants[0]["id"], 1.0  # If more than one tenant, don't try to resolve

        if cfg.leaseholder_role in text_lower:
            leaseholders = [p for p in household if str(p.get("personTenureType", "")).lower() == cfg.leaseholder_role]
            if len(leaseholders) == 1:
                return leaseholders[0]["id"], 1.0  # If more than one leaseholder, don't try to resolve

    return None, 0.0

def _is_minor(household, person_id):
    person = next((hm for hm in household if hm['id'] == person_id), None)
    if not person:
        raise ValueError(f"Person ID {person_id} not found in household.") 

    dob_raw = person.get("dateOfBirth")
    if not dob_raw:
        return False  # Not a minor if dob is unset.

    # safely convert to datetime
    dob = dob_raw if isinstance(dob_raw, date) else datetime.fromisoformat(dob_raw).date()
    today = date.today()
    
    # check if below 18
    age = today.year - dob.year - ((today.month, today.day) < (dob.month, dob.day))
    return age < 18

def resolve_relations(record: dict, cfg: Config) -> dict[str, list[dict]]:
    """Resolve each relation in a record to a person ID. 
    Returns a dict mapping need_ids to a list of resolution metadata dicts."""

    household = record.get("household_members", [])
    needs_lookup = {n["id"]: n for n in record.get("needs", [])}
    persons_lookup = {p["id"]: p for p in record.get("persons", [])}

    # Loop through relations and attempt to resolve
    resolved_needs = {}
    for rel in record.get("relations", []):
        need_id, person_ref_id = (rel['from'], rel['to'])  # ALWAYS EXPECTS NEED -> PERSON
        
        # Handle missing references
        if need_id not in needs_lookup or person_ref_id not in persons_lookup:
            cfg.logger.warning("Skipping unknown relation IDs: %s -> %s in note %s", 
                                need_id, person_ref_id, record.get("id", "unknown_id"))
            continue
        
        # Get extracted text, span positions, label, and confidence from person annotation
        p_pred = persons_lookup[person_ref_id]
        extracted_text = p_pred.get("text", "")
        entity_label = p_pred.get("label", "").lower()

        # Skip property-level relations (and log) 
        n_label = needs_lookup[need_id].get("label", "")
        if n_label.startswith('property_level'):
            cfg.logger.warning("%s label linked to %s in note ID %s", n_label, entity_label, record.get("id"))

        # Try to resolve to person ID
        person_id, score = _resolve_span(extracted_text, entity_label, household, cfg)
        if person_id and not _is_minor(household, person_id):
            resolved_needs.setdefault(need_id, []).append(
                {
                    "target_id": person_id,
                    "linking_score": score,
                    "person_extracted_id": p_pred.get("id"),
                    "person_label": p_pred.get("label"),
                    "person_text": extracted_text,
                    "person_start": p_pred.get("start"),
                    "person_end": p_pred.get("end"),
                    "person_conf": p_pred.get("confidence"),
                }
            )
    
    return resolved_needs


def main():
    if len(sys.argv) < 2:
        print("Usage: python match_needs_to_persons.py <path/to/data.json>")
        sys.exit(1)
    config = Config(input_path_str=sys.argv[1])

    # Load data
    records = load_json(config.input_path, config.logger)
    created_at = datetime.now(timezone.utc).isoformat()
    
    # Loop through each note and attempt to resolve each need, output as CSV 
    # (one row per need per doc)
    rows = []
    for record in records:
        note_id = record.get("id", "")
        tenure_ids = record.get("tenure_ids", [])
        model = record.get("model", "unknown")
        rel_confs = {r['from']: r.get("confidence") for r in record.get("relations", [])}  # ALWAYS EXPECTS NEED -> PERSON

        resolved_links = resolve_relations(record, config)

        for need in record.get("needs", []):
            # Set up row
            base_row = {
                "note_id": note_id,
                "note_date": record.get("date", ""),
                "model": model,
                "pipeline_run_at": created_at,
                "need_id": need["id"],
                "need_label": need["label"],
                "need_text": need.get("text", ""),
                "need_start": need.get("start"),
                "need_end": need.get("end"),
                "need_conf": need.get("confidence"),
                "relation_conf": rel_confs.get(need['id']),
            }

            # Update target_type and target_id depending on if linked to a person/persons 
            if need["id"] in resolved_links:
                for link in resolved_links[need["id"]]:
                    rows.append({
                        **base_row, 
                        "target_id": link["target_id"], 
                        "target_type": "person", 
                        "linking_conf": link["linking_score"],
                        "person_extracted_id": link["person_extracted_id"],
                        "person_label": link["person_label"],
                        "person_text": link["person_text"],
                        "person_start": link["person_start"],
                        "person_end": link["person_end"],
                        "person_conf": link["person_conf"],
                    })
            else:
                # Can't determine which person - resolve to all linked tenancies
                for tenure_id in tenure_ids: 
                    rows.append({
                        **base_row, 
                        "target_id": tenure_id, 
                        "target_type": "tenure", 
                        "linking_conf": 0 if base_row['relation_conf'] is not None else None,
                        "person_extracted_id": None,
                        "person_label": None,
                        "person_text": None,
                        "person_start": None,
                        "person_end": None,
                        "person_conf": None,
                    })

    # Convert to DataFrame and save to CSV
    columns = [
        "note_id", "note_date", "target_id", "target_type", "model", 
        "pipeline_run_at", "need_id", "need_label", "need_text", "need_start", "need_end", 
        "need_conf", "person_extracted_id", "person_label", "person_text", 
        "person_start", "person_end", "person_conf", "relation_conf", "linking_conf",
    ]
    df = pd.DataFrame(rows, columns=columns)
    df.to_csv(config.output_path, index=False, encoding="utf-8")

    config.logger.info("Written %d rows to %s", len(df), config.output_path)


if __name__ == "__main__":
    main()