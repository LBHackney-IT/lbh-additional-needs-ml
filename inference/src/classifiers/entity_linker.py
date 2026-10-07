"""Deterministic entity linker.

Maps predicted (need, person) relations onto real household-member IDs using rule-based heuristics.

HEURISTIC RULES:
    1. person_name spans -> fuzzy-match against household fullName.
    2. person_role spans -> match sole tenant/leaseholder when text names that role
       (skipped if multiple exist).
    3. Matched persons under 18 are excluded (falls through to tenure).
    4. Any unresolved needs -> fallback link to tenure_id.
"""

from __future__ import annotations

import difflib
import logging
from dataclasses import dataclass
from datetime import date, datetime
from typing import Any, TypedDict
from zoneinfo import ZoneInfo

from common.schemas import EnrichedNote, Link


class ResolveResult(TypedDict):
    """Simple result from resolving step."""

    target_id: str | None
    confidence: float
    method: str | None


@dataclass
class DeterministicEntityLinker:
    """Rule-based engine for linking entity span relations to real household records.

    Attributes:
        fuzzy_similarity: Minimum sequence matcher score ratio required for name matches.
        tenant_role: Keyword identifying tenant roles.
        leaseholder_role: Keyword identifying leaseholder roles.
        logger: Optional logger.
    """

    fuzzy_similarity: float = 0.7
    tenant_role: str = "tenant"
    leaseholder_role: str = "leaseholder"
    logger: logging.Logger | None = None

    def _fuzzy(self, a: str, b: str) -> float:
        """Compute case-insensitive string similarity ratio in [0, 1].

        Args:
            a: First string comparison candidate.
            b: Second string comparison candidate.

        Returns:
            Similarity ratio score between 0.0 and 1.0.
        """
        return difflib.SequenceMatcher(None, a.lower(), b.lower()).ratio()

    def _resolve_span(
        self,
        extracted: str,
        entity_type: str,
        household: list[dict[str, Any]],
    ) -> ResolveResult:
        """Match an extracted person span against a household member ID.

        Args:
            extracted: Raw text string of the person mention.
            entity_type: Label of the entity ('person_name' or 'person_role').
            household: List of household member dictionary records.

        Returns: ResolveResult (target_id, confidence, method)
        """
        text_lower = extracted.lower()

        if entity_type == "person_name":
            matches = [(self._fuzzy(extracted, p.get("fullName", "")), p["id"]) for p in household]
            if matches:
                best_score, best_id = max(matches, key=lambda x: x[0])
                if best_score >= self.fuzzy_similarity:
                    return ResolveResult(best_id, best_score, "fuzzy_name")

        elif entity_type == "person_role":
            if self.tenant_role in text_lower:
                tenants = [
                    p
                    for p in household
                    if str(p.get("personTenureType", "")).lower() == self.tenant_role
                ]
                if len(tenants) == 1:
                    return ResolveResult(tenants[0]["id"], 1.0, "role_tenant")

            if self.leaseholder_role in text_lower:
                leaseholders = [
                    p
                    for p in household
                    if str(p.get("personTenureType", "")).lower() == self.leaseholder_role
                ]
                if len(leaseholders) == 1:
                    return ResolveResult(leaseholders[0]["id"], 1.0, "role_leaseholder")

        return ResolveResult(None, 0.0, None)

    def _is_minor(self, household: list[dict[str, Any]], person_id: str) -> bool:
        """Check if a matched household member is under 18 years old.

        Treats missing birth dates as adult by default.

        Args:
            household: List of household member records.
            person_id: Unique person ID to verify.

        Returns:
            True if person is under 18; False otherwise.
        """
        person = next((hm for hm in household if hm["id"] == person_id), None)
        if not person:
            return False

        dob_raw = person.get("dateOfBirth")
        if not dob_raw:
            return False

        dob = dob_raw if isinstance(dob_raw, date) else datetime.fromisoformat(str(dob_raw)).date()
        today = datetime.now(ZoneInfo("UTC")).date()
        age = today.year - dob.year - ((today.month, today.day) < (dob.month, dob.day))
        return age < 18

    def link(self, record: EnrichedNote) -> list[Link]:
        """Resolve all predicted needs in an EnrichedNote to person or tenure targets.

        Args:
            record: EnrichedNote domain object.

        Returns:
            List of resolved Pydantic Link objects.
        """
        household = record.household_members
        needs_lookup = {n.id: n for n in record.needs}
        persons_lookup = {p.id: p for p in record.persons}
        tenure_ids = record.tenure_ids

        # person_links_by_need: dict[str, list[Link]] = {}
        links: list[Link] = []

        # Loop through each relation
        for rel in record.relations:
            need_id, person_ref_id = rel.from_id, rel.to_id

            # Check IDs exist
            if need_id not in needs_lookup or person_ref_id not in persons_lookup:
                if self.logger:
                    self.logger.warning(
                        "Skipping unknown relation IDs: %s -> %s in note %s",
                        need_id,
                        person_ref_id,
                        record.id,
                    )
                continue

            # Get the related spans by IDs
            p_pred = persons_lookup[person_ref_id]
            extracted_text = p_pred.text
            entity_label = p_pred.label.lower()
            n_label = needs_lookup[need_id].label
            if n_label.startswith("property_level") and self.logger:
                self.logger.warning(
                    "%s label linked to %s in note ID %s",
                    n_label,
                    entity_label,
                    record.id,
                )

            # Attempt to resolve; drop if minor
            res = self._resolve_span(extracted_text, entity_label, household)
            if res.person_id and self._is_minor(household, res.person_id):
                if self.logger:
                    self.logger.info(
                        "Excluding minor person %s for need %s", res.person_id, need_id
                    )
                continue

            if res.person_id and res.method:
                links.append(
                    Link(
                        need_id=need_id,
                        target_id=res.person_id,
                        target_type="person",
                        confidence=res.confidence,
                        method=res.method,
                    )
                )

        # Loop through needs and resolve any that don't have a relation or link to tenures
        resolved_needs = [link.need_id for link in links]
        for need in record.needs:
            if need.id in resolved_needs:
                continue
            else:
                for tenure_id in tenure_ids:
                    links.append(
                        Link(
                            need_id=need.id,
                            target_id=tenure_id,
                            target_type="tenure",
                            confidence=0.0,
                            method="tenure_fallback",
                        )
                    )

        return links
