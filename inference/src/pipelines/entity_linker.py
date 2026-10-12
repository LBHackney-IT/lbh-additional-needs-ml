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
import re
from datetime import date, datetime
from typing import TypedDict
from zoneinfo import ZoneInfo

from src.common.schemas import EnrichedNote, HouseholdMember, Link


class ResolveResult(TypedDict):
    """Simple result from resolving step."""

    target_id: str | None
    confidence: float
    method: str | None


class DeterministicEntityLinker:
    """Rule-based engine for linking entity span relations to real household records."""

    def __init__(
        self,
        fuzzy_similarity: float = 0.7,
        tenant_role: str = "tenant",
        leaseholder_role: str = "leaseholder",
        logger: logging.Logger | None = None,
    ):
        self.fuzzy_similarity = fuzzy_similarity
        self.tenant_role = tenant_role
        self.leaseholder_role = leaseholder_role
        self.logger = logger

    @staticmethod
    def _fuzzy(a: str, b: str) -> float:
        """Compute case-insensitive string similarity ratio in [0, 1]."""
        return difflib.SequenceMatcher(None, a.lower(), b.lower()).ratio()

    def _resolve_span(
        self,
        extracted: str,
        entity_type: str,
        household: list[HouseholdMember],
    ) -> ResolveResult:
        """Match an extracted person span against a household member ID."""
        text_lower = extracted.lower().strip()

        if entity_type == "person_name":
            matches = [(self._fuzzy(extracted, p.fullName or ""), p.id) for p in household]
            if matches:
                best_score, best_id = max(matches, key=lambda x: x[0])
                if best_score >= self.fuzzy_similarity:
                    return ResolveResult(
                        target_id=best_id,
                        confidence=best_score,
                        method="fuzzy_name",
                    )

        elif entity_type == "person_role":
            if self._is_role_mention(text_lower, self.tenant_role):
                tenants = [
                    p
                    for p in household
                    if str(p.personTenureType or "").lower() == self.tenant_role
                ]
                if len(tenants) == 1:
                    return ResolveResult(
                        target_id=tenants[0].id,
                        confidence=1.0,
                        method="role_tenant",
                    )

            if self._is_role_mention(text_lower, self.leaseholder_role):
                leaseholders = [
                    p
                    for p in household
                    if str(p.personTenureType or "").lower() == self.leaseholder_role
                ]
                if len(leaseholders) == 1:
                    return ResolveResult(
                        target_id=leaseholders[0].id,
                        confidence=1.0,
                        method="role_leaseholder",
                    )

        return ResolveResult(target_id=None, confidence=0.0, method=None)

    @staticmethod
    def _is_role_mention(text: str, role: str) -> bool:
        """Return true if exact string match to role or 'the role'."""
        normalized = re.sub(r"^the\s+", "", text.strip())
        return normalized == role.lower()

    @staticmethod
    def _is_minor(person: HouseholdMember | None) -> bool:
        """Check if a matched household member is under 18 years old."""
        if not person or not person.dateOfBirth:
            return False

        dob_raw = person.dateOfBirth
        dob = dob_raw if isinstance(dob_raw, date) else datetime.fromisoformat(str(dob_raw)).date()
        today = datetime.now(ZoneInfo("UTC")).date()
        age = today.year - dob.year - ((today.month, today.day) < (dob.month, dob.day))
        return age < 18

    def link(self, record: EnrichedNote) -> list[Link]:
        """Resolve all predicted needs in an EnrichedNote to person or tenure targets."""
        household = record.household_members
        household_lookup = {hm.id: hm for hm in household}
        needs_lookup = {n.id: n for n in record.needs}
        persons_lookup = {p.id: p for p in record.persons}
        tenure_ids = record.tenure_ids

        links: list[Link] = []

        # 1. Resolve relations to person targets
        for rel in record.relations:
            need_id, person_ref_id = rel.from_id, rel.to

            if need_id not in needs_lookup or person_ref_id not in persons_lookup:
                if self.logger:
                    self.logger.warning(
                        "Skipping unknown relation IDs: %s -> %s in note %s",
                        need_id,
                        person_ref_id,
                        record.id,
                    )
                continue

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

            res = self._resolve_span(extracted_text, entity_label, household)
            target_id = res["target_id"]

            # O(1) minor check using dictionary lookup
            if target_id and self._is_minor(household_lookup.get(target_id)):
                if self.logger:
                    self.logger.info("Excluding minor person %s for need %s", target_id, need_id)
                continue

            if target_id and res["method"]:
                links.append(
                    Link(
                        need_id=need_id,
                        target_id=target_id,
                        target_type="person",
                        confidence=res["confidence"],
                        method=res["method"],
                    )
                )

        # 2. Fallback unresolved needs to tenure targets
        resolved_needs = {link.need_id for link in links}
        for need in record.needs:
            if need.id in resolved_needs:
                continue

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
