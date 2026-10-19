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

from src.common.schemas import EnrichedNote, HouseholdMember, Link, Relation


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

    # --- Static ---
    @staticmethod
    def _fuzzy(a: str, b: str) -> float:
        """Compute case-insensitive string similarity ratio in [0, 1]."""
        return difflib.SequenceMatcher(None, a.lower(), b.lower()).ratio()

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

    @staticmethod
    def _unresolved() -> ResolveResult:
        return ResolveResult(target_id=None, confidence=0.0, method=None)

    @staticmethod
    def _has_role(person: HouseholdMember, role: str) -> bool:
        """Return true when any of the person's tenancy memberships has this role."""
        return any(
            str(membership.personTenureType or "").lower() == role
            for membership in person.memberships
        )

    # --- Helpers ---
    def _resolve_name(self, extracted: str, household: list[HouseholdMember]) -> ResolveResult:
        """Resolve a person-name span using the strongest fuzzy match."""
        matches = [(self._fuzzy(extracted, p.fullName or ""), p.id) for p in household]
        if not matches:
            return self._unresolved()

        best_score, best_id = max(matches, key=lambda item: item[0])
        if best_score < self.fuzzy_similarity:
            return self._unresolved()

        return ResolveResult(target_id=best_id, confidence=best_score, method="fuzzy_name")

    def _resolve_role(self, extracted: str, household: list[HouseholdMember]) -> ResolveResult:
        """Resolve a tenant or leaseholder span when it is unique in household."""
        role_matches = (
            (self.tenant_role, "role_tenant"),
            (self.leaseholder_role, "role_leaseholder"),
        )
        for role, method in role_matches:
            if not self._is_role_mention(extracted, role):
                continue

            matches = [person for person in household if self._has_role(person, role)]
            if len(matches) == 1:
                return ResolveResult(
                    target_id=matches[0].id,
                    confidence=1.0,
                    method=method,
                )

        return self._unresolved()

    def _resolve_span(
        self,
        extracted: str,
        entity_type: str,
        household: list[HouseholdMember],
    ) -> ResolveResult:
        """Match an extracted person span against a household member ID."""
        if entity_type == "person_name":
            return self._resolve_name(extracted, household)
        if entity_type == "person_role":
            return self._resolve_role(extracted, household)
        return self._unresolved()

    def _process_relation(
        self,
        rel: Relation,
        needs_lookup: dict,
        persons_lookup: dict,
        household: list[HouseholdMember],
        household_lookup: dict,
        record_id: str,
    ) -> Link | None:
        """Process a single relation and return a Link if valid and resolved."""
        need_id, person_ref_id = rel.from_id, rel.to

        # Check links are valid
        if need_id not in needs_lookup or person_ref_id not in persons_lookup:
            if self.logger:
                self.logger.warning(
                    "Skipping unknown relation IDs: %s -> %s in note %s",
                    need_id,
                    person_ref_id,
                    record_id,
                )
            return None

        p_pred = persons_lookup[person_ref_id]
        n_label = needs_lookup[need_id].label
        res = self._resolve_span(p_pred.text, p_pred.label.lower(), household)
        target_id = res["target_id"]

        # Property-level needs belong to the tenancy, even when the relation
        # model associates them with a person mention.
        if n_label.startswith("property_level"):
            if self.logger:
                self.logger.info(
                    "Keeping property-level need %s at tenure level in note %s",
                    need_id,
                    record_id,
                )
            return None

        # Skip person links for minors or when no household member was resolved.
        if not target_id:
            return None

        if self._is_minor(household_lookup.get(target_id)):
            if self.logger:
                self.logger.info("Excluding minor person %s for need %s", target_id, need_id)
            return None

        if res["method"]:
            return Link(
                need_id=need_id,
                target_id=target_id,
                target_type="person",
                confidence=res["confidence"],
                method=res["method"],
            )

        return None

    def _generate_fallback_links(
        self, record: EnrichedNote, resolved_needs: set[str]
    ) -> list[Link]:
        """Generate fallback tenure links for any needs that remain unresolved."""
        return [
            Link(
                need_id=need.id,
                target_id=tenure_id,
                target_type="tenure",
                confidence=0.0,
                method="tenure_fallback",
            )
            for need in record.needs
            if need.id not in resolved_needs
            for tenure_id in record.tenure_ids
        ]

    # --- Public ---
    def link(self, record: EnrichedNote) -> list[Link]:
        """Resolve all predicted needs in an EnrichedNote to person or tenure targets."""
        household = record.household_members
        household_lookup = {hm.id: hm for hm in household}
        needs_lookup = {n.id: n for n in record.needs}
        persons_lookup = {p.id: p for p in record.persons}

        # 1. Attempt to resolve relations to person targets
        links: list[Link] = [
            link_obj
            for rel in record.relations
            if (
                link_obj := self._process_relation(
                    rel, needs_lookup, persons_lookup, household, household_lookup, record.id
                )
            )
            is not None
        ]

        # 2. Link unresolved needs to tenure targets
        resolved_needs = {link.need_id for link in links}
        links.extend(self._generate_fallback_links(record, resolved_needs))

        return links
