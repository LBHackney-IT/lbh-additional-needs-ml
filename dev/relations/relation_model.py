"""
Shared pieces used by train_relation_extraction.py, eval_relation_extraction.py,
and eval_relation_closest_match_baseline.py, so marker injection and scoring
logic can't drift between training, model eval, and the baseline.
"""

SPECIAL_TOKENS = ["[N_START]", "[N_END]", "[P_START]", "[P_END]"]


def insert_markers(text: str, need: dict, person: dict) -> str:
    """Copy of RelationClassifierPipeline._insert_markers. Crossing spans raise."""
    spans = [
        (need["start"], need["end"], "[N_START]", "[N_END]"),
        (person["start"], person["end"], "[P_START]", "[P_END]"),
    ]
    for start, end, _start_marker, _end_marker in spans:
        if not 0 <= start <= end <= len(text):
            raise ValueError(f"Span offsets are outside the source text: {start}:{end}")

    first, second = spans
    if (first[0] < second[0] < first[1] < second[1]) or (
        second[0] < first[0] < second[1] < first[1]
    ):
        raise ValueError("Need and person spans overlap without containment")

    events = {}
    for start, end, start_marker, end_marker in spans:
        events.setdefault(start, []).append((1, -(end - start), start_marker))
        events.setdefault(end, []).append((0, end - start, end_marker))

    marked_parts = []
    cursor = 0
    for position in sorted(events):
        marked_parts.append(text[cursor:position])
        marked_parts.extend(marker for _, _, marker in sorted(events[position]))
        cursor = position
    marked_parts.append(text[cursor:])
    return "".join(marked_parts)
