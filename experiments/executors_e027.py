"""Visible-input policies. This module never loads hidden data or evaluators."""
from collections import Counter
from fractions import Fraction
import json
import math
import re

SYSTEM = """Choose one travel option for the requested aspect using only the public trip request, displayed options, and additional user statements supplied below. Respect the stated requirements. Do not invent unstated preferences or assume missing properties. When options otherwise meet the stated needs, favor lower cost. Some displayed options may be irrelevant or implausible; use the supplied descriptions to judge them. Choose a single listed option even when the information is insufficient. Do not ask another question.
Return exactly one JSON object containing only the key "id" with one of the listed option IDs. Do not include explanation, Markdown, or any other keys."""
STOP = set("""the and for that this with have has had from into about your you our are was were will would could should can cannot when where which what how who all any not but more most much very also than then them they their there here these those only some such other each both been being over under after before during between through want wants like likes need needs prefer prefers please usually always really often just because without while its it's get gets got one two three time trip travel travelling traveling option options cost total service services amenities name type capacity id city dates date days day choose choosing""".split())


def payload_for(public_case, aspect, revealed_statements):
    return {"trip_request": public_case["user_message"], "target_aspect": aspect,
            "additional_statements": list(revealed_statements),
            "options": public_case["options"][aspect]}


def messages_for(payload):
    assert set(payload) == {"trip_request", "target_aspect", "additional_statements", "options"}
    return [{"role": "system", "content": SYSTEM},
            {"role": "user", "content": json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"))}]


def tokens(text):
    return set(re.findall(r"[a-z]{3,}", text.casefold())) - STOP


def price(option, aspect):
    value = option.get("cost")
    finite = lambda x: isinstance(x, (int, float)) and not isinstance(x, bool) and math.isfinite(x) and x >= 0
    if aspect == "restaurant":
        return {"Cheap": 0, "Normal": 1, "Expensive": 2}.get(option.get("expectation"), math.inf)
    if aspect == "hotel":
        candidates = [v for v in value if finite(v)] if isinstance(value, list) else []
        return min(candidates) if candidates else math.inf
    return value if finite(value) else math.inf


def lexical_distribution(payload):
    options = payload["options"]
    docs = [tokens(json.dumps({k: v for k, v in o.items() if k != "id"}, sort_keys=True)) for o in options]
    counts = Counter(t for doc in docs for t in doc)
    idf = {t: math.log((len(options) + 1) / (n + 1)) + 1 for t, n in counts.items()}
    reply = tokens(" ".join(payload["additional_statements"]))
    request = tokens(payload["trip_request"])
    rows = [{"id": o["id"], "reply_overlap": math.fsum(idf[t] for t in sorted(doc & reply)),
             "request_overlap": math.fsum(idf[t] for t in sorted(doc & request)),
             "base_price": price(o, payload["target_aspect"])} for o, doc in zip(options, docs)]
    keep = rows
    for key in ["reply_overlap", "request_overlap"]:
        best = max(r[key] for r in keep)
        keep = [r for r in keep if abs(r[key] - best) <= 1e-12]
    cheapest = min(r["base_price"] for r in keep)
    keep = [r for r in keep if r["base_price"] == cheapest]
    selected = {r["id"] for r in keep}
    distribution = {o["id"]: str(Fraction(1, len(keep)) if o["id"] in selected else Fraction(0)) for o in options}
    for row in rows:
        if not math.isfinite(row["base_price"]):
            row["base_price"] = None
    return distribution, rows


def parse_choice(raw, allowed):
    def unique(pairs):
        if len(pairs) != len(dict(pairs)):
            raise ValueError("duplicate JSON key")
        return dict(pairs)
    try:
        obj = json.loads(raw, object_pairs_hook=unique)
    except (ValueError, TypeError):
        return None
    if type(obj) is not dict or set(obj) != {"id"} or type(obj["id"]) is not str or obj["id"] not in allowed:
        return None
    return obj["id"]


def score_bounds(distribution, truth_by_id, scale=Fraction(1)):
    lower = upper = Fraction(0)
    for option_id, value in distribution.items():
        coefficient = Fraction(value) * scale
        if truth_by_id[option_id] == "T":
            lower += coefficient
            upper += coefficient
        elif truth_by_id[option_id] == "U":
            lower += min(Fraction(0), coefficient)
            upper += max(Fraction(0), coefficient)
    return {"lower": str(lower), "upper": str(upper)}


def full_tie_gap(full_gains, partial_gains):
    maximum = max(Fraction(v) for v in full_gains.values())
    full_set = sorted(q for q, v in full_gains.items() if Fraction(v) == maximum)
    gap = max(Fraction(v) for v in partial_gains.values()) - max(Fraction(partial_gains[q]) for q in full_set)
    return {"full_maximizers": full_set, "gap": str(gap)}
