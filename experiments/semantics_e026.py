"""Three-valued preference predicates for the frozen E026 catalogue."""
import math
import re

MISSING = object()


def normalize(value):
    return " ".join(value.casefold().replace("-", " ").split())


def number(value):
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)


def strings(value):
    return isinstance(value, list) and all(isinstance(x, str) for x in value)


def truth(condition):
    return "T" if condition else "F"


def phrase_in(value, phrase):
    return re.search(r"(?<!\w)" + re.escape(normalize(phrase)) + r"(?!\w)", normalize(value)) is not None


def evaluate(criterion, option):
    if criterion["status"] != "operational":
        return "U"
    rule = criterion["rule"]
    op = rule["op"]
    if op == "service_available":
        value = option
        for part in rule["path"]:
            if not isinstance(value, dict) or part not in value:
                return "U"
            value = value[part]
        if value is None:
            return "F"
        return "T" if number(value) else "U"
    if op in {"equals", "list_contains", "all_text_words", "numeric_range"}:
        value = option.get(rule["field"], MISSING)
        if op == "equals":
            return truth(normalize(value) == normalize(rule["value"])) if isinstance(value, str) else "U"
        if op == "numeric_range":
            return truth(rule["minimum"] <= value <= rule["maximum"]) if number(value) else "U"
        if not strings(value):
            return "U"
        if op == "list_contains":
            return truth(normalize(rule["value"]) in [normalize(v) for v in value])
        return truth(all(phrase_in(v, rule["value"]) for v in value)) if value else "U"
    if op == "text_words":
        values = [option.get(field, MISSING) for field in rule["fields"]]
        if not all(isinstance(v, str) for v in values):
            return "U"
        return truth(phrase_in(" ".join(values), rule["value"]))
    if op == "room_type":
        rooms = option.get("room")
        if not isinstance(rooms, list):
            return "U"
        known = [r["type"] for r in rooms if isinstance(r, dict) and isinstance(r.get("type"), str)]
        if any(normalize(v) == normalize(rule["value"]) for v in known):
            return "T"
        return "F" if len(known) == len(rooms) else "U"
    if op == "bedrooms_min":
        room = option.get("room")
        match = re.fullmatch(r"(\d+)B\d+B", room, flags=re.I) if isinstance(room, str) else None
        return truth(int(match[1]) >= rule["minimum"]) if match else "U"
    if op == "review_majority":
        reviews = option.get("reviews")
        if not isinstance(reviews, dict):
            return "U"
        counts = [reviews.get(str(i) + "_star", MISSING) for i in range(1, 6)]
        if not all(number(v) and v >= 0 for v in counts) or sum(counts) == 0:
            return "U"
        return truth(2 * counts[-1] > sum(counts))
    if op in {"direct", "layover_city", "layover_range"}:
        path = option.get("path")
        if not strings(path) or len(path) < 2:
            return "U"
        if op == "direct":
            return truth(len(path) == 2)
        if op == "layover_city":
            return truth(normalize(rule["value"]) in [normalize(x) for x in path[1:-1]])
        times = option.get("time")
        if not isinstance(times, list) or len(times) != 2 * len(path) - 3 or not all(number(v) and v >= 0 for v in times):
            return "U"
        return truth(len(path) - 2 == rule["connections"] and
                     all(rule["minimum"] <= v <= rule["maximum"] for v in times[1::2]))
    if op in {"same_companies", "distinct_companies"}:
        companies = option.get("company")
        if not strings(companies) or not companies:
            return "U"
        unique = set(normalize(v) for v in companies)
        return truth(len(unique) == 1) if op == "same_companies" else truth(len(companies) >= 2 and len(unique) == len(companies))
    raise ValueError("Unknown criterion operator: " + op)


def conjunction(values):
    if "F" in values:
        return "F"
    return "T" if all(v == "T" for v in values) else "U"
