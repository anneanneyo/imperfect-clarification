"""Construct criteria from E025 preference records, without evaluating options."""
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results/userbench_semantics_e026"


def criterion(row):
    p, c = row["preference"], row["case_id"]
    contains = {
        "prefer having wifi on the flight": ("amenities", "WiFi"),
        "prefer having meal service on the flight": ("amenities", "Meal Service"),
        "prefer having carry on baggage allowance on the flight": ("amenities", "Carry on Baggage Allowance"),
        "prefer having business workspace": ("amenities", "Business Workspace"),
        "prefer having air conditioning": ("amenities", "Air Conditioning"),
        "prefer having high-speed wifi": ("amenities", "High-Speed WiFi"),
        "prefer restaurant opens till late night": ("tags", "Late Night Service"),
        "prefer having business dining": ("tags", "Business Dining"),
        "prefer having the option of delivery": ("tags", "Delivery Available"),
        "prefer having outdoor seating": ("tags", "Outdoor Seating"),
        "prefer where walk-ins are welcomed": ("tags", "Walk-ins Welcomed"),
        "prefer the restaurant to be pet-friendly": ("tags", "Pet-Friendly"),
        "prefer where reservations are available": ("tags", "Reservations Available"),
    }
    services = {
        "prefer business class": ["service", "cost_business_total"],
        "prefer having belonging insurance": ["insurance", "cost_belonging_waiver"],
        "prefer having liability insurance": ["insurance", "cost_liability_waiver"],
        "prefer having driver's personal accident insurance": ["insurance", "cost_personal_accident_waiver"],
        "prefer having more than one main driver": ["service", "cost_additional_driver_total"],
        "prefer having additional child seat": ["service", "cost_child_seat_total"],
        "prefer having parking service": ["service", "cost_parking_total"],
        "prefer to checkin early": ["service", "cost_early_checkin_fee"],
        "prefer having checked bags service": ["service", "cost_check_bag_total"],
        "prefer underage person is able to drive": ["service", "cost_underage_total"],
        "prefer having barrier-free accessibility services": ["service", "cost_barrier_free_total"],
    }
    categories = {"prefer SUV": "SUV", "prefer electric car": "Electric",
                  "prefer economy car": "Economy", "prefer gasoline car": "Gasoline"}
    unresolved = {
        "prefer reasonable layover time": ("ambiguous", "The implicit statement gives no numeric boundary; the generation template's generally 1-3 hours is not a user-specified threshold."),
        "prefer Double room": ("ambiguous", "The implicit statement needs two beds; the room label Double and capacity do not specify bed count."),
        "prefer minimum total travel time": ("ambiguous", "Relative objective needs route feasibility and a defined comparison set; it is not a per-option boolean threshold."),
        "prefer having enough number of seats (user specifies)": ("unsupported", "The user dislikes reclining comfort in five-seat cars. Seat count above five does not establish reclining space."),
        "prefer to have the room cleaned up everyday": ("unsupported", "A cleaning fee does not specify daily service frequency."),
        "prefer having dessert or bakery or cafe (user specifies)": ("unsupported", "Coffee-shop card eligibility or availability of macarons is not supplied by cuisine category alone."),
    }
    if p in unresolved:
        status, reason = unresolved[p]
        return status, None, reason
    if p in contains:
        field, value = contains[p]
        return "operational", {"op": "list_contains", "field": field, "value": value}, "Source list-membership convention; absent item in a present complete list is F. This does not independently validate real service quality."
    if p in services:
        return "operational", {"op": "service_available", "path": services[p]}, "Source availability convention: explicit null means unavailable, a numeric fee including zero means available; missing field is U. No promise about actual timing, coverage, age eligibility or accessibility details."
    if p in categories:
        return "operational", {"op": "equals", "field": "categories", "value": categories[p]}, "Matches the explicit source preference's car category; the implicit motivation does not independently guarantee safety, traction, comfort or autopilot."
    if p == "prefer Suite (with kitchen) room":
        return "operational", {"op": "room_type", "value": "Suite (with kitchen)"}, "At least one listed room has the specified type; room price and capacity are separate concerns."
    if p == "prefer having more than half of the reviews are 5-star":
        return "operational", {"op": "review_majority"}, "Require a strictly greater than 1/2 share of five-star reviews among the five supplied rating counts."
    if p == "prefer longer layover time":
        if c == "E025-03":
            return "operational", {"op": "layover_range", "minimum": 4, "maximum": 5, "connections": 1}, "Uses the explicitly stated 4-to-5-hour range inclusively, with the source criterion's single connection; other interpretations of soft wording are possible."
        return "ambiguous", None, "Ample time to meet people gives no usable numeric boundary; no cutoff is invented."
    if p == "prefer specific rental car brand (user specifies)":
        target = {"E025-02": "Avis", "E025-06": "Alamo"}[c]
        return "operational", {"op": "equals", "field": "brand", "value": target}, "Brand explicitly named in the implicit statement; no assertion about pickup convenience or fleet."
    if p == "prefer specific car model (user specifies)":
        target = {"E025-02": "Tesla Model 3", "E025-08": "Hyundai Ioniq", "E025-10": "Kia Telluride"}[c]
        return "operational", {"op": "text_words", "fields": ["brand", "model"], "value": target}, "Explicitly named model is matched in combined brand/model fields; do not infer its capabilities."
    if p == "prefer certain platform (user specifies)":
        target = {"E025-02": "Homestay", "E025-07": "Zillow Rental"}[c]
        return "operational", {"op": "text_words", "fields": ["name"], "value": target}, "Source represents the named platform in the listing name. This cannot confirm gift-card acceptance or real platform provenance."
    if p == "prefer having at least three bedrooms":
        return "operational", {"op": "bedrooms_min", "minimum": 3}, "Decode the source nBmB convention as n bedrooms and m bathrooms; unsupported room strings are U."
    if p == "prefer specific airline brand (user specifies)":
        return "operational", {"op": "all_text_words", "field": "company", "value": "United"}, "The user names United; require its name on every listed flight segment."
    if p == "prefer having seafood or steakhouse (user specifies)":
        return "operational", {"op": "equals", "field": "cuisine", "value": "Steakhouse"}, "The user specifies ribeye or steak, so the source's cuisine-category criterion is specialized to Steakhouse. Category is a proxy, not verification of a dish or wine list."
    if p == "prefer a specific layover city (user designated)":
        target = {"E025-04": "Hong Kong", "E025-05": "Bangkok"}[c]
        return "operational", {"op": "layover_city", "value": target}, "Require the explicitly named city among intermediate path points; airport facilities and sufficient connection time are unverified."
    if p == "prefer hotel rating within a specific range (user specifies)":
        return "operational", {"op": "numeric_range", "field": "rating", "minimum": 6, "maximum": 9}, "User explicitly gives 6 to 9; use inclusive endpoints."
    if p in {"prefer having cheap restaurant", "prefer having expensive restaurant"}:
        return "operational", {"op": "equals", "field": "expectation", "value": "Cheap" if "cheap" in p else "Expensive"}, "Source price-tier convention, not a numeric cost or proof of value/ambiance."
    if p == "prefer cuisine from a specific country (user specifies)":
        return "operational", {"op": "equals", "field": "cuisine", "value": {"E025-07": "Mexican", "E025-11": "Korean"}[c]}, "Country/cuisine explicitly named in the implicit statement."
    if p == "prefer direct flight":
        return "operational", {"op": "direct"}, "A two-point route has no connection; route endpoints/date feasibility are not validated by this criterion."
    if p == "prefer consistent airline brand along all flights":
        return "operational", {"op": "same_companies"}, "All normalized company strings match. A direct single-company flight satisfies consistency; the conflicting generation-template layover instruction is not enforced. No cross-name airline alias resolution."
    if p == "prefer different airline brand along different flights":
        return "operational", {"op": "distinct_companies"}, "At least two flight segments with distinct normalized company strings; no cross-name airline alias resolution."
    if p == "prefer specific hotel brand (user specifies)":
        return "operational", {"op": "text_words", "fields": ["name"], "value": "Hayatt"}, "Preserve the source's spelling Hayatt, which appears in the user's statement; no external brand correction."
    if p == "prefer having city or ocean or mountain view (user specifies)":
        return "operational", {"op": "list_contains", "field": "amenities", "value": "City View"}, "The implicit statement specifically names a city view."
    raise ValueError("Unmapped preference: " + p)


def main():
    source = ROOT / "results/userbench_train_e025/preference_inventory.json"
    rows = json.loads(source.read_text())
    result = []
    for row in rows:
        status, rule, reason = criterion(row)
        result.append(dict(row, criterion_id=row["case_id"] + "/" + row["aspect"] + "/" + str(row["index"]),
                           status=status, rule=rule, interpretation=reason,
                           semantic_validation="single-assistant provisional operationalization"))
    assert len(result) == 96
    content = (json.dumps(result, ensure_ascii=False, sort_keys=True, indent=2) + "\n").encode()
    dest = OUT / "criteria.json"
    with dest.open("xb") as f:
        f.write(content)
    lock = {"criteria_sha256": hashlib.sha256(content).hexdigest(),
            "preference_inventory_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
            "builder_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            "phase": "catalogue fixed before option truth-table and mask computation"}
    with (OUT / "criteria_lock.json").open("x") as f:
        json.dump(lock, f, indent=2); f.write("\n")
    print(json.dumps(lock, indent=2))


if __name__ == "__main__":
    main()
