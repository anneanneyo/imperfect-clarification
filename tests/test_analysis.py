from imperfect_clarification import count_field_presence, diagnostic_percentages


def test_count_field_presence_keeps_missing_and_null_separate() -> None:
    records = [{}, {"fee": None}, {"fee": 0}, {"fee": 4}]
    assert count_field_presence(records, "fee") == {
        "absent": 1,
        "explicit_null": 1,
        "numeric_zero": 1,
    }


def test_diagnostic_percentages_converts_stored_ratios() -> None:
    assert diagnostic_percentages({"acceptable": "5/8", "best": "1/16"}) == {
        "acceptable": 62.5,
        "best": 6.25,
    }
