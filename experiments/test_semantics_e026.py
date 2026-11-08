"""Edge-case semantics and exact interval tests; no source-label fitting."""
from fractions import Fraction as F
import itertools
import unittest

from semantics_e026 import evaluate, conjunction
from run_semantics_e026 import bounds, QUESTIONS


def ev(rule, option, status="operational"):
    return evaluate({"status": status, "rule": rule}, option)


class SemanticChecks(unittest.TestCase):
    def test_missing_null_zero_and_bool_fee(self):
        rule = {"op": "service_available", "path": ["service", "fee"]}
        self.assertEqual([ev(rule, x) for x in [{}, {"service": {}}, {"service": {"fee": None}},
                                               {"service": {"fee": 0}}, {"service": {"fee": False}}]],
                         ["U", "U", "F", "T", "U"])

    def test_complete_list_and_normalization(self):
        rule = {"op": "list_contains", "field": "tags", "value": "Pet-Friendly"}
        self.assertEqual(ev(rule, {}), "U")
        self.assertEqual(ev(rule, {"tags": []}), "F")
        self.assertEqual(ev(rule, {"tags": ["  PET  friendly  "]}), "T")
        self.assertEqual(ev(rule, {"tags": [None]}), "U")

    def test_review_majority_is_strict_and_denominator_required(self):
        rule = {"op": "review_majority"}
        reviews = {str(i) + "_star": 0 for i in range(1, 6)}
        self.assertEqual(ev(rule, {"reviews": reviews}), "U")
        reviews.update({"5_star": 5, "4_star": 5})
        self.assertEqual(ev(rule, {"reviews": reviews}), "F")
        reviews["5_star"] = 6
        self.assertEqual(ev(rule, {"reviews": reviews}), "T")
        del reviews["1_star"]
        self.assertEqual(ev(rule, {"reviews": reviews}), "U")

    def test_layover_boundaries_and_malformed_timing(self):
        rule = {"op": "layover_range", "minimum": 4, "maximum": 5, "connections": 1}
        for hours, answer in [(3.99, "F"), (4, "T"), (5, "T"), (5.01, "F")]:
            self.assertEqual(ev(rule, {"path": ["A", "B", "C"], "time": [2, hours, 3]}), answer)
        self.assertEqual(ev(rule, {"path": ["A", "B", "C"], "time": [4, 5]}), "U")

    def test_room_types_use_existential_evidence(self):
        rule = {"op": "room_type", "value": "Suite (with kitchen)"}
        self.assertEqual(ev(rule, {"room": [{"type": "Suite (with kitchen)"}, {}]}), "T")
        self.assertEqual(ev(rule, {"room": [{"type": "King"}, {}]}), "U")
        self.assertEqual(ev(rule, {"room": [{"type": "King"}]}), "F")

    def test_bedroom_count_not_capacity(self):
        rule = {"op": "bedrooms_min", "minimum": 3}
        self.assertEqual(ev(rule, {"room": "2B3B", "capacity": 8}), "F")
        self.assertEqual(ev(rule, {"room": "3B1B"}), "T")
        self.assertEqual(ev(rule, {"room": "Penthouse", "capacity": 10}), "U")

    def test_user_specific_model_phrase_needs_word_boundary(self):
        rule = {"op": "text_words", "fields": ["brand", "model"], "value": "Tesla Model 3"}
        self.assertEqual(ev(rule, {"brand": "Avis", "model": "Tesla Model 3"}), "T")
        self.assertEqual(ev(rule, {"brand": "Tesla", "model": "Model 30"}), "F")

    def test_direct_flight_can_have_consistent_airline(self):
        option = {"path": ["A", "B"], "company": ["Carrier"]}
        self.assertEqual(ev({"op": "direct"}, option), "T")
        self.assertEqual(ev({"op": "same_companies"}, option), "T")
        self.assertEqual(ev({"op": "distinct_companies"}, option), "F")

    def test_unresolved_criterion_and_conjunction(self):
        self.assertEqual(ev(None, {"cost": 10}, status="unsupported"), "U")
        self.assertEqual(conjunction(["T", "U"]), "U")
        self.assertEqual(conjunction(["F", "U"]), "F")
        self.assertEqual(conjunction(["T", "T"]), "T")

    def test_coupled_interval_keeps_shared_unknown(self):
        self.assertEqual(bounds([F(1, 2), F(-1, 2)], ["T", "U"]), {"lower": "0", "upper": "1/2"})
        self.assertEqual(bounds([F(1, 6), F(1, 6), F(-1, 3)], ["T", "U", "F"]), {"lower": "1/6", "upper": "1/3"})

    def test_interval_matches_all_binary_completions(self):
        coefficients = [F(1, 4), F(-1, 6), F(1, 3), F(-5, 12)]
        for outcomes in itertools.product("TFU", repeat=4):
            completions = itertools.product(*([0, 1] if x == "U" else [int(x == "T")] for x in outcomes))
            values = [sum(c * v for c, v in zip(coefficients, z)) for z in completions]
            result = bounds(coefficients, outcomes)
            self.assertEqual(F(result["lower"]), min(values))
            self.assertEqual(F(result["upper"]), max(values))

    def test_question_menu_has_no_active_hidden_categories(self):
        self.assertEqual(set(QUESTIONS), {"flight", "lodging", "rental_car", "restaurant"})
        self.assertTrue(all("requirements" in text for text in QUESTIONS.values()))
        self.assertTrue(all(word not in " ".join(QUESTIONS.values()).lower()
                            for word in ["wifi", "insurance", "cuisine", "kitchen"]))


if __name__ == "__main__":
    unittest.main()
