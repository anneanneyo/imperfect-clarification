from fractions import Fraction as F
import unittest
from executors_e027 import payload_for, messages_for, parse_choice, lexical_distribution, score_bounds, full_tie_gap


class ExecutorChecks(unittest.TestCase):
    def test_public_payload_ignores_unrequested_hidden_fields(self):
        case = {"case_id": "C", "user_message": "Need a car", "options": {"rental_car": [{"id": "O01", "cost": 10}]},
                "preferences": ["secret"], "best_id": "secret"}
        payload = payload_for(case, "rental_car", ["The reply actually revealed"])
        self.assertEqual(set(payload), {"trip_request", "target_aspect", "additional_statements", "options"})
        self.assertNotIn("secret", str(messages_for(payload)))
        self.assertEqual(payload["additional_statements"], ["The reply actually revealed"])

    def test_parser_accepts_only_exact_object_with_listed_id(self):
        self.assertEqual(parse_choice(' {"id":"O02"} ', ["O01", "O02"]), "O02")
        fenced = chr(96) * 3 + 'json\n{"id":"O01"}\n' + chr(96) * 3
        for text in ['{"id":"O03"}', '{"id":"O01","id":"O02"}', '{"id":"O01","reason":"x"}',
                     '["O01"]', '{"id":1}', 'Here: {"id":"O01"}', fenced]:
            self.assertIsNone(parse_choice(text, ["O01", "O02"]))

    def test_reply_overlap_precedes_price(self):
        payload = {"trip_request": "", "target_aspect": "rental_car", "additional_statements": ["Electric"],
                   "options": [{"id": "O01", "categories": "Electric", "cost": 200},
                               {"id": "O02", "categories": "Gasoline", "cost": 20}]}
        distribution, _ = lexical_distribution(payload)
        self.assertEqual(distribution, {"O01": "1", "O02": "0"})

    def test_tie_is_uniform_and_independent_of_ids(self):
        payload = {"trip_request": "", "target_aspect": "rental_car", "additional_statements": [],
                   "options": [{"id": "O01", "cost": 10}, {"id": "O02", "cost": 10}]}
        self.assertEqual(lexical_distribution(payload)[0], {"O01": "1/2", "O02": "1/2"})
        payload["options"].reverse()
        self.assertEqual(lexical_distribution(payload)[0], {"O01": "1/2", "O02": "1/2"})

    def test_missing_prices_are_tied_not_crash(self):
        p = {"trip_request": "", "target_aspect": "hotel", "additional_statements": [],
             "options": [{"id": "O01"}, {"id": "O02", "cost": None}]}
        d, trace = lexical_distribution(p)
        self.assertEqual(d, {"O01": "1/2", "O02": "1/2"})
        self.assertTrue(all(r["base_price"] is None for r in trace))

    def test_unstated_preferences_do_not_enter_overlap(self):
        p = {"trip_request": "", "target_aspect": "rental_car", "additional_statements": [],
             "options": [{"id": "O01", "categories": "Electric", "cost": 100}, {"id": "O02", "categories": "SUV", "cost": 10}]}
        self.assertEqual(lexical_distribution(p)[0]["O02"], "1")

    def test_invalid_mass_scores_zero(self):
        self.assertEqual(score_bounds({"O01": "0"}, {"O01": "T"}), {"lower": "0", "upper": "0"})

    def test_answer_can_harm_and_unknown_is_coupled(self):
        self.assertEqual(score_bounds({"O01": "-1", "O02": "1"}, {"O01": "T", "O02": "F"}, F(1, 4)),
                         {"lower": "-1/4", "upper": "-1/4"})
        self.assertEqual(score_bounds({"O01": "-1", "O02": "1"}, {"O01": "U", "O02": "T"}, F(1, 4)),
                         {"lower": "0", "upper": "1/4"})
        self.assertEqual(score_bounds({"O01": "0"}, {"O01": "U"}), {"lower": "0", "upper": "0"})

    def test_full_mask_ties_cannot_create_spurious_gap(self):
        result = full_tie_gap({"flight": "1/4", "car": "1/4"}, {"flight": "0", "car": "1/4"})
        self.assertEqual(result["gap"], "0")
        self.assertEqual(result["full_maximizers"], ["car", "flight"])

    def test_strict_source_gap_requires_no_full_maximizer_can_match(self):
        result = full_tie_gap({"flight": "1/4", "car": "0"}, {"flight": "0", "car": "1/4"})
        self.assertEqual(result, {"full_maximizers": ["flight"], "gap": "1/4"})


if __name__ == "__main__":
    unittest.main()
