"""Behavioral checks for evaluation leakage, invalid choices and stopping."""
import copy
import unittest

from userbench_choice_adapter import prepare_case, score_choices


def fixture():
    return {"dimensions": ["hotel"], "hotel": {
        "preferences": [["hotel", "amenities", "hidden workspace need"]],
        "correct_ids": ["H8", "H2"], "best_id": "H8",
        "options": {
            "correct": [
                {"id": "H8", "type": "suitable", "reason": "secret reason",
                 "name": "A", "room": [{"type": "Suite", "capacity": 2}]},
                {"id": "H2", "type": "suitable", "name": "B"}],
            "wrong": [{"id": "H3", "type": "not-suitable", "name": "C"}],
            "noise": []}}}


class ChoiceAdapterTests(unittest.TestCase):
    def setUp(self):
        self.task = fixture()
        self.public, self.gold = prepare_case(self.task, "I need a hotel", "fixture")
        self.ids = self.gold["aspects"]["hotel"]["source_to_visible"]

    def test_hidden_metadata_cannot_change_visible_input(self):
        changed = copy.deepcopy(self.task)
        changed["hotel"]["preferences"] = [["new secret"]]
        changed["hotel"]["best_id"] = "H2"
        changed["hotel"]["options"]["correct"][0]["reason"] = "different secret"
        pub, gold = prepare_case(changed, "I need a hotel", "fixture")
        self.assertEqual(pub, self.public)
        self.assertNotEqual(gold, self.gold)

    def test_category_order_and_original_order_cannot_leak_labels(self):
        changed = copy.deepcopy(self.task)
        changed["hotel"]["options"]["correct"].reverse()
        changed["hotel"]["options"] = dict(reversed(list(changed["hotel"]["options"].items())))
        self.assertEqual(prepare_case(changed, "I need a hotel", "fixture")[0], self.public)
        # Move an option between correct/wrong and update gold consistently.
        data = changed["hotel"]
        moved = data["options"]["correct"].pop(0)
        data["options"]["wrong"].append(moved)
        data["correct_ids"].remove(moved["id"])
        self.assertEqual(prepare_case(changed, "I need a hotel", "fixture")[0], self.public)

    def test_nested_room_type_survives_sanitization(self):
        room_option = next(o for o in self.public["options"]["hotel"] if "room" in o)
        self.assertEqual(room_option["room"][0]["type"], "Suite")
        self.assertNotIn("type", room_option)
        self.assertNotIn("reason", room_option)

    def test_best_and_acceptable_are_distinct(self):
        result = score_choices(self.gold, [("hotel", self.ids["H2"])])
        self.assertEqual(result["acceptable_rate"], 1)
        self.assertEqual(result["best_rate"], 0)
        self.assertTrue(result["complete"])

    def test_wrong_then_best_cannot_overwrite_first_choice(self):
        result = score_choices(self.gold, [("hotel", self.ids["H3"]), ("hotel", self.ids["H8"])])
        self.assertEqual(result["acceptable_count"], 0)
        self.assertTrue(result["complete"])
        self.assertEqual(result["rejected"][0]["reason"], "already_locked")

    def test_invalid_first_choice_locks_and_fails(self):
        result = score_choices(self.gold, [("hotel", "H8"), ("hotel", self.ids["H8"])])
        self.assertEqual(result["best_rate"], 0)
        self.assertFalse(result["aspects"][0]["valid"])
        self.assertTrue(result["complete"])

    def test_missing_and_unknown_aspects_fail(self):
        result = score_choices(self.gold, [("flight", "O01")])
        self.assertFalse(result["complete"])
        self.assertEqual(result["acceptable_rate"], 0)
        self.assertFalse(result["whole_task_acceptable"])

    def test_perfect_one_choice_completes_with_no_extra_rewards(self):
        result = score_choices(self.gold, [("hotel", self.ids["H8"])])
        self.assertTrue(result["complete"])
        self.assertEqual(result["best_rate"], 1)
        self.assertTrue(result["whole_task_acceptable"])
        self.assertNotIn("reward", result)

    def test_duplicate_source_ids_rejected(self):
        self.task["hotel"]["options"]["wrong"][0]["id"] = "H8"
        with self.assertRaises(ValueError):
            prepare_case(self.task, "hotel", "fixture")


if __name__ == "__main__":
    unittest.main()
