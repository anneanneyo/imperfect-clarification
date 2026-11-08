"""Fixed-menu diagnostic adapter, NOT an execution of original UserBench.

Only the public object returned by prepare_case may be passed to a policy.
The private object is for a separate evaluator. No model clients are imported.
"""
from __future__ import annotations

import copy
import hashlib


def digest(text):
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def prepare_case(task, user_message, case_id, seed="E025-v1"):
    public = {"case_id": case_id, "user_message": user_message,
              "aspects": sorted(task["dimensions"]), "options": {}}
    private = {"case_id": case_id, "aspects": {}}
    for aspect in public["aspects"]:
        data = task[aspect]
        # Flatten, then sort independently of source category and source list order.
        options = [o for group in data["options"].values() for o in group]
        ids = [o["id"] for o in options]
        if len(ids) != len(set(ids)):
            raise ValueError("Duplicate source option ID")
        if set(data["correct_ids"]) != {o["id"] for o in data["options"]["correct"]}:
            raise ValueError("Correct IDs do not match source option group")
        if data["best_id"] not in data["correct_ids"]:
            raise ValueError("Best ID is not acceptable")
        scope = seed + "|" + user_message + "|" + aspect + "|"
        options = sorted(options, key=lambda o: (digest(scope + o["id"]), o["id"]))
        forward = {o["id"]: "O%02d" % (j + 1) for j, o in enumerate(options)}
        visible = []
        for option in options:
            # Nested room.type is a real room attribute; remove only top-level type.
            clean = copy.deepcopy({k: v for k, v in option.items()
                                   if k not in {"id", "type", "reason"}})
            clean["id"] = forward[option["id"]]
            visible.append(clean)
        # Separate ID remapping from display-order randomization.
        visible.sort(key=lambda o: (digest(scope + "display|" + o["id"]), o["id"]))
        public["options"][aspect] = visible
        private["aspects"][aspect] = {
            "valid_ids": sorted(forward.values()),
            "acceptable_ids": sorted(forward[i] for i in data["correct_ids"]),
            "best_id": forward[data["best_id"]],
            "source_to_visible": forward,
            "preferences": copy.deepcopy(data["preferences"]),
        }
    return public, private


def score_choices(private, submissions):
    """Score a list of (aspect, ID) attempts; the first locks that aspect.

    Missing/invalid selections score zero. Duplicate submissions never overwrite
    the first. Completion means all required aspects have a first submission,
    regardless of correctness. This is our explicit terminal contract.
    """
    accepted, rejected = {}, []
    for aspect, option_id in submissions:
        if aspect not in private["aspects"]:
            rejected.append({"aspect": aspect, "id": option_id, "reason": "unknown_aspect"})
        elif aspect in accepted:
            rejected.append({"aspect": aspect, "id": option_id, "reason": "already_locked"})
        else:
            accepted[aspect] = option_id
    rows = []
    for aspect, gold in private["aspects"].items():
        chosen = accepted.get(aspect)
        rows.append({"aspect": aspect, "chosen_id": chosen,
                     "submitted": aspect in accepted,
                     "valid": chosen in gold["valid_ids"],
                     "best": chosen == gold["best_id"],
                     "acceptable": chosen in gold["acceptable_ids"]})
    n = len(rows)
    if n == 0:
        raise ValueError("Empty task")
    best = sum(r["best"] for r in rows)
    acceptable = sum(r["acceptable"] for r in rows)
    return {"aspects": rows, "rejected": rejected,
            "complete": len(accepted) == n, "aspect_count": n,
            "best_count": best, "acceptable_count": acceptable,
            "best_rate": best / n, "acceptable_rate": acceptable / n,
            "whole_task_best": best == n, "whole_task_acceptable": acceptable == n}
