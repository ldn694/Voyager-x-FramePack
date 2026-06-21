# SPDX-License-Identifier: Apache-2.0
"""
Pure-Python (no-torch) test for the dual-branch schedule expansion.

Validates ``expand_double_branch_schedule`` against the real
``HYVideo-T/2-2branch-cross_attn`` schedule: every block of both branches is visited
exactly once, ranges are contiguous and non-overlapping, and the cross-attention
directions match. Runs anywhere (no torch / GPU):

    python -m voyager.modules.jvp.test_jvp_schedule
"""

from voyager.modules.jvp.jvp_schedule import expand_double_branch_schedule

# HYVideo-T/2-2branch-cross_attn (models.py:1829), with the (-1,-1) terminator the
# model appends at construction.
SCHEDULER = [
    (0, 3), (-3, 11), (15, -4), (-5, 19), (-6, 23), (-9, 31),
    (35, -10), (-11, 39), (-12, 43), (-15, 51), (55, -16), (-17, 59),
    (-1, -1),
]
N_DOUBLE, N_SINGLE, N_SECOND = 20, 40, 18

EXPECTED_CROSS = [
    "second_q", "second_q", "first_q", "second_q", "second_q", "second_q",
    "first_q", "second_q", "second_q", "second_q", "first_q", "second_q",
    None,  # terminator
]


def test_schedule_expansion():
    plan = expand_double_branch_schedule(SCHEDULER, N_DOUBLE, N_SINGLE, N_SECOND)
    assert len(plan) == len(SCHEDULER)

    # cross-attention directions
    assert [p["cross"] for p in plan] == EXPECTED_CROSS, \
        [p["cross"] for p in plan]

    # only the terminator is last_layer
    assert [p["last_layer"] for p in plan] == [False] * 12 + [True]

    # first branch: every index 0..59 visited exactly once, in order
    first_visited = [i for p in plan for i in p["first_range"]]
    assert first_visited == list(range(N_DOUBLE + N_SINGLE)), first_visited

    # second branch: every index 0..17 visited exactly once, in order
    second_visited = [i for p in plan for i in p["second_range"]]
    assert second_visited == list(range(N_SECOND)), second_visited

    # terminator runs no new blocks and no cross-attn
    assert plan[-1]["first_range"] == [] and plan[-1]["second_range"] == []
    assert plan[-1]["cross"] is None

    # cross_attn_id is the enumerate index
    assert [p["cross_attn_id"] for p in plan] == list(range(len(SCHEDULER)))

    print(f"[ok] schedule expansion: {N_DOUBLE + N_SINGLE} first-branch + "
          f"{N_SECOND} second-branch blocks each visited once; "
          f"{EXPECTED_CROSS.count('first_q')} first_q + "
          f"{EXPECTED_CROSS.count('second_q')} second_q cross-attn steps")


def test_unidirectional_schedule():
    # HYVideo-T/2-2branch-cross_attn-unidirectional-q_second (models.py:1859):
    # all cross-attn edges have u <= 0 -> always second_q.
    sched = [
        (0, 3), (-3, 11), (-5, 19), (-6, 23), (-9, 31),
        (-11, 39), (-12, 43), (-15, 51), (-17, 59), (-1, -1),
    ]
    plan = expand_double_branch_schedule(sched, N_DOUBLE, N_SINGLE, N_SECOND)
    crosses = [p["cross"] for p in plan]
    assert crosses == ["second_q"] * 9 + [None], crosses
    assert [i for p in plan for i in p["first_range"]] == list(range(60))
    assert [i for p in plan for i in p["second_range"]] == list(range(18))
    print("[ok] unidirectional schedule: all second_q, full coverage")


if __name__ == "__main__":
    test_schedule_expansion()
    test_unidirectional_schedule()
    print("\nAll schedule-expansion checks passed.")
