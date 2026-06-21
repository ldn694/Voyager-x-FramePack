# SPDX-License-Identifier: Apache-2.0
"""
Pure-Python expansion of the dual-branch cross-attention schedule.

The dual-branch forward (``models.py``, the ``use_second_branch`` path) walks a
``scheduler`` of ``(u, v)`` edges, interleaving: run first-branch blocks up to an
index, run second-branch blocks up to an index, then one cross-attention block.
This module reproduces *exactly* that index bookkeeping as a list of plain dicts so
the JVP forward can drive its loop from it — and so the off-by-one-prone math can be
unit-tested with no torch / no GPU.

Edge semantics (mirrors models.py):
  * ``(-1, -1)`` is the appended terminator: expands to "run all remaining blocks of
    both branches, no cross-attention" (``last_layer``). The model appends this once
    at construction, so ``model.double_branch_scheduler`` already contains it.
  * Otherwise, ``u > 0`` ⇒ first branch is the cross-attention query (``"first_q"``);
    ``u <= 0`` ⇒ second branch is the query (``"second_q"``). The first/second block
    indices come from the signed ``(u, v)`` pair as below.
"""

from typing import List, Dict, Sequence, Tuple


def expand_double_branch_schedule(
    scheduler: Sequence[Tuple[int, int]],
    n_double: int,
    n_single: int,
    n_second: int,
) -> List[Dict]:
    """Expand ``scheduler`` into per-step block ranges + cross-attention direction.

    Args:
        scheduler: the model's ``double_branch_scheduler`` (already including the
            trailing ``(-1, -1)`` terminator).
        n_double / n_single: first-branch double- / single-stream block counts.
        n_second: second-branch block count.

    Returns a list of dicts, one per edge, each with:
        ``first_range`` : list[int] — first-branch global block indices to run
                          (``< n_double`` = double block, else single block
                          ``idx - n_double``).
        ``second_range``: list[int] — second-branch block indices to run.
        ``cross``       : ``"first_q"`` | ``"second_q"`` | ``None`` (no cross-attn).
        ``last_layer``  : bool.
        ``cross_attn_id``: int — index into ``model.cross_attn_blocks``.
    """
    prev_first = -1
    prev_second = -1
    plan: List[Dict] = []

    for cross_attn_id, edge in enumerate(scheduler):
        u, v = edge
        orig_u = u
        if u == -1 and v == -1:
            # terminator: run everything left, no cross-attention
            u = n_double + n_single - 1
            v = -n_second + 1
            last_layer = True
        else:
            last_layer = False

        if u > 0:
            first_idx = u
            second_idx = -v
        else:
            first_idx = v
            second_idx = -u

        first_range = list(range(prev_first + 1, first_idx + 1))
        second_range = list(range(prev_second + 1, second_idx + 1))

        if last_layer:
            cross = None
        elif orig_u > 0:
            cross = "first_q"
        else:
            cross = "second_q"

        plan.append({
            "cross_attn_id": cross_attn_id,
            "first_range": first_range,
            "second_range": second_range,
            "cross": cross,
            "last_layer": last_layer,
        })
        prev_first = first_idx
        prev_second = second_idx

    return plan
