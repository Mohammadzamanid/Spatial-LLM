"""Tests for the unified agent grounded on the real grid cortex (GAPS.md integration capstone, grounded).

The same emergent survival policy, but position is decoded from the real drifting grid code and uncertainty is
the real #7 reconstruction residual. The three POSITION organs (grid, uncertainty read-out, landmark
relocalisation) should each be load-bearing on the real cortex, and the emergent uncertainty x landmark
complementarity should survive grounding (uncertainty helps only WITH landmarks). The interoceptive drive organ
is validated separately in #4; here it barely moves survival, so it is not asserted.
"""
from src.eval.unified_agent_cortex import run_seed


def test_position_organs_cohere_on_the_real_cortex():
    o = run_seed(0)
    intact = o["drive_intact"]

    # the three position organs are each load-bearing when driven by the real cortex
    assert o["drive_no_grid"] > intact + 10, "scrambling the real decode -> can't navigate (catastrophic)"
    assert o["drive_no_uncertainty"] > intact + 4, "ignoring the real residual -> can't tell when it's lost"
    assert o["drive_no_landmark"] > intact + 4, "blocking re-anchoring -> can't undo the real drift"

    # the emergent complementarity survives grounding: the real uncertainty organ only helps WITH landmarks
    assert o["cost_unc_with_lm"] > o["cost_unc_without_lm"] + 3, "uncertainty helps more when landmarks are present"
    assert o["cost_unc_without_lm"] < 4, "knowing you're lost is worthless if you cannot re-anchor"
