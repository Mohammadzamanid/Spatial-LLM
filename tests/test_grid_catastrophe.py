"""
tests/test_grid_catastrophe.py
Locks the catastrophic-error (Fiete) result: adding grid modules suppresses catastrophic errors
exponentially at ~constant local precision; the error law is bimodal; and at matched budget the grid code
is far finer than a place code and no more catastrophe-prone.
"""
import pytest

from src.eval.grid_catastrophe import run_seed, KS, NOISES, CAT


@pytest.fixture(scope="module")
def seed0():
    return run_seed(0)


def test_modules_suppress_catastrophes(seed0):
    A = seed0["A"]
    assert A[KS[0]]["cat"] > 0.4, f"few modules (K={KS[0]}) should be catastrophe-prone, got {A[KS[0]]['cat']:.2f}"
    assert A[KS[-1]]["cat"] < 0.05, f"many modules (K={KS[-1]}) should suppress catastrophes, got {A[KS[-1]]['cat']:.2f}"
    assert A[KS[-1]]["cat"] < A[KS[0]]["cat"]                 # monotone suppression endpoints
    # local precision barely changes once the code is unambiguous (K>=3): not the source of the K=2 error
    assert A[5]["median"] < 0.02 and A[6]["median"] < 0.02


def test_error_distribution_is_bimodal(seed0):
    hist = seed0["hist"]; nloc = int(CAT * len(hist[2]))     # bins below the catastrophic threshold
    k2_cat = 1 - sum(hist[2][:nloc]); k5_cat = 1 - sum(hist[5][:nloc])
    assert k2_cat > 0.3, f"K=2 should have a large catastrophic tail, got {k2_cat:.2f}"
    assert k5_cat < 0.1, f"K=5 should have almost no catastrophic tail, got {k5_cat:.2f}"


def test_grid_beats_place_at_matched_budget(seed0):
    hi = seed0["C"][NOISES[-1]]
    assert hi["grid_med"] < 0.3 * hi["place_med"], "grid should be much finer than place at matched budget"
    assert hi["grid_cat"] <= hi["place_cat"] + 0.02, "grid should be no more catastrophe-prone than place"
