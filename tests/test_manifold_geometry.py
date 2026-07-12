"""Tests for the manifold-deformation follow-up to grid shearing (GAPS.md #5d follow-up).

Beyond the rate-map shearing of #5d, the NEURAL MANIFOLD of the standard continuous attractor should NOT deform:
trapezoid population codes lie on the same manifold as the square's (a rigid torus; Gardner 2022), so #5d is a
warping of the space->manifold MAP. In a non-Euclidean (barrier) environment the fixed grid ignores the wall
while a PLASTIC code reshapes to the geodesic geometry -- manifold deformation requires plasticity.
"""
from src.eval.manifold_geometry import run_seed


def test_rigid_manifold_but_plastic_deforms():
    o = run_seed(0)

    # (A) the rigid CAN manifold does NOT deform: trapezoid overlaps the square manifold ~ the reference does
    assert o["manifold_deformation_grid"] < 0.1, "the grid manifold is a rigid torus (#5d is a map effect)"

    # (B) the fixed grid ignores the barrier; a plastic code deforms to its geodesic geometry
    assert o["barrier_respect_grid"] < 0.1, "the fixed grid manifold ignores the wall (Euclidean, not geodesic)"
    assert o["barrier_respect_plastic"] > o["barrier_respect_grid"] + 0.1, "a plastic code reshapes to the wall"
    assert o["barrier_respect_plastic"] > 0.15, "the plastic manifold clearly respects the geodesic geometry"
