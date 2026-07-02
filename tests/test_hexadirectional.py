"""
tests/test_hexadirectional.py
Locks GAPS.md #2 (the hexadirectional signal / grid code for concepts): the model's HEXAGONAL grid produces a
6-fold direction signal that (a) sticks out above the 4-fold and the adjacent 5/7-fold control, (b) FLIPS to
4-fold for a SQUARE lattice (symmetry inherited, not imposed), and (c) collapses for a LINEAR read-out
(nonlinearity necessary). Reduced sampling for test speed.
"""
import src.eval.hexadirectional as HX


def test_hexadirectional_emerges_and_inherits_lattice_symmetry(monkeypatch):
    monkeypatch.setattr(HX, "N_RUNS", 24)
    monkeypatch.setattr(HX, "K", 60)
    r = HX.run_seed(0)
    # (a) hex grid -> 6-fold dominant, above the 4-fold AND the adjacent 5/7-fold control
    assert r["hex_a6"] > 1.8 * r["hex_a4"], "hex A6 should exceed A4"
    assert r["hex_a6"] > 1.8 * r["hex_adj"], "hex A6 should exceed the adjacent 5/7-fold control"
    assert r["index_hex"] > 0.60, "hex 6-fold index should be clearly hexadirectional"
    # (b) square lattice -> 4-fold (symmetry inherited from the lattice, not imposed)
    assert r["index_square"] < 0.40, "square lattice should flip to 4-fold"
    # (c) linear read-out -> direction-invariant (nonlinearity necessary)
    assert r["lin_a6"] < 0.5 * r["hex_a6"], "a linear read-out should carry far less 6-fold signal"
