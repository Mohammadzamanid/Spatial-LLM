"""
tests/test_btsp.py
Locks the BTSP organ + its emergent signatures (Bittner, Milstein & Magee 2017):
(1) the asymmetric seconds-wide kernel is finite even at millisecond tau (the 0*inf guard);
(2) one plateau imprints a one-shot field ONLY with a seconds-scale kernel (STDP-scale imprints ~nothing);
(3) the field shifts UPSTREAM of the plateau ONLY with the asymmetric kernel (predictive), symmetric ~ centred;
(4) the predictive shift SCALES WITH running speed.
"""
import torch

from src.models.neuro import BTSPPlasticity
from src.eval.btsp import run_seed, run_once, X_STAR, SPEEDS


def test_kernel_is_finite_at_millisecond_tau():
    # the 0*inf bug: masked-out side must not overflow. A millisecond-scale kernel evaluated far from the
    # plateau must stay finite.
    k = BTSPPlasticity(0.02, 0.02, symmetric=True)
    dt = torch.linspace(-8, 8, 200)
    assert torch.isfinite(k.kernel(dt)).all()


def test_kernel_asymmetry_favours_the_past():
    # the biological asymmetry: an input the SAME lag before the plateau is potentiated more than after it.
    k = BTSPPlasticity(tau_pre=1.3, tau_post=0.55)
    before = k.kernel(torch.tensor([-0.8])).item()   # 0.8 s before the plateau
    after = k.kernel(torch.tensor([0.8])).item()      # 0.8 s after
    assert before > after > 0


def test_signatures():
    r = run_seed(0)
    c = r["conditions"]
    # (A) one-shot field needs a SECONDS-scale kernel; the millisecond STDP kernel imprints almost nothing
    assert c["stdp"]["strength"] < 0.2 * c["btsp"]["strength"]
    # (B) the PREDICTIVE (upstream, negative) shift needs the ASYMMETRY
    assert c["btsp"]["shift"] < -3.0, f"BTSP field should shift upstream (got {c['btsp']['shift']:.2f})"
    assert abs(c["symmetric"]["shift"]) < 2.5, "symmetric kernel should sit on the plateau"
    # (C) the shift scales with running speed (faster -> larger upstream shift)
    ss = r["speed_shift"]
    assert ss[SPEEDS[-1]] < ss[SPEEDS[0]] - 2.0, "predictive shift should grow with speed"


def test_one_shot_from_a_single_plateau():
    # a single traversal with one plateau yields a real field (not a control with no plateau signal).
    import numpy as np  # noqa: F401  (ensure numpy import path is fine)
    gen = torch.Generator().manual_seed(1)
    pref = torch.linspace(0, 300.0, 151)
    k = BTSPPlasticity(1.3, 0.55)
    strength, shift, width = run_once(k, 25.0, pref, gen)
    assert strength > 0 and width > 5.0, "one plateau should imprint a coherent field in one pass"
