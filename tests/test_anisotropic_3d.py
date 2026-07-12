"""Tests for anisotropic 3-D coding (GAPS.md isotropic-3D-lattice item).

Vertical field elongation + impaired vertical odometry (Hayman 2011) EMERGE from gravity-biased experience under
ISOTROPIC hardware (isotropic code noise, isotropic init, one shared power budget). A capacity-limited code
allocates resolution to well-experienced axes (rate-distortion/water-filling); the low-experience vertical axis is
disproportionately under-coded. Falsifier: isotropic experience -> isotropic code. Nothing about the anisotropy is
imposed.
"""
from src.eval.anisotropic_3d import run_seed


def test_vertical_anisotropy_emerges_from_experience_not_hardware():
    o = run_seed(0)

    # (A) with gravity-biased experience, vertical resolution is disproportionately coarse (normalized error)
    assert o["ratio_terr"] > 1.8, "vertical normalized error >> horizontal -> elongated vertical fields (Hayman)"
    assert o["vert_norm"] > o["horiz_norm"], "vertical axis is the coarse one"

    # (B) falsifier: the SAME isotropic hardware with isotropic experience gives an isotropic code
    assert o["ratio_iso"] < 1.3, "isotropic experience -> isotropic code (the anisotropy is experience, not hardware)"
    assert o["ratio_terr"] > o["ratio_iso"] + 0.8, "gravity-biased experience is far more anisotropic than isotropic"

    # (C) dose-response: anisotropy grows monotonically as vertical experience shrinks
    assert o["dose_10"] < o["dose_06"] < o["dose_03"] < o["dose_015"], "anisotropy tracks the vertical deficit"
    assert o["dose_10"] < 1.3, "with equal experience there is no anisotropy"

    # (D) honesty: absolute vertical error is SMALL (small range) — only the normalized measure reveals the loss
    assert o["vert_abs"] < o["horiz_abs"], "absolute vertical error is small (little range), unlike the normalized"
