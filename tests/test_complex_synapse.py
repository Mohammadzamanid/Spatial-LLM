"""Tests for the multi-timescale (Benna-Fusi) synapse (GAPS.md Tier 5, #B2).

A complex synapse (a chain of coupled variables at geometric timescales) should forget as a POWER LAW (slope
≈ -0.5, straight on log-log), while a leaky scalar synapse forgets EXPONENTIALLY; the complex synapse's memory
lifetime is far longer at matched initial SNR and grows with chain depth. All measured, never fit into a loss.
"""
from src.eval.complex_synapse import run_seed


def test_powerlaw_vs_exponential_forgetting():
    o = run_seed(0)

    # (A) SHAPE: the Benna-Fusi synapse is fit far better by a power law than an exponential; the leaky scalar
    # is fit far better by an exponential than a power law — the forgetting shapes are opposite.
    assert o["powerlaw_margin_bf"] > 0.1, "Benna-Fusi forgetting should be power-law (log-log R² >> semilog R²)"
    assert o["exp_margin_scalar"] > 0.05, "leaky scalar forgetting should be exponential (semilog R² >> log-log R²)"

    # the measured power-law slope is ~ -0.5 (the 1/sqrt(t) law) — emergent from the diffusion, not imposed
    assert -0.65 < o["bf_slope"] < -0.30, "power-law slope should be near -0.5"

    # (B) LIFETIME: at matched initial SNR the complex synapse remembers much longer than the scalar
    assert o["lifetime_ratio"] > 1.8, "Benna-Fusi memory lifetime should far exceed the scalar's"

    # (C) DOSE-RESPONSE: memory lifetime grows with chain depth (a deep chain >> a shallow one)
    assert o["lifetime_N7"] > o["lifetime_N3"] + 30, "lifetime should grow with the number of beakers"
