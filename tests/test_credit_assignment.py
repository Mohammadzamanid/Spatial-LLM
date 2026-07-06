"""Tests for deep credit assignment without backprop (GAPS.md Tier 5, #A1).

Feedback alignment (a fixed RANDOM backward pathway — no weight transport, no forward/backward symmetry)
should reach backprop's spatial performance because the forward weights align to the feedback; a feedback
pathway that is RE-RANDOMISED every step (the falsifier) should not. All signatures are measured, not trained.
"""
import torch

from src.eval.credit_assignment import run_seed, _init, _grads


def test_feedback_alignment_signatures():
    o = run_seed(0, iters=600)

    # (A) PARITY: the biological rule (feedback alignment) learns the spatial task as well as backprop,
    # far below the position-blind floor — WITHOUT weight transport.
    assert o["feedback_decode"] < 0.6 * o["floor"], "feedback alignment should learn the task (beat the floor)"
    assert o["feedback_decode"] < o["backprop_decode"] + 0.04, "feedback should reach backprop's decode error"

    # (B) ALIGNMENT EMERGES and is the discriminator vs the shuffled null: the feedback-delivered error
    # aligns with the true gradient, while a re-randomised feedback delivers a ~orthogonal (~0) signal.
    assert o["feedback_walign"] > 0.03, "forward weights should rotate to align with the fixed feedback"
    assert o["feedback_galign"] > o["shuffled_galign"] + 0.03, "consistent feedback aligns; shuffled does not"

    # (C) FALSIFIER: shuffling the feedback every step cripples learning (it is the CONSISTENT feedback
    # pathway, not any random matrix, that assigns credit).
    assert o["falsifier_gap"] > 0.02, "shuffled feedback should learn worse than consistent feedback"
    assert o["shuffled_decode"] > o["feedback_decode"] + 0.02


def test_feedback_uses_no_weight_transport():
    """Mechanism check: the feedback rule's hidden update must NOT equal backprop's (it uses fixed random B,
    not Wᵀ), yet must be a real, finite update."""
    gen = torch.Generator().manual_seed(0)
    net = _init(gen)
    x = torch.rand(16, 2, generator=gen)
    target = torch.rand(16, 64, generator=gen)
    dW1_bp, _, _ = _grads(x, target, net, "backprop", gen)
    dW1_fa, _, _ = _grads(x, target, net, "feedback", gen)
    assert dW1_bp.shape == dW1_fa.shape
    assert not torch.isnan(dW1_fa).any()
    # at init, the fixed random feedback is NOT the transpose of the forward weights -> updates differ
    assert not torch.allclose(dW1_bp, dW1_fa, atol=1e-6), "feedback update must not equal the weight-transport update"
