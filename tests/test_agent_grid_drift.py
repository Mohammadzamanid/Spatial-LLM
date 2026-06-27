"""
tests/test_agent_grid_drift.py
Locks the path-integration drift + boundary-vector-cell correction result: the BVC learned read-out
localizes near walls; under noisy self-motion the un-anchored grid estimate drifts far while BVC
anchoring keeps it bounded (much smaller error).
"""
import statistics as st

import torch

from src.eval.agent_grid_cortex import build_cortex, train_decoder
from src.eval.agent_grid_drift import train_bvc, bvc_coord_err, walk


def test_bvc_readout_localizes_near_walls():
    gen = torch.Generator().manual_seed(3)
    bvc, loc = train_bvc(gen, iters=800)
    assert bvc_coord_err(bvc, loc, gen) < 0.1          # BVC organ -> wall coordinate is decodable near walls


def test_anchoring_bounds_drift():
    mod = build_cortex(0)
    gen = torch.Generator().manual_seed(11)
    dec = train_decoder(mod, gen, nonlinear=True, iters=800)
    bvc, loc = train_bvc(gen, iters=800)

    def final_err(noise, do_anchor, n=12):
        out = []
        for _ in range(n):
            e = walk(mod, dec, bvc, loc, gen, noise, do_anchor)
            out.append(st.mean(e[-20:]))
        return st.mean(out)

    # no noise: both tiny
    assert final_err(0.0, False) < 0.1
    # under noise: un-anchored drifts large; anchored stays much smaller (>2x improvement)
    na = final_err(0.12, False); an = final_err(0.12, True)
    assert na > 0.4, f"expected un-anchored drift to be large, got {na:.3f}"
    assert an < na / 2.0, f"expected BVC anchoring to at least halve drift: na={na:.3f} an={an:.3f}"
