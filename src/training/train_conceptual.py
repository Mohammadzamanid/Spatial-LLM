"""
src/training/train_conceptual.py

#8 — the LLM reads a 2-D CONCEPTUAL grid (Constantinescu, Behrens 2016; Bellmund 2018), the TEM cognitive-map
claim extended from SPACE to MEANING at the language level.

** GPU (T4) REQUIRED — this trains a frozen Qwen-1.5B + LoRA. It is NOT run on CPU. The DESIGN was validated
   on CPU first in src/eval/conceptual_grid_cortex.py (n=5): on the ACTUAL frozen cortex.encode pipeline this
   trainer reads, the code carries a control-clean 2-D metric — OFF-AXIS "closer" 0.65 (>chance, where a 1-D
   code is <=0.5 by construction), held-out decode 0.63 vs shuffled 3.4 spacing, shuffled Spearman ~0. This
   trainer swaps the CPU read-out for the frozen LLM, expected to SHARPEN the signal (1-D precedent
   structural_transfer_cortex -> train_relational: 1.0 -> 0.99). The triple-item forward is NEW vs the proven
   two-item train_relational.py — if a run errors it likely needs one quick debug pass (it reuses the same
   TrajectoryLLM cortex / to_tokens / gated-fusion / LoRA-LLM path). **

A space-only-trained cortex is FROZEN. Concepts are laid at 2-D coordinates (a G x G grid). Each concept enters
by its OWN position through the frozen cortex (heading=atan2(y,x), speed=r/T; never a relative displacement ->
no leak). A LoRA-Qwen reads THREE concept codes — an anchor A and two candidates B, C — and answers the
LINGUISTIC comparison "Is the first concept closer to the anchor than the second?" We train on NEAR triples
(both candidates near the anchor) and test on FAR / OFF-AXIS triples (where the 1-D x-projection ordering
disagrees with the true 2-D answer — un-fakeable by a 1-D code). Falsifiers: cortex-OFF text-only (same LoRA
budget, no spatial code -> chance), shuffled positions (concept<->position permuted -> collapse).

    python -m src.training.train_conceptual --G 6 --spacing 0.8 --steps 1800 --seed 0 \
        --out results/conceptual_llm/conceptual_s0.json
"""
import argparse
import json
import math
import os

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoTokenizer

from ..models.trajectory_llm import TrajectoryLLM

PROMPT = ("[CONCEPTS] Three items were shown as places on a map: an ANCHOR, then a FIRST, then a SECOND.\n"
          "[QUESTION] Is the FIRST item closer to the ANCHOR than the SECOND?\n[ANSWER]")
ENC_T = 8


def walk_2d(pos, device, T=ENC_T):
    """Each 2-D concept position (x,y) -> a directed T-step path reaching net (x,y,0). heading=atan2(y,x)."""
    x, y = pos[:, 0].to(device), pos[:, 1].to(device)
    r = torch.sqrt(x * x + y * y).clamp_min(1e-6)
    heading = torch.atan2(y, x).unsqueeze(1).expand(-1, T)
    speed = (r / T).unsqueeze(1).expand(-1, T)
    return heading.contiguous(), speed.contiguous(), torch.zeros(pos.shape[0], T, device=device)


def triple_spatial(model, head, pa, pb, pc, ablate=False):
    """JOINT triple read-out: concat the three FROZEN concept codes -> trainable head -> spatial tokens
    (matches train_relational's concat-the-codes pair readout that made the relation linearly accessible).
    Cortex stays frozen; each concept enters by its own position -> no leak."""
    B = pa[0].shape[0]
    llm_dim = model.to_tokens.out_features // model.n_tokens
    if ablate:
        return torch.zeros(B, model.n_tokens, llm_dim, device=pa[0].device)
    code = torch.cat([model.cortex.encode(*pa), model.cortex.encode(*pb), model.cortex.encode(*pc)], -1)
    return head(code).view(B, model.n_tokens, llm_dim)


def triple_out(model, head, ids, attn, pa, pb, pc, labels=None, ablate=False):
    text = model._embed()(ids)
    spatial = triple_spatial(model, head, pa, pb, pc, ablate).to(text.dtype)
    fused = model.fusion(text, spatial)
    return model.llm(inputs_embeds=fused, attention_mask=attn, labels=labels)


def pretrain_freeze_cortex(model, device, epochs=45, env_half=4.0, K=512, sigma=1.2, T=8, seed=0):
    """Self-supervised Euclidean place-code pretraining over directed walks, then FREEZE (faithful; identical
    to train_relational / structural_transfer_cortex — the cortex sees ONLY physical space)."""
    cx = model.cortex; g = torch.Generator().manual_seed(seed)
    centers = (torch.rand(K, 3, generator=g) * (2 * env_half) - env_half).to(device)
    sup = nn.Linear(cx.embed_dim if hasattr(cx, "embed_dim") else 128, K).to(device)
    opt = torch.optim.Adam(list(cx.parameters()) + list(sup.parameters()), lr=3e-3)

    def place(pos):
        d2 = ((pos.unsqueeze(1) - centers.unsqueeze(0)) ** 2).sum(-1)
        return torch.exp(-d2 / (2 * sigma ** 2))
    cx.train()
    for _ in range(epochs):
        h = (torch.rand(256, T, generator=g) * 2 * math.pi).to(device)
        s = (torch.rand(256, T, generator=g) * 0.8).to(device)
        vz = ((torch.rand(256, T, generator=g) - 0.5) * 0.4).to(device)
        pos = torch.stack([(s * h.cos()).sum(1), (s * h.sin()).sum(1), vz.sum(1)], -1)
        opt.zero_grad(); F.mse_loss(sup(cx.encode(h, s, vz)), place(pos)).backward(); opt.step()
    for p in cx.parameters():
        p.requires_grad_(False)
    cx.eval()


def build_grid(G, spacing):
    xs = torch.arange(G).float() * spacing - (G - 1) * spacing / 2
    return torch.stack(torch.meshgrid(xs, xs, indexing="ij"), -1).reshape(-1, 2)


def make_triples(grid, near_r, margin):
    """All (anchor, cand1, cand2). near = both candidates within near_r of anchor (TRAIN). far = otherwise.
    off-axis = 1-D x-projection ordering disagrees with true 2-D answer. `margin` (absolute, ~0.2*spacing)
    drops triples whose two candidate-distances are too close to call (ties)."""
    N = grid.shape[0]; posd = torch.cdist(grid, grid)
    tr, far, faroff = [], [], []
    labels = {}
    for a in range(N):
        for b in range(N):
            for c in range(N):
                if len({a, b, c}) < 3:
                    continue
                db, dc = posd[a, b], posd[a, c]
                if abs(db - dc) <= margin:
                    continue
                lab = int(db < dc)                                     # 1 => first(b) closer -> "Yes"
                labels[(a, b, c)] = lab
                x1 = (grid[a, 0] - grid[b, 0]).abs() < (grid[a, 0] - grid[c, 0]).abs()
                is_off = (int(x1) != lab)
                if max(db, dc) <= near_r:
                    tr.append((a, b, c))
                else:
                    far.append((a, b, c))
                    if is_off:
                        faroff.append((a, b, c))
    return tr, far, faroff, labels


def collate(triples, labels, tok, device, max_len=56):
    full = [PROMPT + (" Yes." if labels[t] else " No.") for t in triples]
    enc = tok(full, max_length=max_len, padding="max_length", truncation=True, return_tensors="pt")
    lab = enc["input_ids"].clone()
    plen = len(tok(PROMPT)["input_ids"]); lab[:, :plen] = -100
    lab[enc["attention_mask"] == 0] = -100
    return {k: v.to(device) for k, v in {"input_ids": enc["input_ids"],
                                         "attention_mask": enc["attention_mask"], "labels": lab}.items()}


@torch.no_grad()
def evaluate(model, head, tok, triples, labels, grid, device, ablate=False, shuffle_perm=None, bs=8):
    """Candidate-NLL scoring: for each triple, model's NLL of ' Yes.' vs ' No.' given the concept codes.
    Small default bs: the LM-head logits are (bs, seq, ~152k vocab) — the dominant memory term on a T4."""
    plen = len(tok(PROMPT)["input_ids"]); cands = [" Yes.", " No."]; cor = tot = 0
    pos = grid if shuffle_perm is None else grid[shuffle_perm]     # shuffled: concept<->position permuted
    for k in range(0, len(triples), bs):
        tb = triples[k:k + bs]; B = len(tb)
        ia = torch.tensor([t[0] for t in tb]); ib = torch.tensor([t[1] for t in tb]); ic = torch.tensor([t[2] for t in tb])
        pa = walk_2d(pos[ia], device); pbp = walk_2d(pos[ib], device); pcp = walk_2d(pos[ic], device)
        sp = triple_spatial(model, head, pa, pbp, pcp, ablate)
        nlls = []
        for cand in cands:
            enc = tok([PROMPT + cand] * B, max_length=56, padding="max_length", truncation=True, return_tensors="pt")
            ids = enc["input_ids"].to(device); attn = enc["attention_mask"].to(device)
            lab = ids.clone(); lab[:, :plen] = -100; lab[attn == 0] = -100
            text = model._embed()(ids); fused = model.fusion(text, sp.to(text.dtype))
            logits = model.llm(inputs_embeds=fused, attention_mask=attn).logits
            lp = logits[:, :-1, :]; ll = lab[:, 1:]
            nll = F.cross_entropy(lp.reshape(-1, lp.size(-1)), ll.reshape(-1),
                                  reduction="none", ignore_index=-100).reshape(B, -1).sum(1)
            nlls.append(nll)
        pred = (nlls[0] < nlls[1]).long().cpu()
        for r, t in enumerate(tb):
            cor += int(int(pred[r]) == labels[t]); tot += 1
    return cor / max(tot, 1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base_llm", default="Qwen/Qwen2.5-1.5B")
    ap.add_argument("--G", type=int, default=6)
    ap.add_argument("--spacing", type=float, default=0.8)
    ap.add_argument("--steps", type=int, default=1800)
    ap.add_argument("--jitter", type=float, default=0.12)          # < spacing/2 keeps the geometry
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--bs", type=int, default=8)                   # T4-safe (LM-head logits over ~152k vocab)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    torch.manual_seed(a.seed)

    tok = AutoTokenizer.from_pretrained(a.base_llm, use_fast=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    model = TrajectoryLLM(base_llm=a.base_llm, cortex_constrained_velocity=True).to(device)
    pretrain_freeze_cortex(model, device, seed=a.seed)
    llm_dim = model.to_tokens.out_features // model.n_tokens
    head = nn.Linear(3 * model.cortex.embed_dim, llm_dim * model.n_tokens).to(device)   # joint triple readout
    # T4 memory: gradient-checkpoint the LLM (frees transformer-layer activations to make room for the
    # (bs, seq, ~152k vocab) LM-head logits). inputs_embeds carry grad from the trainable fusion/head, so
    # checkpointing is transparent to the trainable params. Guarded — degrades gracefully if unavailable.
    try:
        model.llm.config.use_cache = False
        model.llm.gradient_checkpointing_enable()
        print("gradient checkpointing: ON", flush=True)
    except Exception as e:
        print(f"gradient checkpointing unavailable ({e}); relying on small --bs", flush=True)

    grid = build_grid(a.G, a.spacing); N = grid.shape[0]
    near_r = 2.1 * a.spacing; margin = 0.2 * a.spacing        # near = local (<= ~2-step) with a resolvable margin
    tr, far, faroff, labels = make_triples(grid, near_r, margin)
    print(f"triples: train(near)={len(tr)} far={len(far)} far-offaxis={len(faroff)}", flush=True)
    if len(tr) == 0 or len(faroff) == 0:
        raise SystemExit("empty train or off-axis set — adjust --G/--spacing/near_r/margin")

    train_params = [p for p in model.parameters() if p.requires_grad] + list(head.parameters())
    opt = torch.optim.AdamW(train_params, lr=a.lr); model.train()
    g = torch.Generator().manual_seed(a.seed); tr_t = torch.tensor(tr)
    for step in range(a.steps):
        sel = tr_t[torch.randint(len(tr_t), (a.bs,), generator=g)]
        # jitter each of the three positions (keeps the closer-ordering: jitter < spacing/2)
        def jit(ix):
            return grid[ix] + a.jitter * torch.randn(a.bs, 2, generator=g)
        pa = walk_2d(jit(sel[:, 0]), device); pb = walk_2d(jit(sel[:, 1]), device); pc = walk_2d(jit(sel[:, 2]), device)
        b = collate([tuple(t.tolist()) for t in sel], labels, tok, device)
        out = triple_out(model, head, b["input_ids"], b["attention_mask"], pa, pb, pc, labels=b["labels"])
        opt.zero_grad(); out.loss.backward(); opt.step()
        if step % 200 == 0 or step == a.steps - 1:
            print(f"step {step}: loss {out.loss.item():.3f}", flush=True)

    model.eval()
    del opt
    if torch.cuda.is_available():
        torch.cuda.empty_cache()                                   # free training grads/optimizer before eval
    perm = torch.randperm(N, generator=torch.Generator().manual_seed(a.seed + 7))
    res = {
        "closer_far": round(evaluate(model, head, tok, far, labels, grid, device), 4),
        "closer_far_OFFAXIS": round(evaluate(model, head, tok, faroff, labels, grid, device), 4),
        "closer_near_trained": round(evaluate(model, head, tok, tr, labels, grid, device), 4),
        "closer_far_cortex_OFF": round(evaluate(model, head, tok, far, labels, grid, device, ablate=True), 4),
        "closer_far_shuffled_pos": round(evaluate(model, head, tok, far, labels, grid, device, shuffle_perm=perm), 4),
    }
    print("\nCONCEPTUAL GRID through the frozen LLM (chance 50%):", flush=True)
    for k, v in res.items():
        print(f"  {k:26} {v:.1%}", flush=True)
    print("  (cortex-OFF & shuffled -> ~chance; OFF-AXIS > chance = genuine 2-D reasoning through the map)", flush=True)
    if res["closer_near_trained"] < 0.6:
        print("  WARNING: near (TRAINED) ~chance despite low loss -> eval/readout mismatch, not a real null.", flush=True)
    if a.out:
        os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
        json.dump({"seed": a.seed, "G": a.G, "spacing": a.spacing, "results": res}, open(a.out, "w"), indent=2)
        ckpt = {f"model.{n}": p.detach().cpu() for n, p in model.named_parameters() if p.requires_grad}
        ckpt.update({f"head.{n}": p.detach().cpu() for n, p in head.named_parameters()})
        torch.save(ckpt, a.out.replace(".json", ".pt"))
        print(f"\nwrote {a.out} (+ .pt checkpoint)", flush=True)


if __name__ == "__main__":
    main()
