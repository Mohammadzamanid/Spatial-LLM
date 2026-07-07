"""
src/training/train_social.py

#9 — the LLM reasons over a 2-D SOCIAL space (Tavares 2015; Park, Miller 2021: humans map social hierarchy of
POWER x AFFILIATION with the same grid/hippocampal machinery; builds on gap #4's self/other place cells). The
cognitive-map claim extended from space to the SOCIAL domain at the language level.

** GPU (T4) REQUIRED — trains a frozen Qwen-1.5B + LoRA; NOT run on CPU. The DESIGN was validated on CPU first
   in src/eval/social_grid_cortex.py (n=5): on the ACTUAL frozen cortex.encode this trainer reads, held-out
   DOMINANCE from the power axis = 0.96, the axis DISSOCIATION (power->dominance 0.96 vs affiliation->dominance
   0.45, gap +0.51) is clean, social-distance OFF-AXIS = 0.65 (>chance), and shuffled collapses dominance to
   ~0.44. The DOMINANCE task reuses the PROVEN two-item forward of train_relational (lowest debug risk); the
   social-distance task reuses the triple forward of train_conceptual. **

Agents live in a 2-D social space (axis-0 = POWER, axis-1 = AFFILIATION). Each agent enters by its OWN social
position through the FROZEN space-pretrained cortex (heading=atan2, speed=r/T; never a relative displacement ->
no leak). Two queries:
  DOMINANCE (default; pair):  "Is the FIRST person more dominant than the SECOND?" (answer = the POWER axis).
    Trained on power-ADJACENT pairs; tested on FAR power pairs (social transitive inference) and the
    AFFILIATION-DISSOCIATION set (pairs whose affiliation ordering OPPOSES power — the model must read power,
    not affiliation). Falsifiers: cortex-OFF text-only, shuffled positions.
  SOCIAL-DISTANCE (--task distance; triple): "Is the FIRST socially closer to the ANCHOR than the SECOND?"
    (genuine 2-D; OFF-AXIS = power-axis ordering disagrees with 2-D social distance). Reuses train_conceptual.

    python -m src.training.train_social --task dominance --G 6 --spacing 0.8 --steps 1500 --seed 0 \
        --out results/social_llm/social_s0.json
"""
import argparse
import json
import os

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoTokenizer

from ..models.trajectory_llm import TrajectoryLLM
from .train_conceptual import walk_2d, build_grid, pretrain_freeze_cortex

DOM_PROMPT = ("[SOCIAL] Two people were shown as places on a social map.\n"
              "[QUESTION] Is the first person more dominant than the second?\n[ANSWER]")


def pair_spatial(model, head, pa, pb, ablate=False):
    """Concat the two FROZEN agent codes -> trainable head -> spatial tokens (the proven train_relational
    readout). Cortex frozen; each agent enters by its own social position -> no leak."""
    B = pa[0].shape[0]; llm_dim = model.to_tokens.out_features // model.n_tokens
    if ablate:
        return torch.zeros(B, model.n_tokens, llm_dim, device=pa[0].device)
    code = torch.cat([model.cortex.encode(*pa), model.cortex.encode(*pb)], -1)
    return head(code).view(B, model.n_tokens, llm_dim)


def collate(pairs, ymore, tok, device, max_len=48):
    full = [DOM_PROMPT + (" Yes." if ymore[k] else " No.") for k in range(len(pairs))]
    enc = tok(full, max_length=max_len, padding="max_length", truncation=True, return_tensors="pt")
    lab = enc["input_ids"].clone(); plen = len(tok(DOM_PROMPT)["input_ids"])
    lab[:, :plen] = -100; lab[enc["attention_mask"] == 0] = -100
    return {k: v.to(device) for k, v in {"input_ids": enc["input_ids"],
                                         "attention_mask": enc["attention_mask"], "labels": lab}.items()}


@torch.no_grad()
def evaluate(model, head, tok, pairs, grid, device, ablate=False, shuffle_perm=None, bs=32):
    """Candidate-NLL: 'Yes' (first more dominant) vs 'No'. Correct = sign of the POWER-axis difference."""
    plen = len(tok(DOM_PROMPT)["input_ids"]); cands = [" Yes.", " No."]; cor = tot = 0
    pos = grid if shuffle_perm is None else grid[shuffle_perm]
    for k in range(0, len(pairs), bs):
        pb_ = pairs[k:k + bs]; B = len(pb_)
        ia = torch.tensor([p[0] for p in pb_]); ib = torch.tensor([p[1] for p in pb_])
        pa = walk_2d(pos[ia], device); pbp = walk_2d(pos[ib], device)
        sp = pair_spatial(model, head, pa, pbp, ablate)
        nlls = []
        for cand in cands:
            enc = tok([DOM_PROMPT + cand] * B, max_length=48, padding="max_length", truncation=True, return_tensors="pt")
            ids = enc["input_ids"].to(device); attn = enc["attention_mask"].to(device)
            lab = ids.clone(); lab[:, :plen] = -100; lab[attn == 0] = -100
            text = model._embed()(ids); fused = model.fusion(text, sp.to(text.dtype))
            logits = model.llm(inputs_embeds=fused, attention_mask=attn).logits
            lp = logits[:, :-1, :]; ll = lab[:, 1:]
            nll = F.cross_entropy(lp.reshape(-1, lp.size(-1)), ll.reshape(-1),
                                  reduction="none", ignore_index=-100).reshape(B, -1).sum(1)
            nlls.append(nll)
        pred = (nlls[0] < nlls[1]).long().cpu()                                   # 1 => "Yes" (first dominant)
        for r, (i, j) in enumerate(pb_):
            cor += int(int(pred[r]) == int(grid[i, 0] > grid[j, 0])); tot += 1    # truth = POWER (axis 0)
    return cor / max(tot, 1)


def dominance_pairs(grid, G):
    """Power = axis-0 rank (0..G-1) shared by a whole affiliation column. Build:
      - adjacent power pairs (|dpower|=1) with any affiliation  -> TRAIN
      - far power pairs (|dpower|>=2)                            -> transitive-inference TEST
      - dissociation pairs: |dpower|>=2 AND affiliation ordering OPPOSES power -> must read power not affiliation
    """
    N = grid.shape[0]; xs = torch.unique(grid[:, 0])
    prank = {float(v): i for i, v in enumerate(xs.tolist())}
    arank_vals = torch.unique(grid[:, 1]); ar = {float(v): i for i, v in enumerate(arank_vals.tolist())}
    adj, far, diss = [], [], []
    for i in range(N):
        for j in range(N):
            if i == j:
                continue
            dp = prank[float(grid[i, 0])] - prank[float(grid[j, 0])]
            da = ar[float(grid[i, 1])] - ar[float(grid[j, 1])]
            if abs(dp) == 1:
                adj.append((i, j))
            elif abs(dp) >= 2:
                far.append((i, j))
                if dp * da < 0:                                    # affiliation ordering opposes power
                    diss.append((i, j))
    return adj, far, diss


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base_llm", default="Qwen/Qwen2.5-1.5B")
    ap.add_argument("--task", choices=["dominance", "distance"], default="dominance")
    ap.add_argument("--G", type=int, default=6)
    ap.add_argument("--spacing", type=float, default=0.8)
    ap.add_argument("--steps", type=int, default=1500)
    ap.add_argument("--jitter", type=float, default=0.12)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--bs", type=int, default=16)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    torch.manual_seed(a.seed)

    if a.task == "distance":                                       # social-distance reuses the #8 triple trainer
        from . import train_conceptual
        import sys
        sys.argv = ["train_social", "--G", str(a.G), "--spacing", str(a.spacing), "--steps", str(a.steps),
                    "--seed", str(a.seed)] + (["--out", a.out] if a.out else [])
        return train_conceptual.main()

    tok = AutoTokenizer.from_pretrained(a.base_llm, use_fast=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    model = TrajectoryLLM(base_llm=a.base_llm, cortex_constrained_velocity=True).to(device)
    pretrain_freeze_cortex(model, device, seed=a.seed)
    llm_dim = model.to_tokens.out_features // model.n_tokens
    head = nn.Linear(2 * model.cortex.embed_dim, llm_dim * model.n_tokens).to(device)

    grid = build_grid(a.G, a.spacing); N = grid.shape[0]
    adj, far, diss = dominance_pairs(grid, a.G)
    print(f"dominance pairs: train(adj power)={len(adj)} far={len(far)} dissociation={len(diss)}", flush=True)
    if len(adj) == 0 or len(diss) == 0:
        raise SystemExit("empty train or dissociation set — adjust --G/--spacing")

    train_params = [p for p in model.parameters() if p.requires_grad] + list(head.parameters())
    opt = torch.optim.AdamW(train_params, lr=a.lr); model.train()
    g = torch.Generator().manual_seed(a.seed); adj_t = torch.tensor(adj)
    for step in range(a.steps):
        sel = adj_t[torch.randint(len(adj_t), (a.bs,), generator=g)]
        ii, jj = sel[:, 0], sel[:, 1]
        pa = walk_2d(grid[ii] + a.jitter * torch.randn(a.bs, 2, generator=g), device)
        pb = walk_2d(grid[jj] + a.jitter * torch.randn(a.bs, 2, generator=g), device)
        ymore = [bool(grid[i, 0] > grid[j, 0]) for i, j in zip(ii.tolist(), jj.tolist())]
        b = collate(list(zip(ii.tolist(), jj.tolist())), ymore, tok, device)
        text = model._embed()(b["input_ids"])
        fused = model.fusion(text, pair_spatial(model, head, pa, pb).to(text.dtype))
        out = model.llm(inputs_embeds=fused, attention_mask=b["attention_mask"], labels=b["labels"])
        opt.zero_grad(); out.loss.backward(); opt.step()
        if step % 200 == 0 or step == a.steps - 1:
            print(f"step {step}: loss {out.loss.item():.3f}", flush=True)

    model.eval()
    perm = torch.randperm(N, generator=torch.Generator().manual_seed(a.seed + 7))
    res = {
        "dominance_far": round(evaluate(model, head, tok, far, grid, device), 4),
        "dominance_dissociation": round(evaluate(model, head, tok, diss, grid, device), 4),
        "dominance_adj_trained": round(evaluate(model, head, tok, adj, grid, device), 4),
        "dominance_far_cortex_OFF": round(evaluate(model, head, tok, far, grid, device, ablate=True), 4),
        "dominance_far_shuffled_pos": round(evaluate(model, head, tok, far, grid, device, shuffle_perm=perm), 4),
    }
    print("\nSOCIAL DOMINANCE through the frozen LLM (chance 50%):", flush=True)
    for k, v in res.items():
        print(f"  {k:28} {v:.1%}", flush=True)
    print("  (cortex-OFF & shuffled -> ~chance; dissociation > chance = reads POWER, not affiliation)", flush=True)
    if res["dominance_adj_trained"] < 0.6:
        print("  WARNING: adjacent (TRAINED) ~chance despite low loss -> eval/readout mismatch, not a real null.", flush=True)
    if a.out:
        os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
        json.dump({"seed": a.seed, "G": a.G, "spacing": a.spacing, "results": res}, open(a.out, "w"), indent=2)
        ckpt = {f"model.{n}": p.detach().cpu() for n, p in model.named_parameters() if p.requires_grad}
        ckpt.update({f"head.{n}": p.detach().cpu() for n, p in head.named_parameters()})
        torch.save(ckpt, a.out.replace(".json", ".pt"))
        print(f"\nwrote {a.out} (+ .pt checkpoint)", flush=True)


if __name__ == "__main__":
    main()
