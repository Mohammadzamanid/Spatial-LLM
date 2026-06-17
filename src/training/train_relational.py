"""
src/training/train_relational.py

STRUCTURAL TRANSFER through a frozen LLM — the TEM headline at the language level.

A space-only-trained cortex is FROZEN. An abstract ordered structure (ranks 0..N-1) is laid along a
1-D concept axis; each item enters the model by its OWN position through the frozen cortex (never the
signed relative displacement — that would leak the answer). A LoRA-Qwen reads BOTH items' spatial
tokens (two cortex encodings, concatenated) and answers a LINGUISTIC comparison ("Is the first item
ranked higher than the second?"). We train the LoRA readout on ADJACENT pairs only and test transitive
inference on never-seen far pairs — plus the falsifiers (shuffled positions, scrambled 2nd item) and
the cortex-OFF text-only control.

Design validated on CPU first (src/eval/structural_transfer_cortex.py): through the actual frozen
cortex.encode, TI = 0.99 and the shuffled control collapses. This reuses the proven TrajectoryLLM
components (cortex / to_tokens / gated fusion / LoRA-LLM); the only addition is the two-item forward.

    python -m src.training.train_relational --n_items 12 --epochs 4 --seed 0 --out results/relational_llm_s0.json
"""
import argparse
import json
import math
import os
from collections import Counter

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoTokenizer

from ..models.trajectory_llm import TrajectoryLLM

PROMPT = ("[RANKING] Two items were shown, in order, as places along a line.\n"
          "[QUESTION] Is the first item ranked higher than the second?\n[ANSWER]")
ENC_T = 8                                            # steps used to encode one item's position


def item_paths(xpos, device):
    """Each 1-D concept position x -> a directed ENC_T-step path reaching (x,0,0) (heading 0 or pi)."""
    x = xpos.to(device)
    heading = torch.where(x >= 0, torch.zeros_like(x), torch.full_like(x, math.pi)).unsqueeze(1).expand(-1, ENC_T)
    speed = (x.abs() / ENC_T).unsqueeze(1).expand(-1, ENC_T)
    vz = torch.zeros(x.shape[0], ENC_T, device=device)
    return heading.contiguous(), speed.contiguous(), vz


def two_path_out(model, ids, attn, pa, pb, labels=None, ablate=False):
    """Reuse TrajectoryLLM submodules: encode two item-paths -> concat spatial tokens -> fuse -> LLM."""
    text = model._embed()(ids)
    B = text.shape[0]
    if ablate:
        spatial = torch.zeros(B, 2 * model.n_tokens, text.shape[-1], device=text.device, dtype=text.dtype)
    else:
        ta = model.to_tokens(model.cortex.encode(*pa)).view(B, model.n_tokens, -1)
        tb = model.to_tokens(model.cortex.encode(*pb)).view(B, model.n_tokens, -1)
        spatial = torch.cat([ta, tb], dim=1).to(text.dtype)
    fused = model.fusion(text, spatial)
    return model.llm(inputs_embeds=fused, attention_mask=attn, labels=labels)


def pretrain_freeze_cortex(model, device, epochs=45, env_half=4.0, K=512, sigma=1.2, T=8, seed=0):
    """Self-supervised Euclidean place-code pretraining over directed walks, then FREEZE (faithful)."""
    cx = model.cortex; g = torch.Generator().manual_seed(seed)
    centers = (torch.rand(K, 3, generator=g) * (2 * env_half) - env_half).to(device)
    sup = nn.Linear(model.cortex.embed_dim if hasattr(model.cortex, "embed_dim") else 128, K).to(device)
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


def collate(pairs, ranks, tok, device, max_len=48):
    full = [PROMPT + (" Yes." if ranks[i] > ranks[j] else " No.") for i, j in pairs]
    enc = tok(full, max_length=max_len, padding="max_length", truncation=True, return_tensors="pt")
    labels = enc["input_ids"].clone()
    plen = len(tok(PROMPT)["input_ids"]); labels[:, :plen] = -100
    labels[enc["attention_mask"] == 0] = -100
    return {k: v.to(device) for k, v in {"input_ids": enc["input_ids"], "attention_mask": enc["attention_mask"],
                                         "labels": labels}.items()}


@torch.no_grad()
def evaluate(model, tok, pairs, ranks, xpos, device, ablate=False, scramble=False, bs=64):
    """Score by comparing the Yes-vs-No next-token LOGITS at [ANSWER] (deterministic; no generation/
    parsing, so a model that never says yes/no can't masquerade as 50%)."""
    yes_id = tok(" Yes", add_special_tokens=False).input_ids[0]
    no_id = tok(" No", add_special_tokens=False).input_ids[0]
    cor = tot = 0
    prompt_ids = tok(PROMPT, return_tensors="pt").input_ids.to(device)
    for k in range(0, len(pairs), bs):
        pb_ = pairs[k:k + bs]
        ii = torch.tensor([p[0] for p in pb_]); jj = torch.tensor([p[1] for p in pb_])
        if scramble:
            jj = jj[torch.randperm(len(jj))]
        pa = item_paths(xpos[ii], device); pbp = item_paths(xpos[jj], device)
        ids = prompt_ids.expand(len(pb_), -1); attn = torch.ones_like(ids)
        text = model._embed()(ids); B = len(pb_)
        if ablate:
            sp = torch.zeros(B, 2 * model.n_tokens, text.shape[-1], device=device, dtype=text.dtype)
        else:
            ta = model.to_tokens(model.cortex.encode(*pa)).view(B, model.n_tokens, -1)
            tb = model.to_tokens(model.cortex.encode(*pbp)).view(B, model.n_tokens, -1)
            sp = torch.cat([ta, tb], 1).to(text.dtype)
        fused = model.fusion(text, sp)
        last = model.llm(inputs_embeds=fused, attention_mask=attn).logits[:, -1, :]   # next-token logits
        pred = (last[:, yes_id] > last[:, no_id]).long().cpu()
        for r, (i, j) in enumerate(pb_):
            cor += int(int(pred[r]) == int(ranks[i] > ranks[j])); tot += 1
    return cor / max(tot, 1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base_llm", default="Qwen/Qwen2.5-1.5B")
    ap.add_argument("--n_items", type=int, default=12)
    ap.add_argument("--spacing", type=float, default=0.5)
    ap.add_argument("--steps", type=int, default=1500)        # gradient steps (jittered -> unlimited data)
    ap.add_argument("--jitter", type=float, default=0.15)     # position noise (< spacing/2 keeps rank order)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--bs", type=int, default=16)
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

    N, D = a.n_items, a.spacing
    ranks = torch.arange(N).float()
    xpos = ranks * D - (N - 1) * D / 2
    adj = [(i, i + 1) for i in range(N - 1)] + [(i + 1, i) for i in range(N - 1)]
    far = [(i, j) for i in range(N) for j in range(N) if abs(i - j) >= 2]

    train_params = [p for n, p in model.named_parameters() if p.requires_grad]   # LoRA + to_tokens + fusion
    opt = torch.optim.AdamW(train_params, lr=a.lr)
    model.train()
    g = torch.Generator().manual_seed(a.seed)
    adj_t = torch.tensor(adj)
    for step in range(a.steps):                          # step-based; positions JITTERED -> unlimited data
        sel = adj_t[torch.randint(len(adj_t), (a.bs,), generator=g)]
        ii, jj = sel[:, 0], sel[:, 1]
        xa = xpos[ii] + a.jitter * torch.randn(a.bs, generator=g)   # jitter keeps rank order (< spacing/2)
        xb = xpos[jj] + a.jitter * torch.randn(a.bs, generator=g)
        b = collate(list(zip(ii.tolist(), jj.tolist())), ranks, tok, device)
        pa = item_paths(xa, device); pb = item_paths(xb, device)
        out = two_path_out(model, b["input_ids"], b["attention_mask"], pa, pb, labels=b["labels"])
        opt.zero_grad(); out.loss.backward(); opt.step()
        if step % 200 == 0 or step == a.steps - 1:
            print(f"step {step}: loss {out.loss.item():.3f}", flush=True)

    model.eval()
    res = {
        "transitive_inference_far": round(evaluate(model, tok, far, ranks, xpos, device), 4),
        "adjacent_trained": round(evaluate(model, tok, adj, ranks, xpos, device), 4),
        "far_cortex_OFF": round(evaluate(model, tok, far, ranks, xpos, device, ablate=True), 4),
        "far_scrambled_2nd": round(evaluate(model, tok, far, ranks, xpos, device, scramble=True), 4),
    }
    # shuffled-position falsifier: re-place ranks at random positions, eval the SAME model
    xpos_sh = xpos[torch.randperm(N)]
    res["far_shuffled_positions"] = round(evaluate(model, tok, far, ranks, xpos_sh, device), 4)
    print("\nSTRUCTURAL TRANSFER through the frozen LLM:", flush=True)
    for k, v in res.items():
        print(f"  {k:28} {v:.1%}", flush=True)
    print("  (chance 50%; cortex-OFF should be ~chance; shuffled/scrambled should collapse toward chance)", flush=True)
    if a.out:
        os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
        json.dump({"seed": a.seed, "n_items": N, "results": res}, open(a.out, "w"), indent=2)
        print(f"\nwrote {a.out}", flush=True)


if __name__ == "__main__":
    main()
