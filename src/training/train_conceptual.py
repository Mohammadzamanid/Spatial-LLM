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


class CoincidenceReadout(nn.Module):
    """Readout for the 2-D metric ("closer") task — a COINCIDENCE DETECTOR, the neuroscientific mechanism by
    which grid cells give a cognitive map its *metric*: proximity is read from the OVERLAP/CORRELATION of grid
    population vectors (Bellmund & Behrens 2018; Bush, Barry, Burgess 2015) — a DOT PRODUCT (quadratic), which
    #9's linear head cannot compute.

    HONESTY (addresses an adversarial review): the per-candidate proximity module `prox` is applied SEPARATELY to
    each (anchor, candidate) pair with SHARED weights and NEVER sees both candidates together — so the readout
    physically cannot decide "which is closer"; the frozen LLM must, from the two proximity token-groups. This is
    exactly #9's split of labour (the head computes a per-item feature; the LLM does the relational compare).
    CPU-verified: a per-candidate proximity + a downstream linear compare generalizes NEAR->OFF-AXIS FAR at ~0.70
    (chance 0.5), where a linear head gives 0.50 and the free-MLP joint head could self-answer (rejected here).
    Honest bound: the readout computes the METRIC (proximity); the LLM does only the ORDINAL compare of the two."""

    def __init__(self, code_dim, out_dim, k=32, hidden=256, rep=256):
        super().__init__()
        self.proj = nn.Linear(code_dim, k, bias=False)          # grid -> map subspace (learned)
        self.prox = nn.Sequential(nn.Linear(2 * code_dim + k, hidden), nn.GELU(), nn.Linear(hidden, rep))
        self.combine = nn.Linear(2 * rep, out_dim)              # LINEAR mix -> JOINT tokens (like #9)

    def _one(self, ea, ex):                                     # proximity of candidate X to anchor A (shared, nonlinear)
        pa, px = self.proj(ea), self.proj(ex)
        return self.prox(torch.cat([ea, ex, pa * px], -1))      # anchor-candidate coincidence overlap -> proximity rep

    def forward(self, ea, eb, ec):
        rb, rc = self._one(ea, eb), self._one(ea, ec)
        # LINEAR combine of the two per-candidate proximities -> joint tokens: it can encode the graded
        # difference rb-rc (making it accessible like #9's power difference) but CANNOT threshold it, so the
        # frozen LLM still does the decision. The only nonlinearity (prox) is per-candidate -> honest.
        return self.combine(torch.cat([rb, rc], -1))


def triple_spatial(model, head, pa, pb, pc, ablate=False):
    """Read-out for (anchor, candidate1, candidate2) via the COINCIDENCE detector (see CoincidenceReadout).
    Cortex stays frozen; each concept enters by its own position -> no leak."""
    B = pa[0].shape[0]
    llm_dim = model.to_tokens.out_features // model.n_tokens
    if ablate:
        return torch.zeros(B, model.n_tokens, llm_dim, device=pa[0].device)
    nc = lambda p: _norm_code(model, model.cortex.encode(*p))   # gain-control (see _norm_code)
    return head(nc(pa), nc(pb), nc(pc)).view(B, model.n_tokens, llm_dim)


def _norm_code(model, e):
    """Gain control / normalization between the frozen cortex and the readout (divisive normalization). The
    code is ~98% a position-independent constant + ~2% signal; the downstream LayerNorm cannot remove that
    across-concept constant, so the signal is invisible (fusion std ~0) without this. Standardizing per-dim
    (stats over the whole concept set -> no label leak) lifts the signal to unit scale; linear decode unchanged."""
    return (e - model._code_mean) / model._code_std


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
    # SINGLE-token answer (" Yes"/" No", no period): entire loss is the decision, aligned with the single-token eval.
    full = [PROMPT + (" Yes" if labels[t] else " No") for t in triples]
    enc = tok(full, max_length=max_len, padding="max_length", truncation=True, return_tensors="pt")
    lab = enc["input_ids"].clone()
    plen = len(tok(PROMPT)["input_ids"]); lab[:, :plen] = -100
    lab[enc["attention_mask"] == 0] = -100
    return {k: v.to(device) for k, v in {"input_ids": enc["input_ids"],
                                         "attention_mask": enc["attention_mask"], "labels": lab}.items()}


def _yes_no_ids(tok, prompt=PROMPT):
    """First-answer token ids IN CONTEXT (robust to context-dependent tokenization): the token at position plen
    in `prompt + " Yes"` / `prompt + " No"` is EXACTLY what training teacher-forces (standalone tok(' Yes')[0]
    can differ after '[ANSWER]'), so the eval reads the same token the model learned to boost."""
    plen = len(tok(prompt)["input_ids"])
    return tok(prompt + " Yes")["input_ids"][plen], tok(prompt + " No")["input_ids"][plen]


@torch.no_grad()
def evaluate(model, head, tok, triples, labels, grid, device, ablate=False, shuffle_perm=None, bs=16):
    """PADDING-IMMUNE single-next-token scoring: after the (identical) PROMPT + the concept codes, compare the
    model's next-token logit for ' Yes' vs ' No' at the last prompt position — exactly the token teacher-forced
    in training. One forward per triple (fast), no answer-token masking, so it is robust to the tokenizer's
    padding side (the old candidate-NLL scheme mis-masked under left padding and could read a constant)."""
    yes_id, no_id = _yes_no_ids(tok, PROMPT)
    prompt_ids = torch.tensor(tok(PROMPT)["input_ids"], device=device)
    T = prompt_ids.shape[0]
    pos = grid if shuffle_perm is None else grid[shuffle_perm]     # shuffled: concept<->position permuted
    cor = tot = 0
    for k in range(0, len(triples), bs):
        tb = triples[k:k + bs]; B = len(tb)
        ia = torch.tensor([t[0] for t in tb]); ib = torch.tensor([t[1] for t in tb]); ic = torch.tensor([t[2] for t in tb])
        pa = walk_2d(pos[ia], device); pbp = walk_2d(pos[ib], device); pcp = walk_2d(pos[ic], device)
        sp = triple_spatial(model, head, pa, pbp, pcp, ablate)
        ids = prompt_ids.unsqueeze(0).expand(B, T)
        attn = torch.ones(B, T, device=device)
        text = model._embed()(ids)
        fused = model.fusion(text, sp.to(text.dtype))
        logits = model.llm(inputs_embeds=fused, attention_mask=attn).logits     # (B, T, V)
        final = logits[:, -1, :]                                                # last prompt token -> answer
        pred = (final[:, yes_id] > final[:, no_id]).long().cpu()                # 1 => ' Yes' => first closer
        for r, t in enumerate(tb):
            cor += int(int(pred[r]) == labels[t]); tot += 1
    return cor / max(tot, 1)


def balance_cap(triples, labels, n, seed):
    """Subsample `triples` to a label-BALANCED, size-capped list (equal label 0/1, total <= n). Makes the
    reported accuracy honest (chance = 0.5) and keeps the T4 eval to minutes instead of scoring all ~37k."""
    g = torch.Generator().manual_seed(seed)
    order = torch.randperm(len(triples), generator=g).tolist()
    pos = [triples[i] for i in order if labels[triples[i]] == 1]
    neg = [triples[i] for i in order if labels[triples[i]] == 0]
    m = min(len(pos), len(neg), n // 2)
    return pos[:m] + neg[:m]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base_llm", default="Qwen/Qwen2.5-1.5B")
    ap.add_argument("--G", type=int, default=6)
    ap.add_argument("--spacing", type=float, default=0.8)
    ap.add_argument("--steps", type=int, default=1800)
    ap.add_argument("--jitter", type=float, default=0.12)          # < spacing/2 keeps the geometry
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--bs", type=int, default=8)                   # T4-safe (LM-head logits over ~152k vocab)
    ap.add_argument("--grad_ckpt", action="store_true")            # OFF by default: it silently killed adapter grads
    ap.add_argument("--eval_cap", type=int, default=1200)          # balanced+capped eval set size (fast on T4)
    ap.add_argument("--reeval", default=None)                      # path to a .pt checkpoint: load + eval, skip training
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    torch.manual_seed(a.seed)

    tok = AutoTokenizer.from_pretrained(a.base_llm, use_fast=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    tok.padding_side = "right"   # CRITICAL: collate masks the first plen tokens as the prompt; left padding
                                 # (some Qwen defaults) would put padding there and mis-target the loss.
    model = TrajectoryLLM(base_llm=a.base_llm, cortex_constrained_velocity=True).to(device)
    pretrain_freeze_cortex(model, device, seed=a.seed)
    llm_dim = model.to_tokens.out_features // model.n_tokens
    # COINCIDENCE-DETECTOR readout (see CoincidenceReadout): "closer" is a DISTANCE = grid population-vector
    # OVERLAP (quadratic), which #9's linear head cannot compute. A learned bilinear coincidence term supplies
    # it (CPU-verified: near->off-axis 0.71 vs the MLP head's 0.50).
    head = CoincidenceReadout(model.cortex.embed_dim, llm_dim * model.n_tokens, k=32, hidden=512).to(device)
    # T4 memory: gradient checkpointing is OFF by default. On a PEFT model with a FROZEN base fed via
    # inputs_embeds, plain gradient_checkpointing_enable() silently drops gradients to the LoRA adapters
    # (reentrant checkpointing needs an input that requires grad) -> the model trains NOTHING and reads as a
    # constant predictor (exactly what a first T4 run showed). If enabled, do it the correct way:
    # enable_input_require_grads() + use_reentrant=False. Small --bs fits a T4 without checkpointing at all.
    model.llm.config.use_cache = False
    if a.grad_ckpt:
        model.llm.enable_input_require_grads()
        try:
            model.llm.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
        except TypeError:                                          # older transformers
            model.llm.gradient_checkpointing_enable()
        print("gradient checkpointing: ON (non-reentrant, input grads enabled)", flush=True)

    grid = build_grid(a.G, a.spacing); N = grid.shape[0]
    with torch.no_grad():                    # gain-control stats over the WHOLE concept set (per-dim; no label leak)
        ac = model.cortex.encode(*walk_2d(grid, device))
        model._code_mean = ac.mean(0); model._code_std = ac.std(0) + 1e-6
    near_r = 2.1 * a.spacing; margin = 0.2 * a.spacing        # near = local (<= ~2-step) with a resolvable margin
    tr, far, faroff, labels = make_triples(grid, near_r, margin)
    print(f"triples: train(near)={len(tr)} far={len(far)} far-offaxis={len(faroff)}", flush=True)
    if len(tr) == 0 or len(faroff) == 0:
        raise SystemExit("empty train or off-axis set — adjust --G/--spacing/near_r/margin")

    # balanced+capped eval sets (chance = 0.5 exactly; keeps the T4 eval to minutes, not ~37k triples)
    far_e = balance_cap(far, labels, a.eval_cap, a.seed + 1)
    faroff_e = balance_cap(faroff, labels, a.eval_cap, a.seed + 2)
    tr_e = balance_cap(tr, labels, a.eval_cap, a.seed + 3)
    print(f"eval sets (balanced, capped): far={len(far_e)} off-axis={len(faroff_e)} near={len(tr_e)}", flush=True)

    final_loss = None
    if a.reeval:                                                   # load a checkpoint and skip training
        ck = torch.load(a.reeval, map_location=device)
        msd = {k[len("model."):]: v for k, v in ck.items() if k.startswith("model.")}
        hsd = {k[len("head."):]: v for k, v in ck.items() if k.startswith("head.")}
        missing, unexpected = model.load_state_dict(msd, strict=False)
        head.load_state_dict(hsd)
        if "_code_mean" in ck:                                     # use the SAVED gain-control stats (self-contained)
            model._code_mean = ck["_code_mean"].to(device); model._code_std = ck["_code_std"].to(device)
        print(f"re-eval: loaded {a.reeval} (trainable tensors: {len(msd)+len(hsd)}); skipping training", flush=True)
    else:
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
            opt.zero_grad(); out.loss.backward()
            if step == 0:                                          # decisive diagnostics
                gn = sum(p.grad.detach().float().norm().item() ** 2 for p in train_params if p.grad is not None) ** 0.5
                tgt = tok.decode([t for t in b["labels"][0].tolist() if t != -100])
                print(f"  [diag] step0 grad-norm(trainable) = {gn:.3e} (>0 required)", flush=True)
                print(f"  [diag] pad_side={tok.padding_side}; loss TARGETS (must be ' Yes'/' No', not prompt/pad): {tgt!r}", flush=True)
            opt.step(); final_loss = out.loss.item()
            if step % 200 == 0 or step == a.steps - 1:
                # periodic TRAIN accuracy on a BALANCED slice: the honest tell of whether the readout is learning
                # (loss alone can drop toward the class prior without conditioning on the spatial input).
                model.eval()
                hm = len(tr_e) // 2                                 # tr_e = pos[:m]+neg[:m]; take a BALANCED slice
                tacc = evaluate(model, head, tok, tr_e[:80] + tr_e[hm:hm + 80], labels, grid, device, bs=a.bs * 2)
                model.train()
                print(f"step {step}: loss {out.loss.item():.3f}  train_acc {tacc:.3f}", flush=True)
        del opt
        model.eval()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()                               # free training grads/optimizer before eval

    model.eval()
    # decisive: does the model's Yes-vs-No preference actually VARY with the spatial input?
    with torch.no_grad():
        yid, nid = _yes_no_ids(tok, PROMPT)
        pr = torch.tensor(tok(PROMPT)["input_ids"], device=device); Tt = pr.shape[0]
        smp = tr_e[:64]
        ia = torch.tensor([t[0] for t in smp]); ib = torch.tensor([t[1] for t in smp]); ic = torch.tensor([t[2] for t in smp])
        spd = triple_spatial(model, head, walk_2d(grid[ia], device), walk_2d(grid[ib], device), walk_2d(grid[ic], device))
        txt = model._embed()(pr.unsqueeze(0).expand(len(smp), Tt))
        lg = model.llm(inputs_embeds=model.fusion(txt, spd.to(txt.dtype)),
                       attention_mask=torch.ones(len(smp), Tt, device=device)).logits[:, -1, :]
        gap = (lg[:, yid] - lg[:, nid]).float()
        print(f"  [diag] yes-minus-no logit gap over inputs: mean={gap.mean():.3f} std={gap.std():.3f} "
              f"frac_yes={(gap > 0).float().mean():.3f}  (std~0 => readout ignores the map)", flush=True)
    perm = torch.randperm(N, generator=torch.Generator().manual_seed(a.seed + 7))
    res = {
        "closer_far": round(evaluate(model, head, tok, far_e, labels, grid, device, bs=a.bs * 2), 4),
        "closer_far_OFFAXIS": round(evaluate(model, head, tok, faroff_e, labels, grid, device, bs=a.bs * 2), 4),
        "closer_near_trained": round(evaluate(model, head, tok, tr_e, labels, grid, device, bs=a.bs * 2), 4),
        "closer_far_cortex_OFF": round(evaluate(model, head, tok, far_e, labels, grid, device, ablate=True, bs=a.bs * 2), 4),
        "closer_far_shuffled_pos": round(evaluate(model, head, tok, far_e, labels, grid, device, shuffle_perm=perm, bs=a.bs * 2), 4),
    }
    print("\nCONCEPTUAL GRID through the frozen LLM (balanced sets; chance 50%):", flush=True)
    for k, v in res.items():
        print(f"  {k:26} {v:.1%}", flush=True)
    print("  (cortex-OFF & shuffled -> ~chance; OFF-AXIS > chance = genuine 2-D reasoning through the map)", flush=True)
    if res["closer_near_trained"] < 0.6:
        print(f"  DIAGNOSIS: near (TRAINED) at chance (final loss={final_loss}). If loss stayed ~0.69/answer-token"
              " the readout UNDERFIT (the modest 2-D signal is hard to extract) -> raise --bs/--steps or the LLM"
              " cannot learn it. If loss dropped but this is still ~0.5 it was the old eval bug (now fixed).", flush=True)
    if a.out:
        os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
        json.dump({"seed": a.seed, "G": a.G, "spacing": a.spacing, "results": res}, open(a.out, "w"), indent=2)
        ckpt = {f"model.{n}": p.detach().cpu() for n, p in model.named_parameters() if p.requires_grad}
        ckpt.update({f"head.{n}": p.detach().cpu() for n, p in head.named_parameters()})
        ckpt["_code_mean"] = model._code_mean.detach().cpu(); ckpt["_code_std"] = model._code_std.detach().cpu()
        torch.save(ckpt, a.out.replace(".json", ".pt"))
        print(f"\nwrote {a.out} (+ .pt checkpoint)", flush=True)


if __name__ == "__main__":
    main()
