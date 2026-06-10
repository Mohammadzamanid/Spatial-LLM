"""
src/training/train_trajectory.py  —  Milestone 2 trainer/eval.

Trains TrajectoryLLM to answer "Are you back where you started?" in language, where the
path reaches the model only through the trajectory cortex. Reports yes/no accuracy with
the cortex ON vs ABLATED (zeroed) — the ablated run is the control: if the LLM could
answer from the (question-only) text alone it would still score high; it shouldn't.

Length generalization
----------------------
``--train_lengths`` and ``--eval_lengths`` let the model train on a MIX of (short) path
lengths and be evaluated on LONGER, held-out ones. Paired with ``--cortex_scale_free``
(readout(u) instead of readout(u/T)) this is the recommendation from the generalization
stress-test (``src/eval/generalize_trajectory.py``): a cortex trained on one fixed length
locks to it, but a scale-free cortex trained on mixed lengths extrapolates. The eval then
prints accuracy per length and flags any length beyond the training range as EXTRAPOLATION.

    # original single-length run (reproduces prior results)
    python -m src.training.train_trajectory --T 8 --epochs 3
    # length-generalization run: train short+mixed, test longer
    python -m src.training.train_trajectory --train_lengths 6 8 10 12 \
        --eval_lengths 8 16 24 --cortex_scale_free --epochs 3
"""
import argparse
import json
import os
import random

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0")   # single GPU, before torch inits CUDA
os.environ.setdefault("HF_HUB_DISABLE_XET", "1")

import torch
import torch.nn as nn
from transformers import AutoTokenizer, set_seed

from ..data.trajectory_qa import (PROMPT, QUESTION, RETURN_TOL, TrajectoryQADataset,
                                   collate, make_trajectory_qa)
from ..models.trajectory_llm import TrajectoryLLM


def _final_pos(H, S, V):
    """Ground-truth final (x, y, z) displacement of each path."""
    return torch.stack([(S * H.cos()).sum(1), (S * H.sin()).sum(1), V.sum(1)], dim=-1)


def _place_code(pos, centers, sigma=0.9):
    """Place-cell code of a position: Gaussian bumps over a fixed set of centers. A
    sensory function of WHERE you ended up — identical for short and long paths that end
    at the same point, so training on it across lengths teaches length-invariance."""
    d2 = ((pos.unsqueeze(1) - centers.unsqueeze(0)) ** 2).sum(-1)
    return torch.exp(-d2 / (2 * sigma ** 2))


def _len_minibatches(by_len, bs):
    """Yield (T, idx) batches that are HOMOGENEOUS in length (the cortex loops over T,
    so a batch must share one T), shuffled across lengths."""
    batches = []
    for T in by_len:
        n = by_len[T][0].shape[0]
        perm = torch.randperm(n)
        for i in range(0, n, bs):
            batches.append((T, perm[i:i + bs]))
    random.shuffle(batches)
    return batches


def _yesno(text):
    t = text.strip().lower()
    yi, ni = t.find("yes"), t.find("no")
    if yi == -1 and ni == -1:
        return None
    if yi == -1:
        return "no"
    if ni == -1:
        return "yes"
    return "yes" if yi < ni else "no"


@torch.no_grad()
def evaluate(model, tok, H, S, V, ans, device, ablate, bs=16, max_length=64):
    model.eval()
    prompt = PROMPT.format(q=QUESTION)
    enc = tok(prompt, return_tensors="pt")
    correct = total = 0
    for i in range(0, len(ans), bs):
        n = min(bs, len(ans) - i)
        input_ids = enc["input_ids"].repeat(n, 1).to(device)
        attn = enc["attention_mask"].repeat(n, 1).to(device)
        out = model.generate_answer(
            input_ids, attn,
            H[i:i + n].to(device), S[i:i + n].to(device), V[i:i + n].to(device),
            ablate_cortex=ablate, max_new_tokens=4,
        )
        for j in range(n):
            pred = _yesno(tok.decode(out[j], skip_special_tokens=True))
            truth = _yesno(ans[i + j])
            total += 1
            correct += int(pred == truth)
    return correct / max(total, 1)


def main(a):
    set_seed(a.seed)
    random.seed(a.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    train_lengths = a.train_lengths or [a.T]
    eval_lengths = a.eval_lengths or [a.T]
    scale_free = a.cortex_scale_free
    print(f"device={device}  base_llm={a.base_llm}", flush=True)
    print(f"train_lengths={train_lengths}  eval_lengths={eval_lengths}  "
          f"cortex_scale_free={scale_free}  cortex_pretrain={a.cortex_pretrain}", flush=True)

    tok = AutoTokenizer.from_pretrained(a.base_llm, use_fast=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    model = TrajectoryLLM(base_llm=a.base_llm, cortex_dim=a.cortex_dim,
                          cortex_length_norm=not scale_free).to(device)
    if hasattr(model.llm, "gradient_checkpointing_enable"):
        model.llm.gradient_checkpointing_enable()
        model.llm.enable_input_require_grads()
        model.llm.config.use_cache = False

    # ---- data: one balanced QA set per length (moves never enter the text) ----
    n_per = max(a.bs, a.n_train // len(train_lengths))
    nval_per = max(16, a.n_val // len(eval_lengths))
    train_by_len = {T: make_trajectory_qa(n_per, T=T, seed=1000 + T) for T in train_lengths}
    val_by_len = {T: make_trajectory_qa(nval_per, T=T, seed=2000 + T) for T in eval_lengths}
    print(f"data: {n_per}/length train ({len(train_lengths)} lengths), "
          f"{nval_per}/length val ({len(eval_lengths)} lengths)", flush=True)

    # ---- Pre-train + freeze the cortex (the LLM then reads a CLEAN spatial rep) ----
    # Learning path integration from a single yes/no token collapses to the class prior,
    # so we teach the cortex to integrate FIRST, then freeze it. Two protocols:
    #   selfsup    : predict the place-cell code of the DESTINATION (a fixed sensory
    #                function of position). NO coordinates are ever given. The target is
    #                length-invariant by construction, so training across mixed lengths
    #                teaches the cortex to integrate ANY length (Banino 2018; Cueva 2018).
    #   supervised : regress directly to the ground-truth final (x,y,z) (baseline scaffold).
    if a.cortex_pretrain != "none":
        cortex = model.cortex
        mse = nn.MSELoss()
        if a.cortex_pretrain == "selfsup":
            K, cg = 128, torch.Generator().manual_seed(0)
            centers = (torch.rand(K, 3, generator=cg) * 5 - 2.5).to(device)   # fixed environment
            place_head = nn.Linear(a.cortex_dim, K).to(device)
            params = list(cortex.parameters()) + list(place_head.parameters())
        else:
            params = list(cortex.parameters())
        copt = torch.optim.Adam(params, lr=3e-3)
        cortex.train()
        for _ in range(a.cortex_epochs):
            for T, idx in _len_minibatches(train_by_len, 256):
                H = train_by_len[T][0][idx].to(device)
                S = train_by_len[T][1][idx].to(device)
                V = train_by_len[T][2][idx].to(device)
                copt.zero_grad()
                h = cortex.encode(H, S, V)
                target = _final_pos(H, S, V)
                if a.cortex_pretrain == "selfsup":
                    loss = mse(place_head(h), _place_code(target, centers))
                else:
                    loss = mse(cortex.readout(h), target)
                loss.backward()
                copt.step()
        for p in cortex.parameters():
            p.requires_grad_(False)
        cortex.eval()

        # Probe (diagnostic): is "back-at-start" readable from the FROZEN rep — at each
        # eval length? A flat probe acc across lengths (incl. those longer than training)
        # is the cortex-level signature of a length-invariant code, previewed before the
        # (slow) LLM eval. Small MLP, since place-cell codes encode position nonlinearly.
        @torch.no_grad()
        def _enc(by_len, lengths):
            hs, ys = [], []
            for T in lengths:
                H, S, V, _ = by_len[T]
                hs.append(cortex.encode(H.to(device), S.to(device), V.to(device)))
                ys.append((_final_pos(H, S, V).norm(dim=-1) < RETURN_TOL).float().to(device))
            return hs, ys

        htr_list, ytr_list = _enc(train_by_len, train_lengths)
        htr_f, ytr_b = torch.cat(htr_list), torch.cat(ytr_list)
        probe = nn.Sequential(nn.Linear(a.cortex_dim, 64), nn.ReLU(), nn.Linear(64, 1)).to(device)
        popt = torch.optim.Adam(probe.parameters(), lr=1e-2)
        bce = nn.BCEWithLogitsLoss()
        for _ in range(300):
            popt.zero_grad(); bce(probe(htr_f).squeeze(-1), ytr_b).backward(); popt.step()
        probe_acc = {}
        with torch.no_grad():
            for T in eval_lengths:
                H, S, V, _ = val_by_len[T]
                h = cortex.encode(H.to(device), S.to(device), V.to(device))
                y = (_final_pos(H, S, V).norm(dim=-1) < RETURN_TOL).float().to(device)
                probe_acc[T] = round(((probe(h).squeeze(-1) > 0).float() == y).float().mean().item(), 4)
        print("cortex frozen. 'back-at-start' probe acc by length: "
              + "  ".join(f"T{T}:{probe_acc[T]:.1%}" for T in eval_lengths), flush=True)
    else:
        probe_acc = {}

    # ---- train the LLM to read the frozen cortex (length-homogeneous batches) ----
    train_sets = {T: TrajectoryQADataset(*train_by_len[T]) for T in train_lengths}
    opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=a.lr)
    model.train()
    if a.cortex_pretrain != "none":
        model.cortex.eval()   # keep the frozen cortex deterministic

    for ep in range(a.epochs):
        batches = _len_minibatches(train_by_len, a.bs)
        tot = 0.0
        for bi, (T, idx) in enumerate(batches):
            ds = train_sets[T]
            batch = collate([ds[j] for j in idx.tolist()], tok)
            batch = {k: v.to(device) for k, v in batch.items()}
            opt.zero_grad()
            loss = model(**batch).loss
            loss.backward()
            opt.step()
            tot += loss.item()
            if bi % 25 == 0:
                print(f"  ep{ep+1} step {bi}/{len(batches)} (T={T})  loss={loss.item():.3f}", flush=True)
        Tq = min(eval_lengths)
        q = evaluate(model, tok, *val_by_len[Tq][:3], val_by_len[Tq][3], device, ablate=False)
        print(f"epoch {ep+1}/{a.epochs}  loss={tot/max(len(batches),1):.3f}  "
              f"val_acc(full,T={Tq})={q:.1%}", flush=True)

    # ---- per-length eval: cortex ON vs OFF, flag extrapolation lengths ----
    max_train = max(train_lengths)
    print("\n================ RESULT (per length) ================", flush=True)
    results_by_len = {}
    for T in eval_lengths:
        H, S, V, ans = val_by_len[T]
        full = evaluate(model, tok, H, S, V, ans, device, ablate=False)
        abl = evaluate(model, tok, H, S, V, ans, device, ablate=True)
        yes_frac = sum(x == "Yes." for x in ans) / len(ans)
        chance = max(yes_frac, 1 - yes_frac)
        tag = "train-range " if T <= max_train else "EXTRAPOLATION"
        results_by_len[T] = {"cortex_on": round(full, 4), "cortex_off": round(abl, 4),
                             "yes_frac": round(yes_frac, 4), "extrapolation": T > max_train}
        print(f"  T={T:2d} [{tag}]  cortex ON={full:.1%}  OFF={abl:.1%}  "
              f"(chance~{chance:.0%})  => contributes {full-abl:+.1%}", flush=True)

    if a.out:
        os.makedirs(os.path.dirname(a.out), exist_ok=True)
        with open(a.out, "w") as f:
            json.dump({
                "config": {k: v for k, v in vars(a).items()},
                "train_lengths": train_lengths, "eval_lengths": eval_lengths,
                "cortex_scale_free": scale_free, "cortex_pretrain": a.cortex_pretrain,
                "probe_acc_by_len": probe_acc, "results_by_len": results_by_len,
                "note": ("cortex ON should hold at EXTRAPOLATION lengths (> max train length) "
                         "when scale_free + mixed train_lengths; OFF is the text-only control "
                         "(should sit near chance)."),
            }, f, indent=2)
        print(f"\nwrote {a.out}", flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--base_llm", default="Qwen/Qwen2.5-1.5B")
    ap.add_argument("--cortex_dim", type=int, default=128)
    ap.add_argument("--n_train", type=int, default=4000)
    ap.add_argument("--n_val", type=int, default=600)
    ap.add_argument("--T", type=int, default=8, help="single-length fallback when "
                    "--train_lengths/--eval_lengths are not given")
    ap.add_argument("--train_lengths", type=int, nargs="+", default=None,
                    help="path lengths to TRAIN on (mixed). Defaults to [--T].")
    ap.add_argument("--eval_lengths", type=int, nargs="+", default=None,
                    help="path lengths to EVALUATE on; any beyond max(train_lengths) is "
                         "held-out EXTRAPOLATION. Defaults to [--T].")
    ap.add_argument("--cortex_scale_free", action="store_true",
                    help="use readout(u) instead of readout(u/T) — needed for length "
                         "generalization (see generalize_trajectory.py)")
    ap.add_argument("--epochs", type=int, default=3)
    ap.add_argument("--bs", type=int, default=4)   # fp32 1.5B fits a T4 at bs=4 + grad checkpointing
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--cortex_pretrain", choices=["selfsup", "supervised", "none"], default="selfsup",
                    help="selfsup = place-cell prediction (no coordinate labels, biologically "
                         "faithful); supervised = regress final (x,y,z); none = end-to-end")
    ap.add_argument("--cortex_epochs", type=int, default=80)
    ap.add_argument("--out", type=str, default=None, help="optional path to write results JSON")
    main(ap.parse_args())
