"""
src/training/train_trajectory.py  —  Milestone 2 trainer/eval.

Trains TrajectoryLLM to answer a navigation question in language, where the path reaches
the model ONLY through the trajectory cortex. Reports accuracy with the cortex ON vs
ABLATED (zeroed) — the ablated run is the control: if the LLM could answer from the
(question-only) text alone it would still score high; it shouldn't.

Questions (``--task``), increasing difficulty:
  - return   : "Are you back where you started?"        -> Yes./No.   (binary; forgiving)
  - distance : "How far are you from where you started?" -> bucket 0..5 (MAGNITUDE)
  - bearing  : "Which direction is the start from here?" -> compass word (DIRECTION)
distance/bearing force the model to read the actual displacement vector, far harder than
the binary return question. For these we report EXACT and WITHIN-1 accuracy (within-1 is
circular for bearing) and enlarge the self-supervised place-cell environment.

Length generalization (the DEFAULT recipe)
-------------------------------------------
By default the model trains on a MIX of short path lengths (6,8,10,12) with a scale-free
cortex readout (readout(u), not readout(u/T)) and is evaluated on LONGER, held-out lengths
(8,16,24). This is the recommendation proven by the generalization stress-test
(``src/eval/generalize_trajectory.py``): a fixed-length cortex locks to its length, but a
scale-free cortex trained on mixed lengths extrapolates. Eval flags lengths beyond the
training range as EXTRAPOLATION.

    # default generalizing recipe, binary return question
    python -m src.training.train_trajectory --epochs 3
    # harder magnitude question
    python -m src.training.train_trajectory --task distance --epochs 3
    # harder direction question (scale-invariant -> the cleanest extrapolation test)
    python -m src.training.train_trajectory --task bearing --epochs 3
    # old length-LOCKED baseline
    python -m src.training.train_trajectory --train_lengths 8 --no-cortex_scale_free --epochs 3
"""
import argparse
import json
import os
import random
from collections import Counter

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0")   # single GPU, before torch inits CUDA
os.environ.setdefault("HF_HUB_DISABLE_XET", "1")

import torch
import torch.nn as nn
from transformers import AutoTokenizer, set_seed

from ..data.trajectory_qa import (PROMPT, QUESTIONS, TrajectoryQADataset, answer_index,
                                   collate, is_circular, make_trajectory_qa, num_classes,
                                   parse_answer)
from ..models.trajectory_llm import TrajectoryLLM


def _final_pos(H, S, V):
    """Ground-truth final (x, y, z) displacement of each path."""
    return torch.stack([(S * H.cos()).sum(1), (S * H.sin()).sum(1), V.sum(1)], dim=-1)


def _place_code(pos, centers, sigma=0.9):
    """Place-cell code of a position: Gaussian bumps over fixed centers. A sensory function
    of WHERE you ended up — identical for short and long paths ending at the same point, so
    training on it across lengths teaches length-invariance."""
    d2 = ((pos.unsqueeze(1) - centers.unsqueeze(0)) ** 2).sum(-1)
    return torch.exp(-d2 / (2 * sigma ** 2))


def _len_minibatches(by_len, bs):
    """Yield (T, idx) batches HOMOGENEOUS in length (the cortex loops over T, so a batch
    must share one T), shuffled across lengths."""
    batches = []
    for T in by_len:
        n = by_len[T][0].shape[0]
        perm = torch.randperm(n)
        for i in range(0, n, bs):
            batches.append((T, perm[i:i + bs]))
    random.shuffle(batches)
    return batches


def _labels(task, ans, device):
    """Ground-truth class index per example (long tensor)."""
    return torch.tensor([answer_index(task, parse_answer(task, x)) for x in ans],
                        dtype=torch.long, device=device)


def _chance(task, ans):
    """Most-common-class frequency (the trivial prior-best accuracy)."""
    c = Counter(answer_index(task, parse_answer(task, x)) for x in ans)
    return max(c.values()) / len(ans)


@torch.no_grad()
def evaluate(model, tok, H, S, V, ans, device, ablate, task, question, bs=16):
    """Exact and within-1 accuracy (within-1 is circular for bearing; == exact for return)."""
    model.eval()
    prompt = PROMPT.format(q=question)
    enc = tok(prompt, return_tensors="pt")
    ncls = num_classes(task)
    max_new = 6 if task == "bearing" else 4
    exact = within1 = total = 0
    for i in range(0, len(ans), bs):
        n = min(bs, len(ans) - i)
        input_ids = enc["input_ids"].repeat(n, 1).to(device)
        attn = enc["attention_mask"].repeat(n, 1).to(device)
        out = model.generate_answer(
            input_ids, attn,
            H[i:i + n].to(device), S[i:i + n].to(device), V[i:i + n].to(device),
            ablate_cortex=ablate, max_new_tokens=max_new,
        )
        for j in range(n):
            pi = answer_index(task, parse_answer(task, tok.decode(out[j], skip_special_tokens=True)))
            ti = answer_index(task, parse_answer(task, ans[i + j]))
            total += 1
            if pi is None:
                continue
            if pi == ti:
                exact += 1
                within1 += 1
            elif task != "return":
                d = abs(pi - ti)
                if is_circular(task):
                    d = min(d, ncls - d)
                if d <= 1:
                    within1 += 1
    return {"exact": exact / max(total, 1), "within1": within1 / max(total, 1)}


def main(a):
    set_seed(a.seed)
    random.seed(a.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    train_lengths = a.train_lengths or [a.T]
    eval_lengths = a.eval_lengths or [a.T]
    scale_free = a.cortex_scale_free
    task = a.task
    question = QUESTIONS[task]
    print(f"device={device}  base_llm={a.base_llm}  task={task}  Q={question!r}", flush=True)
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

    # ---- data: one QA set per length (moves never enter the text) ----
    n_per = max(a.bs, a.n_train // len(train_lengths))
    nval_per = max(16, a.n_val // len(eval_lengths))
    train_by_len = {T: make_trajectory_qa(n_per, T=T, seed=1000 + T, task=task) for T in train_lengths}
    val_by_len = {T: make_trajectory_qa(nval_per, T=T, seed=2000 + T, task=task) for T in eval_lengths}
    print(f"data: {n_per}/length train ({len(train_lengths)} lengths), "
          f"{nval_per}/length val ({len(eval_lengths)} lengths)", flush=True)

    # ---- Pre-train + freeze the cortex (the LLM then reads a CLEAN spatial rep) ----
    # Learning integration from the answer token alone collapses to the class prior, so we
    # teach the cortex to integrate FIRST, then freeze it.
    #   selfsup    : predict the place-cell code of the DESTINATION (no coordinates). Target
    #                is length-invariant, so mixed-length training teaches any-length
    #                integration (Banino 2018; Cueva 2018). Environment is enlarged for the
    #                magnitude/direction tasks so far endpoints stay in-range.
    #   supervised : regress final (x,y,z) directly (baseline scaffold; no env limit).
    if a.cortex_pretrain != "none":
        cortex = model.cortex
        mse = nn.MSELoss()
        if a.cortex_pretrain == "selfsup":
            env_half = a.env_half if a.env_half is not None else (4.0 if task != "return" else 2.5)
            K = a.n_centers if a.n_centers is not None else (512 if task != "return" else 128)
            # bumps must OVERLAP (spacing ~ sigma) or the place code is near-zero everywhere
            # and carries no position signal; magnitude/direction tasks need the denser code.
            sigma = a.place_sigma if a.place_sigma is not None else (1.2 if task != "return" else 0.9)
            cg = torch.Generator().manual_seed(0)
            centers = (torch.rand(K, 3, generator=cg) * (2 * env_half) - env_half).to(device)
            place_head = nn.Linear(a.cortex_dim, K).to(device)
            params = list(cortex.parameters()) + list(place_head.parameters())
            print(f"selfsup environment: env_half={env_half}  n_centers={K}  sigma={sigma}", flush=True)
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
                    loss = mse(place_head(h), _place_code(target, centers, sigma))
                else:
                    loss = mse(cortex.readout(h), target)
                loss.backward()
                copt.step()
        for p in cortex.parameters():
            p.requires_grad_(False)
        cortex.eval()

        # Probe (diagnostic): is the ANSWER decodable from the FROZEN rep — at each eval
        # length? A flat probe acc across lengths (incl. beyond training) is the cortex-level
        # signature of a length-invariant code, previewed before the (slow) LLM eval. Small
        # MLP classifier over the task's classes.
        @torch.no_grad()
        def _enc(by_len, lengths):
            return {T: cortex.encode(by_len[T][0].to(device), by_len[T][1].to(device),
                                     by_len[T][2].to(device)) for T in lengths}

        htr = _enc(train_by_len, train_lengths)
        htr_f = torch.cat([htr[T] for T in train_lengths])
        ytr = torch.cat([_labels(task, train_by_len[T][3], device) for T in train_lengths])
        ncls = num_classes(task)
        probe = nn.Sequential(nn.Linear(a.cortex_dim, 64), nn.ReLU(), nn.Linear(64, ncls)).to(device)
        popt = torch.optim.Adam(probe.parameters(), lr=1e-2)
        ce = nn.CrossEntropyLoss()
        for _ in range(400):
            popt.zero_grad(); ce(probe(htr_f), ytr).backward(); popt.step()
        probe_acc = {}
        with torch.no_grad():
            for T in eval_lengths:
                h = cortex.encode(val_by_len[T][0].to(device), val_by_len[T][1].to(device),
                                  val_by_len[T][2].to(device))
                y = _labels(task, val_by_len[T][3], device)
                probe_acc[T] = round((probe(h).argmax(-1) == y).float().mean().item(), 4)
        print(f"cortex frozen. '{task}' probe acc by length: "
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
            batch = collate([ds[j] for j in idx.tolist()], tok, question=question)
            batch = {k: v.to(device) for k, v in batch.items()}
            opt.zero_grad()
            loss = model(**batch).loss
            loss.backward()
            opt.step()
            tot += loss.item()
            if bi % 25 == 0:
                print(f"  ep{ep+1} step {bi}/{len(batches)} (T={T})  loss={loss.item():.3f}", flush=True)
        Tq = min(eval_lengths)
        q = evaluate(model, tok, *val_by_len[Tq][:3], val_by_len[Tq][3], device,
                     ablate=False, task=task, question=question)
        print(f"epoch {ep+1}/{a.epochs}  loss={tot/max(len(batches),1):.3f}  "
              f"val(full,T={Tq}) exact={q['exact']:.1%} within1={q['within1']:.1%}", flush=True)

    # ---- per-length eval: cortex ON vs OFF, flag extrapolation lengths ----
    max_train = max(train_lengths)
    print("\n================ RESULT (per length) ================", flush=True)
    results_by_len = {}
    for T in eval_lengths:
        H, S, V, ans = val_by_len[T]
        on = evaluate(model, tok, H, S, V, ans, device, False, task, question)
        off = evaluate(model, tok, H, S, V, ans, device, True, task, question)
        chance = _chance(task, ans)
        tag = "train-range " if T <= max_train else "EXTRAPOLATION"
        results_by_len[T] = {
            "cortex_on_exact": round(on["exact"], 4), "cortex_on_within1": round(on["within1"], 4),
            "cortex_off_exact": round(off["exact"], 4), "cortex_off_within1": round(off["within1"], 4),
            "chance": round(chance, 4), "extrapolation": T > max_train,
        }
        w1 = "" if task == "return" else f" | within1 ON={on['within1']:.1%} OFF={off['within1']:.1%}"
        print(f"  T={T:2d} [{tag}]  exact ON={on['exact']:.1%} OFF={off['exact']:.1%} "
              f"(chance~{chance:.0%}){w1}", flush=True)

    if a.out:
        os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
        with open(a.out, "w") as f:
            json.dump({
                "config": {k: v for k, v in vars(a).items()},
                "task": task, "question": question,
                "train_lengths": train_lengths, "eval_lengths": eval_lengths,
                "cortex_scale_free": scale_free, "cortex_pretrain": a.cortex_pretrain,
                "probe_acc_by_len": probe_acc, "results_by_len": results_by_len,
                "note": ("cortex ON should hold at EXTRAPOLATION lengths (> max train length) "
                         "when scale_free + mixed train_lengths; OFF is the text-only control "
                         "(~chance). distance/bearing also report within-1 accuracy."),
            }, f, indent=2)
        print(f"\nwrote {a.out}", flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--base_llm", default="Qwen/Qwen2.5-1.5B")
    ap.add_argument("--task", choices=["return", "distance", "bearing"], default="return",
                    help="return=yes/no (default); distance=bucket 0-5 (magnitude); "
                         "bearing=8-way compass home direction")
    ap.add_argument("--cortex_dim", type=int, default=128)
    ap.add_argument("--n_train", type=int, default=4000)
    ap.add_argument("--n_val", type=int, default=600)
    ap.add_argument("--T", type=int, default=8, help="single-length fallback when "
                    "--train_lengths/--eval_lengths are explicitly set to []")
    ap.add_argument("--train_lengths", type=int, nargs="+", default=[6, 8, 10, 12],
                    help="path lengths to TRAIN on. DEFAULT is mixed (generalizing recipe); "
                         "pass a single value for the length-locked baseline.")
    ap.add_argument("--eval_lengths", type=int, nargs="+", default=[8, 16, 24],
                    help="path lengths to EVALUATE on; any beyond max(train_lengths) is "
                         "held-out EXTRAPOLATION.")
    ap.add_argument("--cortex_scale_free", action=argparse.BooleanOptionalAction, default=True,
                    help="readout(u) (DEFAULT) vs --no-cortex_scale_free for readout(u/T); "
                         "scale-free + mixed lengths is what generalizes (generalize_trajectory.py)")
    ap.add_argument("--env_half", type=float, default=None,
                    help="half-width of the selfsup place-cell environment (default 2.5 for "
                         "return, 4.0 for distance/bearing)")
    ap.add_argument("--n_centers", type=int, default=None,
                    help="number of selfsup place cells (default 128 / 512)")
    ap.add_argument("--place_sigma", type=float, default=None,
                    help="place-cell width; bumps should overlap (default 0.9 return / 1.2 else)")
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
