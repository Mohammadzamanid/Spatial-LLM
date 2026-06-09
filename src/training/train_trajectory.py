"""
src/training/train_trajectory.py  —  Milestone 2 trainer/eval.

Trains TrajectoryLLM to answer "Are you back where you started?" in language, where the
path reaches the model only through the trajectory cortex. Reports yes/no accuracy with
the cortex ON vs ABLATED (zeroed) — the ablated run is the control: if the LLM could
answer from the (question-only) text alone it would still score high; it shouldn't.

    python -m src.training.train_trajectory --base_llm Qwen/Qwen2.5-1.5B --epochs 3
"""
import argparse
import os

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0")   # single GPU, before torch inits CUDA
os.environ.setdefault("HF_HUB_DISABLE_XET", "1")

import torch
from transformers import AutoTokenizer, set_seed

from ..data.trajectory_qa import (PROMPT, QUESTION, TrajectoryQADataset, collate,
                                   make_trajectory_qa)
from ..models.trajectory_llm import TrajectoryLLM


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
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"device={device}  base_llm={a.base_llm}")

    tok = AutoTokenizer.from_pretrained(a.base_llm, use_fast=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    model = TrajectoryLLM(base_llm=a.base_llm, cortex_dim=a.cortex_dim).to(device)
    if hasattr(model.llm, "gradient_checkpointing_enable"):
        model.llm.gradient_checkpointing_enable()
        model.llm.enable_input_require_grads()
        model.llm.config.use_cache = False

    Htr, Str, Vtr, atr = make_trajectory_qa(a.n_train, T=a.T, seed=1)
    Hva, Sva, Vva, ava = make_trajectory_qa(a.n_val, T=a.T, seed=2)
    yes_frac = sum(x == "Yes." for x in ava) / len(ava)
    print(f"data: {a.n_train} train / {a.n_val} val (val 'Yes' fraction={yes_frac:.2f})")

    ds = TrajectoryQADataset(Htr, Str, Vtr, atr)
    opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=a.lr)

    model.train()
    for ep in range(a.epochs):
        perm = torch.randperm(len(ds))
        tot = 0.0
        for i in range(0, len(ds), a.bs):
            batch = collate([ds[j] for j in perm[i:i + a.bs].tolist()], tok)
            batch = {k: v.to(device) for k, v in batch.items()}
            opt.zero_grad()
            loss = model(**batch).loss
            loss.backward()
            opt.step()
            tot += loss.item()
        full = evaluate(model, tok, Hva, Sva, Vva, ava, device, ablate=False)
        print(f"epoch {ep+1}/{a.epochs}  loss={tot/(len(ds)//a.bs):.3f}  val_acc(full)={full:.1%}", flush=True)

    full = evaluate(model, tok, Hva, Sva, Vva, ava, device, ablate=False)
    abl = evaluate(model, tok, Hva, Sva, Vva, ava, device, ablate=True)
    print("\n================ RESULT ================")
    print(f"  cortex ON  : {full:.1%}")
    print(f"  cortex OFF : {abl:.1%}   (control — should fall toward chance ~{max(yes_frac,1-yes_frac):.0%})")
    print(f"  => the cortex contributes {full-abl:+.1%}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--base_llm", default="Qwen/Qwen2.5-1.5B")
    ap.add_argument("--cortex_dim", type=int, default=128)
    ap.add_argument("--n_train", type=int, default=4000)
    ap.add_argument("--n_val", type=int, default=600)
    ap.add_argument("--T", type=int, default=8)
    ap.add_argument("--epochs", type=int, default=3)
    ap.add_argument("--bs", type=int, default=4)   # fp32 1.5B fits a T4 at bs=4 + grad checkpointing
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--seed", type=int, default=42)
    main(ap.parse_args())
