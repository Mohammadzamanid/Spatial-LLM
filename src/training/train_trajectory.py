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

from ..data.trajectory_qa import (PROMPT, QUESTION, RETURN_TOL, TrajectoryQADataset,
                                   collate, make_trajectory_qa)
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

    # ---- Pre-train + freeze the cortex (the LLM then reads a CLEAN spatial rep) ----
    # Learning path integration from a single yes/no token collapses to the class prior,
    # so we teach the cortex to integrate FIRST, then freeze it. Two protocols:
    #   selfsup    : predict the place-cell code of the DESTINATION (a fixed sensory
    #                function of position). NO coordinates are ever given — the model
    #                must recover position internally. Biologically faithful, the way
    #                grid/place codes emerge (Banino et al. 2018; Cueva & Wei 2018).
    #   supervised : regress directly to the ground-truth final (x,y,z). A scaffold that
    #                uses position labels the brain never receives (kept as a baseline).
    def _final_pos(H, S, V):
        return torch.stack([(S * H.cos()).sum(1), (S * H.sin()).sum(1), V.sum(1)], dim=-1)

    def _place_code(pos, centers, sigma=0.9):
        d2 = ((pos.unsqueeze(1) - centers.unsqueeze(0)) ** 2).sum(-1)
        return torch.exp(-d2 / (2 * sigma ** 2))

    if a.cortex_pretrain != "none":
        cortex = model.cortex
        fin_tr = _final_pos(Htr, Str, Vtr).to(device)
        fin_va = _final_pos(Hva, Sva, Vva).to(device)
        mse = torch.nn.MSELoss()
        if a.cortex_pretrain == "selfsup":
            K, cg = 128, torch.Generator().manual_seed(0)
            centers = (torch.rand(K, 3, generator=cg) * 5 - 2.5).to(device)   # fixed environment
            place_head = torch.nn.Linear(a.cortex_dim, K).to(device)
            params = list(cortex.parameters()) + list(place_head.parameters())
        else:
            params = list(cortex.parameters())
        copt = torch.optim.Adam(params, lr=3e-3)
        cortex.train()
        for _ in range(a.cortex_epochs):
            perm = torch.randperm(a.n_train)
            for i in range(0, a.n_train, 256):
                idx = perm[i:i + 256]
                copt.zero_grad()
                h = cortex.encode(Htr[idx].to(device), Str[idx].to(device), Vtr[idx].to(device))
                if a.cortex_pretrain == "selfsup":
                    loss = mse(place_head(h), _place_code(fin_tr[idx], centers))
                else:
                    loss = mse(cortex.readout(h), fin_tr[idx])
                loss.backward()
                copt.step()
        for p in cortex.parameters():
            p.requires_grad_(False)
        # Probe (diagnostic): is the ANSWER readable from the FROZEN rep? For selfsup this
        # is the key evidence the cortex learned navigation WITHOUT coordinate labels.
        # (A linear position probe would mislead — place-cell codes encode position
        # nonlinearly — so we probe the actual target with a small MLP.)
        with torch.no_grad():
            htr_f = cortex.encode(Htr.to(device), Str.to(device), Vtr.to(device))
            hva_f = cortex.encode(Hva.to(device), Sva.to(device), Vva.to(device))
        ytr_b = (fin_tr.norm(dim=-1) < RETURN_TOL).float()
        yva_b = (fin_va.norm(dim=-1) < RETURN_TOL).float()
        probe = torch.nn.Sequential(torch.nn.Linear(a.cortex_dim, 64), torch.nn.ReLU(),
                                    torch.nn.Linear(64, 1)).to(device)
        popt = torch.optim.Adam(probe.parameters(), lr=1e-2)
        bce = torch.nn.BCEWithLogitsLoss()
        for _ in range(300):
            popt.zero_grad(); bce(probe(htr_f).squeeze(-1), ytr_b).backward(); popt.step()
        with torch.no_grad():
            pacc = ((probe(hva_f).squeeze(-1) > 0).float() == yva_b).float().mean().item()
        print(f"cortex pre-trained [{a.cortex_pretrain}] -> frozen; 'back-at-start' readable "
              f"from rep: probe acc={pacc:.1%} (chance ~{max(yva_b.mean().item(), 1-yva_b.mean().item()):.0%})",
              flush=True)

    ds = TrajectoryQADataset(Htr, Str, Vtr, atr)
    opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=a.lr)

    model.train()
    if a.cortex_pretrain != "none":
        model.cortex.eval()   # keep the frozen cortex deterministic
    nsteps = (len(ds) + a.bs - 1) // a.bs
    print(f"starting training: {a.epochs} epochs x {nsteps} steps (logging every 25)", flush=True)
    for ep in range(a.epochs):
        perm = torch.randperm(len(ds))
        tot = 0.0
        for bi, i in enumerate(range(0, len(ds), a.bs)):
            batch = collate([ds[j] for j in perm[i:i + a.bs].tolist()], tok)
            batch = {k: v.to(device) for k, v in batch.items()}
            opt.zero_grad()
            loss = model(**batch).loss
            loss.backward()
            opt.step()
            tot += loss.item()
            if bi % 25 == 0:
                print(f"  ep{ep+1} step {bi}/{nsteps}  loss={loss.item():.3f}", flush=True)
        full = evaluate(model, tok, Hva, Sva, Vva, ava, device, ablate=False)
        print(f"epoch {ep+1}/{a.epochs}  loss={tot/nsteps:.3f}  val_acc(full)={full:.1%}", flush=True)

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
    ap.add_argument("--cortex_pretrain", choices=["selfsup", "supervised", "none"], default="selfsup",
                    help="selfsup = place-cell prediction (no coordinate labels, biologically "
                         "faithful); supervised = regress final (x,y,z); none = end-to-end")
    ap.add_argument("--cortex_epochs", type=int, default=80)
    main(ap.parse_args())
