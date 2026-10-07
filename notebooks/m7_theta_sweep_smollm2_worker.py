"""SmolLM2 theta-sweep replication worker.

Protocol V2 fixes two issues discovered in the diagnostic run:
1) forced-choice 0-vs-1 logits replace free generation, so answer formatting cannot affect accuracy;
2) fixed 1600 training steps; multiple workers may run independent seeds on separate GPUs.

Launch with CUDA_VISIBLE_DEVICES set externally, e.g.
CUDA_VISIBLE_DEVICES=0 python -u notebooks/m7_theta_sweep_smollm2_worker.py --seeds 0 2 4 6
"""
import argparse, json, math, os, random, time, sys
from pathlib import Path

# Make repository root importable even when this file is executed as notebooks/<script>.py.
REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM, AutoTokenizer, set_seed
from peft import LoraConfig, TaskType, get_peft_model

from src.models.fusion import MultiScaleSpatialFusion
from src.models.llm_wrapper import _get_embed_layer
from src.models.neuro.theta_sweep import ThetaSweepSampler
from src.eval.agent_grid_cortex import build_cortex, R

DEV = "cuda"
BASE = "HuggingFaceTB/SmolLM2-1.7B-Instruct"
STEPS = 1600
BS = 8
LR = 2e-4
WARMUP = 200
OBS_SIG = 0.4
SENSE_NOISE = 0.20
N_CUR = 4
N_EVAL = 400
OUTDIR = Path("results_sweep_llm_smollm2_v2")
P = ("[STATE] You sense your current cell and a theta-sweep look-ahead of the space in front of you.\n"
     "[QUESTION] Is the path directly ahead blocked? Answer 1 for yes, 0 for no.\n[ANSWER]")


def obstacle_sense(pos, centers):
    return torch.exp(-((pos - centers) ** 2).sum(-1) / (2 * OBS_SIG ** 2))


def make_organs(seed):
    mod = build_cortex(seed).to(DEV)
    sampler = ThetaSweepSampler()
    return mod, sampler, mod.K * mod.M


def sweep_codes(mod, sampler, pos, head, centers, gen):
    length = sampler.sweep_frac * sampler.spacings(mod).mean()
    ks = torch.arange(1, sampler.steps + 1, device=DEV) / sampler.steps
    toks, truth = [], []
    for cyc in (0, 1):
        side = -1.0 if cyc == 0 else 1.0
        direction = head + side * sampler.angle
        d = torch.stack([direction.cos(), direction.sin()], -1)
        swept = pos.unsqueeze(1) + ks.view(1, -1, 1) * length * d.unsqueeze(1)
        code = mod.grid_code_at(swept.reshape(-1, 2)).view(pos.shape[0], sampler.steps, -1)
        sense = obstacle_sense(swept, centers.unsqueeze(1))
        noisy = (sense + torch.randn(sense.shape, device=DEV, generator=gen) * SENSE_NOISE).unsqueeze(-1)
        toks.append(torch.cat([code, noisy], -1))
        truth.append(sense)
    return torch.cat(toks, 1), torch.cat(truth, 1).max(1).values


class LookaheadLLM(nn.Module):
    def __init__(self, base, token_dim, n_cur=N_CUR):
        super().__init__()
        try:
            llm = AutoModelForCausalLM.from_pretrained(base, dtype=torch.float32)
        except TypeError:
            llm = AutoModelForCausalLM.from_pretrained(base, torch_dtype=torch.float32)
        cfg = LoraConfig(
            task_type=TaskType.CAUSAL_LM, r=16, lora_alpha=32, lora_dropout=0.05, bias="none",
            target_modules=["q_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        )
        self.llm = get_peft_model(llm, cfg)
        D = llm.config.hidden_size
        self.n_cur = n_cur
        self.cur_to_tokens = nn.Linear(token_dim, D * n_cur)
        self.sweep_to_token = nn.Linear(token_dim, D)
        self.fusion = MultiScaleSpatialFusion(hidden_dim=D, num_heads=8, num_layers=2, gate_init=2.0)
        self._emb = []

    def emb(self):
        if not self._emb:
            self._emb.append(_get_embed_layer(self.llm.base_model))
        return self._emb[0]

    def tokens(self, cur, sweep):
        ct = self.cur_to_tokens(cur).view(cur.shape[0], self.n_cur, -1)
        st = self.sweep_to_token(sweep)
        return torch.cat([ct, st], 1)

    def forward(self, input_ids, attn, cur, sweep, labels=None):
        text = self.emb()(input_ids)
        sp = self.tokens(cur, sweep).to(text.dtype)
        return self.llm(
            inputs_embeds=self.fusion(text, sp),
            attention_mask=attn,
            labels=labels,
        )


def label_token_id(tok, label):
    ids = tok(" " + label, add_special_tokens=False)["input_ids"]
    if len(ids) == 1:
        return ids[0]
    ids = tok(label, add_special_tokens=False)["input_ids"]
    if len(ids) == 1:
        return ids[0]
    raise RuntimeError(f"Label {label!r} is not a single token for {BASE}: {ids}")


def run_seed(seed):
    set_seed(seed)
    torch.manual_seed(seed)
    mod, sampler, KM = make_organs(seed)
    token_dim = KM + 1
    gen = torch.Generator(device=DEV).manual_seed(seed + 11)

    tok = AutoTokenizer.from_pretrained(BASE, use_fast=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    id0, id1 = label_token_id(tok, "0"), label_token_id(tok, "1")
    print(f"  forced-choice token ids: 0={id0}, 1={id1}", flush=True)

    def sample(bs):
        pos = (torch.rand(bs, 2, device=DEV, generator=gen) * 2 - 1) * (R * 0.55)
        head = torch.rand(bs, device=DEV, generator=gen) * 2 * math.pi
        length = sampler.sweep_frac * sampler.spacings(mod).mean()
        blocked = torch.rand(bs, device=DEV, generator=gen) < 0.5
        side = (torch.randint(0, 2, (bs,), device=DEV, generator=gen) * 2 - 1).float()
        cdir = head + side * sampler.angle
        far = pos + length * torch.stack([cdir.cos(), cdir.sin()], -1)
        far = far + torch.randn(bs, 2, device=DEV, generator=gen) * 0.08
        elsewhere = (torch.rand(bs, 2, device=DEV, generator=gen) * 2 - 1) * (R * 0.7)
        centers = torch.where(blocked.unsqueeze(-1), far, elsewhere)
        cur_sense = (
            obstacle_sense(pos, centers)
            + torch.randn(bs, device=DEV, generator=gen) * SENSE_NOISE
        ).unsqueeze(-1)
        cur = torch.cat([mod.grid_code_at(pos), cur_sense], -1)
        real, truth = sweep_codes(mod, sampler, pos, head, centers, gen)
        bad = torch.rand(bs, device=DEV, generator=gen) * 2 * math.pi
        shuf, _ = sweep_codes(mod, sampler, pos, bad, centers, gen)
        free = obstacle_sense(pos, centers) < 0.35
        y = (truth > 0.5).long()
        return cur[free], real[free], shuf[free], y[free]

    def balanced(bs):
        while True:
            cur, real, shuf, y = sample(bs * 4)
            pos_i = (y == 1).nonzero(as_tuple=True)[0]
            neg_i = (y == 0).nonzero(as_tuple=True)[0]
            m = min(len(pos_i), len(neg_i), bs // 2)
            if m > 0:
                idx = torch.cat([pos_i[:m], neg_i[:m]])
                return cur[idx], real[idx], shuf[idx], y[idx]

    def lesion(cur, real, shuf, mode):
        z_c, z_s = torch.zeros_like(cur), torch.zeros_like(real)
        if mode == "on":
            return cur, real
        if mode == "off":
            return z_c, z_s
        if mode == "no_sweep":
            return cur, z_s
        if mode == "shuffle":
            return cur, shuf
        raise ValueError(mode)

    model = LookaheadLLM(BASE, token_dim).to(DEV)
    if hasattr(model.llm, "gradient_checkpointing_enable"):
        model.llm.gradient_checkpointing_enable()
        model.llm.enable_input_require_grads()
        model.llm.config.use_cache = False
    opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=LR)

    def train_batch(bs):
        cur, real, shuf, y = balanced(bs)
        fulls = [P + f" {int(y[i])}" for i in range(len(y))]
        enc = tok(fulls, max_length=64, padding="max_length", truncation=True, return_tensors="pt")
        labels = enc["input_ids"].clone()
        plen = len(tok(P)["input_ids"])
        labels[:, :plen] = -100
        labels[enc["attention_mask"] == 0] = -100
        return (
            enc["input_ids"].to(DEV), enc["attention_mask"].to(DEV),
            cur, real, shuf, labels.to(DEV),
        )

    ids, attn, cur, real, shuf, lab = train_batch(BS)
    loss = model(ids, attn, cur, real, labels=lab).loss
    loss.backward()
    opt.zero_grad(set_to_none=True)
    print(f"  smoke OK (loss {loss.detach().item():.3f}, token_dim {token_dim})", flush=True)

    model.train()
    t0 = time.time()
    for it in range(STEPS):
        for group in opt.param_groups:
            group["lr"] = LR * min(1.0, (it + 1) / WARMUP)
        ids, attn, cur, real, shuf, lab = train_batch(BS)
        opt.zero_grad(set_to_none=True)
        loss = model(ids, attn, cur, real, labels=lab).loss
        loss.backward()
        opt.step()
        if it % 200 == 0:
            print(
                f"  seed {seed} step {it}/{STEPS} loss {loss.detach().item():.3f} "
                f"({time.time() - t0:.0f}s)", flush=True
            )

    prompt = tok(P, return_tensors="pt")
    prompt_ids = prompt["input_ids"].to(DEV)
    prompt_attn = prompt["attention_mask"].to(DEV)

    @torch.no_grad()
    def acc(mode, n=N_EVAL):
        model.eval()
        ok = total = 0
        while total < n:
            cur, real, shuf, y = balanced(BS)
            c, s = lesion(cur, real, shuf, mode)
            b = len(y)
            ids = prompt_ids.repeat(b, 1)
            attn = prompt_attn.repeat(b, 1)
            out = model(ids, attn, c, s)
            last = attn.sum(1) - 1
            logits = out.logits[torch.arange(b, device=DEV), last]
            pred = (logits[:, id1] > logits[:, id0]).long()
            take = min(b, n - total)
            ok += (pred[:take] == y[:take]).sum().item()
            total += take
        return ok / total

    result = {
        "seed": seed,
        "protocol": "smollm2_v2_forced_choice",
        "steps": STEPS,
        "n_eval": N_EVAL,
        "on": acc("on"),
        "off": acc("off"),
        "no_sweep": acc("no_sweep"),
        "shuffle": acc("shuffle"),
    }
    del model, mod
    torch.cuda.empty_cache()
    return result


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, nargs="+", required=True)
    args = ap.parse_args()
    OUTDIR.mkdir(exist_ok=True)
    print("visible GPU:", torch.cuda.get_device_name(0), "| seeds:", args.seeds, flush=True)
    for seed in args.seeds:
        path = OUTDIR / f"seed{seed}.json"
        if path.exists():
            r = json.loads(path.read_text())
            print(f"===== seed {seed}: cached V2 =====", flush=True)
        else:
            print(f"\n===== V2 LOOK-AHEAD READOUT seed {seed} =====", flush=True)
            r = run_seed(seed)
            path.write_text(json.dumps(r, indent=2))
        print(
            f"  seed {seed}: ON {r['on']:.1%} | OFF {r['off']:.1%} | "
            f"no-sweep {r['no_sweep']:.1%} | shuffle {r['shuffle']:.1%}",
            flush=True,
        )


if __name__ == "__main__":
    main()
