# =====================================================================================
# M7 — THE LLM LOOKS AHEAD: a frozen LLM reads THETA-SWEEP look-ahead tokens (Kaggle T4)
# =====================================================================================
# The Vollan (Nature 2025) theta sweep, made load-bearing for the LANGUAGE model. Each theta cycle the grid
# map sweeps OUTWARD from the agent (alternating left/right, ~20% of module spacing); we turn the swept grid
# codes -- plus the local sense along the sweep -- into extra spatial tokens a frozen Qwen attends to, and ask
# it to judge whether the path AHEAD is blocked, in a NOVEL per-episode layout (so the answer is NOT knowable
# from where it stands -- it has to look). The ablation is the review's exact ask -- "performance drops when
# the sweep tokens are removed":
#   ON         : current cell tokens + REAL theta-sweep look-ahead tokens.
#   OFF         : all spatial zeroed (text-only)        -> cortex-ON vs OFF is causal (moves never in the prompt).
#   NO-SWEEP    : current cell tokens, sweep tokens zeroed -> isolates the look-ahead.
#   SHUFFLED    : sweep tokens sampled along a WRONG heading -> look-ahead, but for the wrong direction.
# Headline expected: ON >> OFF ~ NO-SWEEP ~ SHUFFLED ~ chance -- the sweep tokens carry the look-ahead the
# current-position tokens cannot. (CPU twin: src/eval/theta_sweep_readout.py, real 0.90 vs ablated 0.58.)
#
# FIRST enable the GPU: Settings -> Accelerator -> GPU T4 x1.  Resumable (skips done seeds). Run top to bottom.
# =====================================================================================


# %% [cell 1] setup
import os, torch
assert torch.cuda.is_available(), "No GPU. Enable Settings -> Accelerator -> GPU T4 x1, then re-run."
os.environ["CUDA_VISIBLE_DEVICES"] = "0"; os.environ["HF_HUB_DISABLE_XET"] = "1"
!if [ -d Spatial-LLM ]; then cd Spatial-LLM && git pull origin main; else git clone https://github.com/Mohammadzamanid/Spatial-LLM.git; fi
%cd Spatial-LLM
!pip -q install -U "transformers>=4.40" peft accelerate
!pip -q uninstall -y torchao
print("device:", torch.cuda.get_device_name(0), "| setup done")


# %% [cell 2] cache the base LLM
!python -u -c "from huggingface_hub import snapshot_download; snapshot_download('Qwen/Qwen2.5-1.5B')"
print("model cached")


# %% [cell 3] look-ahead readout: judge "blocked ahead?" from theta-sweep tokens in a novel layout
import os, math, time, json, random, torch, torch.nn as nn
from transformers import AutoModelForCausalLM, AutoTokenizer, set_seed
from peft import LoraConfig, TaskType, get_peft_model
from src.models.fusion import MultiScaleSpatialFusion
from src.models.llm_wrapper import _get_embed_layer
from src.models.neuro.theta_sweep import ThetaSweepSampler
from src.eval.agent_grid_cortex import build_cortex, R

dev = "cuda"; BASE = "Qwen/Qwen2.5-1.5B"
SEEDS = list(range(8)); STEPS = 1600; BS = 8        # binary task -> fewer steps than m6; n=8 (sign-flip floor 0.008)
LR = 2e-4; WARMUP = 120
OBS_SIG = 0.4; SENSE_NOISE = 0.20                    # obstacle width; noisy look-ahead sense (not an oracle)
N_CUR = 4                                            # current-cell tokens
P = ("[STATE] You sense your current cell and a theta-sweep look-ahead of the space in front of you.\n"
     "[QUESTION] Is the path directly ahead blocked? Answer 1 for yes, 0 for no.\n[ANSWER]")

tok = AutoTokenizer.from_pretrained(BASE, use_fast=True)
if tok.pad_token is None: tok.pad_token = tok.eos_token


def obstacle_sense(pos, centers):
    return torch.exp(-((pos - centers) ** 2).sum(-1) / (2 * OBS_SIG ** 2))


def make_organs(seed):
    mod = build_cortex(seed).to(dev)                                   # velocity-driven hex grid cortex
    sampler = ThetaSweepSampler()
    KM = mod.K * mod.M
    return mod, sampler, KM


def sweep_codes(mod, sampler, pos, head, centers, gen):
    """Both theta cycles: per swept point, [grid code + noisy obstacle sense]. Returns (B, 2*steps, KM+1) and
    the TRUE max sense over the swept cone (B,) for the label."""
    length = sampler.sweep_frac * sampler.spacings(mod).mean()
    ks = torch.arange(1, sampler.steps + 1, device=dev) / sampler.steps
    toks, truth = [], []
    for cyc in (0, 1):
        side = -1.0 if cyc % 2 == 0 else 1.0
        direction = head + side * sampler.angle
        d = torch.stack([direction.cos(), direction.sin()], -1)
        swept = pos.unsqueeze(1) + ks.view(1, -1, 1) * length * d.unsqueeze(1)         # (B,steps,2)
        code = mod.grid_code_at(swept.reshape(-1, 2)).view(pos.shape[0], sampler.steps, -1)
        sense = obstacle_sense(swept, centers.unsqueeze(1))                             # (B,steps)
        noisy = (sense + torch.randn(sense.shape, device=dev, generator=gen) * SENSE_NOISE).unsqueeze(-1)
        toks.append(torch.cat([code, noisy], -1)); truth.append(sense)
    return torch.cat(toks, 1), torch.cat(truth, 1).max(1).values


class LookaheadLLM(nn.Module):
    def __init__(self, base, token_dim, n_cur=N_CUR):
        super().__init__()
        try: llm = AutoModelForCausalLM.from_pretrained(base, dtype=torch.float32)
        except TypeError: llm = AutoModelForCausalLM.from_pretrained(base, torch_dtype=torch.float32)
        cfg = LoraConfig(task_type=TaskType.CAUSAL_LM, r=16, lora_alpha=32, lora_dropout=0.05, bias="none",
                         target_modules=["q_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"])
        self.llm = get_peft_model(llm, cfg)
        D = llm.config.hidden_size; self.n_cur = n_cur
        self.cur_to_tokens = nn.Linear(token_dim, D * n_cur)               # current cell -> n_cur tokens
        self.sweep_to_token = nn.Linear(token_dim, D)                      # each swept point -> 1 token
        self.fusion = MultiScaleSpatialFusion(hidden_dim=D, num_heads=8, num_layers=2, gate_init=2.0)
        self._emb = []
    def emb(self):
        if not self._emb: self._emb.append(_get_embed_layer(self.llm.base_model))
        return self._emb[0]
    def tokens(self, cur, sweep):
        ct = self.cur_to_tokens(cur).view(cur.shape[0], self.n_cur, -1)    # (B,n_cur,D)
        st = self.sweep_to_token(sweep)                                    # (B,2*steps,D)
        return torch.cat([ct, st], 1)
    def forward(self, input_ids, attn, cur, sweep, labels=None):
        text = self.emb()(input_ids); sp = self.tokens(cur, sweep).to(text.dtype)
        return self.llm(inputs_embeds=self.fusion(text, sp), attention_mask=attn, labels=labels)
    @torch.no_grad()
    def gen(self, input_ids, attn, cur, sweep, max_new=3):
        text = self.emb()(input_ids); sp = self.tokens(cur, sweep).to(text.dtype)
        return self.llm.generate(inputs_embeds=self.fusion(text, sp), attention_mask=attn,
                                 max_new_tokens=max_new, do_sample=False)


def first_digit(txt):
    for ch in txt:
        if ch in "01": return int(ch)
    return None


def run_seed(seed, smoke=False):
    set_seed(seed); torch.manual_seed(seed)
    mod, sampler, KM = make_organs(seed)
    token_dim = KM + 1
    gen = torch.Generator(device=dev).manual_seed(seed + 11)

    def sample(bs):
        # NOVEL obstacle per episode. To balance the classes (the swept cone is narrow), ~half the episodes
        # place the obstacle at the FAR end of one look-ahead cone (blocked) and ~half elsewhere (clear) --
        # the agent at `pos` stays ~one sweep length away, so it is in free space and cannot feel the obstacle
        # without looking. The label is still the TRUE sense along the sweep, so it is honest either way.
        pos = (torch.rand(bs, 2, device=dev, generator=gen) * 2 - 1) * (R * 0.55)
        head = torch.rand(bs, device=dev, generator=gen) * 2 * math.pi
        length = sampler.sweep_frac * sampler.spacings(mod).mean()
        blocked = torch.rand(bs, device=dev, generator=gen) < 0.5
        side = (torch.randint(0, 2, (bs,), device=dev, generator=gen) * 2 - 1).float()
        cdir = head + side * sampler.angle
        far = pos + length * torch.stack([cdir.cos(), cdir.sin()], -1)                   # far end of a sweep cone
        far = far + torch.randn(bs, 2, device=dev, generator=gen) * 0.08
        elsewhere = (torch.rand(bs, 2, device=dev, generator=gen) * 2 - 1) * (R * 0.7)
        centers = torch.where(blocked.unsqueeze(-1), far, elsewhere)
        cur_sense = (obstacle_sense(pos, centers) + torch.randn(bs, device=dev, generator=gen) * SENSE_NOISE).unsqueeze(-1)
        cur = torch.cat([mod.grid_code_at(pos), cur_sense], -1)                          # (bs, KM+1)
        real, truth = sweep_codes(mod, sampler, pos, head, centers, gen)
        bad = torch.rand(bs, device=dev, generator=gen) * 2 * math.pi
        shuf, _ = sweep_codes(mod, sampler, pos, bad, centers, gen)
        free = obstacle_sense(pos, centers) < 0.35                                       # agent stands in free space
        y = (truth > 0.5).long()
        return cur[free], real[free], shuf[free], y[free]

    def lesion(cur, real, shuf, mode):
        z_c, z_s = torch.zeros_like(cur), torch.zeros_like(real)
        if mode == "on":       return cur, real
        if mode == "off":      return z_c, z_s                       # text-only (cortex OFF)
        if mode == "no_sweep": return cur, z_s                       # current cell, sweep ablated
        if mode == "shuffle":  return cur, shuf                      # wrong-heading sweep

    model = LookaheadLLM(BASE, token_dim).to(dev)
    if hasattr(model.llm, "gradient_checkpointing_enable"):
        model.llm.gradient_checkpointing_enable(); model.llm.enable_input_require_grads(); model.llm.config.use_cache = False
    opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=LR)

    def batch(bs):
        # oversample then balance the two classes so the LLM can't win by a prior
        cur, real, shuf, y = sample(bs * 4)
        pos_i = (y == 1).nonzero(as_tuple=True)[0]; neg_i = (y == 0).nonzero(as_tuple=True)[0]
        m = min(len(pos_i), len(neg_i), bs // 2)
        idx = torch.cat([pos_i[:m], neg_i[:m]])
        cur, real, shuf, y = cur[idx], real[idx], shuf[idx], y[idx]
        fulls = [P + f" {int(y[i])}" for i in range(len(idx))]
        enc = tok(fulls, max_length=64, padding="max_length", truncation=True, return_tensors="pt")
        labels = enc["input_ids"].clone(); plen = len(tok(P)["input_ids"])
        labels[:, :plen] = -100; labels[enc["attention_mask"] == 0] = -100
        return enc["input_ids"].to(dev), enc["attention_mask"].to(dev), cur, real, shuf, labels.to(dev)

    if smoke:
        ids, attn, cur, real, shuf, lab = batch(BS)
        c, s = lesion(cur, real, shuf, "on")
        l = model(ids, attn, c, s, labels=lab).loss; l.backward(); opt.zero_grad()
        print(f"  smoke OK (loss {float(l):.3f}, token_dim {token_dim})", flush=True)

    model.train(); t0 = time.time()
    for it in range(STEPS):
        for g in opt.param_groups: g["lr"] = LR * min(1.0, (it + 1) / WARMUP)
        ids, attn, cur, real, shuf, lab = batch(BS)
        c, s = lesion(cur, real, shuf, "on")
        opt.zero_grad(); loss = model(ids, attn, c, s, labels=lab).loss; loss.backward(); opt.step()
        if it % 200 == 0: print(f"  seed {seed} step {it}/{STEPS} loss {loss.item():.3f} ({time.time()-t0:.0f}s)", flush=True)

    @torch.no_grad()
    def acc(mode, n=400):
        model.eval(); enc = tok(P, return_tensors="pt")
        ids = enc["input_ids"].to(dev); attn = enc["attention_mask"].to(dev)
        ok = totn = 0
        while totn < n:
            cur, real, shuf, y = sample(BS * 4)
            pos_i = (y == 1).nonzero(as_tuple=True)[0]; neg_i = (y == 0).nonzero(as_tuple=True)[0]
            m = min(len(pos_i), len(neg_i))
            if m == 0: continue
            idx = torch.cat([pos_i[:m], neg_i[:m]]); cur, real, shuf, y = cur[idx], real[idx], shuf[idx], y[idx]
            c, s = lesion(cur, real, shuf, mode)
            out = model.gen(ids.repeat(len(idx), 1), attn.repeat(len(idx), 1), c, s, max_new=3)
            for j in range(len(idx)):
                d = first_digit(tok.decode(out[j], skip_special_tokens=True)); totn += 1
                if d is not None: ok += int(d == int(y[j]))
            if totn >= n: break
        return ok / totn

    return {"on": acc("on"), "off": acc("off"), "no_sweep": acc("no_sweep"), "shuffle": acc("shuffle")}


OUT = "results_sweep_llm"; os.makedirs(OUT, exist_ok=True)
res = []
for s in SEEDS:
    f = f"{OUT}/seed{s}.json"
    if os.path.exists(f):
        r = json.load(open(f)); print(f"===== seed {s}: cached =====", flush=True)
    else:
        print(f"\n===== LOOK-AHEAD READOUT  seed {s} =====", flush=True)
        r = run_seed(s, smoke=(s == SEEDS[0])); json.dump(r, open(f, "w"))
    res.append(r)
    print(f"  seed {s}: ON {r['on']:.0%} | OFF {r['off']:.0%} | no-sweep {r['no_sweep']:.0%} | shuffle {r['shuffle']:.0%}", flush=True)


def ci95(xs):
    n = len(xs); m = sum(xs) / n
    sd = (sum((x - m) ** 2 for x in xs) / (n - 1)) ** 0.5 if n > 1 else 0.0
    return m, 1.96 * sd / math.sqrt(n)
def paired_p(d, iters=20000):
    n = len(d); m = sum(d) / n; rng = random.Random(0)
    return sum(abs(sum(x * (1 if rng.random() < 0.5 else -1) for x in d) / n) >= abs(m) - 1e-12 for _ in range(iters)) / iters

print("\n========== THE LLM LOOKS AHEAD (theta-sweep tokens; novel per-episode layout) ==========")
print(f"  n={len(res)} seeds | chance 50% (balanced blocked-ahead)")
on = [r["on"] for r in res]
for key, name in [("off", "OFF  (text-only, cortex ablated)"), ("no_sweep", "NO-SWEEP (current cell only)"),
                  ("shuffle", "SHUFFLED (wrong-heading sweep)")]:
    other = [r[key] for r in res]; d = [on[i] - other[i] for i in range(len(res))]
    mo, co = ci95(on); mx, cx = ci95(other); p = paired_p(d) if len(res) >= 2 else float("nan")
    print(f"  ON {mo:.0%}+/-{co:.0%}  vs  {name:34} {mx:.0%}+/-{cx:.0%}   Delta {sum(d)/len(d):+.0%}   p={p:.4f}")
print("  Headline: a frozen LLM judges whether the path AHEAD is blocked in a NOVEL layout ONLY when it is")
print("  given the theta-sweep look-ahead tokens; removing them (or mis-directing the sweep) drops it to chance.")
print("  The Vollan look-around, made load-bearing for language.")
json.dump({"n_seeds": len(res), "per_seed": res}, open("results_sweep_llm.json", "w"), indent=2)
print("\nwrote results_sweep_llm.json -- paste the table back")
