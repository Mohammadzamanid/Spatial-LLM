# =====================================================================================
# M4 — "WHAT HAPPENED WHEN?" through a frozen LLM (Kaggle T4).  The content-binding capstone:
# a frozen LoRA-Qwen reads ONLY the content-binding cortex and names BOTH the event (what) and
# the elapsed-time bin (when) — neither is ever in the prompt.  cortex-ON vs text-only-OFF =
# a causal, leakage-proof statement that the LLM reads the emergent what-where-when code.
# The third member of the family: torus-QA (space), elapsed-time (time), and now what x when.
#
# FIRST enable the GPU: Settings -> Accelerator -> GPU T4 x1.  ~25-30 min/seed; resumable.
# Run cells top to bottom.  Nothing is hard-coded; the cortex code is emergent (see M3).
# =====================================================================================


# %% [cell 1] setup: GPU guard + clone repo + deps
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


# %% [cell 3] "what happened when?" readout: cortex-ON vs text-only-OFF (multi-seed, resumable)
import os, math, time, json, random, torch, torch.nn as nn
from transformers import AutoModelForCausalLM, AutoTokenizer, set_seed
from peft import LoraConfig, TaskType, get_peft_model
from src.models.fusion import MultiScaleSpatialFusion
from src.models.llm_wrapper import _get_embed_layer
from src.models.neuro.temporal_cortex import TemporalCortex

dev = "cuda"
T = 50; HIDDEN = 128; K = 3; C = 6; NOISE = 0.06; ACT_COST = 1e-3      # K events ("what"), C time bins ("when")
BASE = "Qwen/Qwen2.5-1.5B"
PROMPT = ("[EPISODE] An event occurred at the start; then time passed.\n"
          "[QUESTION] Which event (0-2) and how much time has elapsed (0-5)? Answer: <event> <time>.\n[ANSWER]")
SEEDS = list(range(6)); CORTEX_ITERS = 2000; STEPS = 1500; BS = 4      # resumable; n=6 -> paired p can reach 0.03

tok = AutoTokenizer.from_pretrained(BASE, use_fast=True)
if tok.pad_token is None: tok.pad_token = tok.eos_token

class TemporalReadoutLLM(nn.Module):                                    # same proven fusion path as M3/torus-QA
    def __init__(self, base, hidden, n_tokens=8):
        super().__init__()
        try: llm = AutoModelForCausalLM.from_pretrained(base, dtype=torch.float32)
        except TypeError: llm = AutoModelForCausalLM.from_pretrained(base, torch_dtype=torch.float32)
        cfg = LoraConfig(task_type=TaskType.CAUSAL_LM, r=16, lora_alpha=32, lora_dropout=0.05, bias="none",
                         target_modules=["q_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"])
        self.llm = get_peft_model(llm, cfg)
        D = llm.config.hidden_size; self.n = n_tokens
        self.to_tokens = nn.Linear(hidden, D * n_tokens)
        self.fusion = MultiScaleSpatialFusion(hidden_dim=D, num_heads=8, num_layers=2, gate_init=2.0)
        self._emb = []
    def emb(self):
        if not self._emb: self._emb.append(_get_embed_layer(self.llm.base_model))
        return self._emb[0]
    def _tok(self, code, ablate):
        t = self.to_tokens(code).view(code.shape[0], self.n, -1)
        return torch.zeros_like(t) if ablate else t
    def forward(self, input_ids, attn, code, labels=None, ablate=False):
        text = self.emb()(input_ids); sp = self._tok(code, ablate).to(text.dtype)
        return self.llm(inputs_embeds=self.fusion(text, sp), attention_mask=attn, labels=labels)
    @torch.no_grad()
    def gen(self, input_ids, attn, code, ablate=False, max_new=5):
        text = self.emb()(input_ids); sp = self._tok(code, ablate).to(text.dtype)
        return self.llm.generate(inputs_embeds=self.fusion(text, sp), attention_mask=attn,
                                 max_new_tokens=max_new, do_sample=False)

def bin_of(probe): return [min(int(p.item() / T * C), C - 1) for p in probe]
def parse(txt):
    ds = [c for c in txt if c.isdigit()]
    ev = int(ds[0]) if ds else None
    tm = int(ds[1]) if len(ds) > 1 else None
    return ev, tm

def run_seed(seed, smoke=False):
    set_seed(seed); torch.manual_seed(seed)
    # 1. train + FREEZE the CONTENT-BINDING cortex (event at t=0; report what + when)
    cx = TemporalCortex(hidden=HIDDEN, n_in=K + 1).to(dev)
    th = nn.Linear(HIDDEN, 1).to(dev); eh = nn.Linear(HIDDEN, K).to(dev)
    co = torch.optim.Adam(list(cx.parameters()) + list(th.parameters()) + list(eh.parameters()), 3e-3)
    def pulse(B):
        x = torch.zeros(B, T, K + 1, device=dev); ev = torch.randint(K, (B,), device=dev)
        x[torch.arange(B, device=dev), 0, ev] = 1.0
        probe = torch.randint(T // 5, T, (B,), device=dev); x[torch.arange(B, device=dev), probe, K] = 1.0
        return x, ev, probe
    for _ in range(CORTEX_ITERS):
        x, ev, probe = pulse(96); R = cx.dynamics(x, noise=NOISE)
        rp = R[torch.arange(96, device=dev), probe]
        loss = ((th(rp).squeeze(-1) - probe.float() / T) ** 2).mean() \
            + nn.functional.cross_entropy(eh(rp), ev) + ACT_COST * R.pow(2).mean()
        co.zero_grad(); loss.backward(); co.step()
    for p in cx.parameters(): p.requires_grad_(False)
    cx.eval()

    @torch.no_grad()
    def code(ev, probe):                                               # (B,) event ids + probe times -> (B,HIDDEN)
        B = ev.shape[0]; x = torch.zeros(B, T, K + 1, device=dev); x[torch.arange(B, device=dev), 0, ev] = 1.0
        R = cx.dynamics(x, noise=NOISE); return R[torch.arange(B, device=dev), probe]

    # 2. frozen Qwen + LoRA reads the cortex; answer "<event> <time>"
    model = TemporalReadoutLLM(BASE, HIDDEN).to(dev)
    if hasattr(model.llm, "gradient_checkpointing_enable"):
        model.llm.gradient_checkpointing_enable(); model.llm.enable_input_require_grads(); model.llm.config.use_cache = False
    opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=2e-4)

    def batch(bs):
        ev = torch.randint(K, (bs,), device=dev); probe = torch.randint(T // 5, T, (bs,), device=dev)
        c = code(ev, probe); tb = bin_of(probe)
        fulls = [PROMPT + f" {int(ev[i])} {tb[i]}" for i in range(bs)]
        enc = tok(fulls, max_length=64, padding="max_length", truncation=True, return_tensors="pt")
        labels = enc["input_ids"].clone(); plen = len(tok(PROMPT)["input_ids"])
        labels[:, :plen] = -100; labels[enc["attention_mask"] == 0] = -100
        return enc["input_ids"].to(dev), enc["attention_mask"].to(dev), c, labels.to(dev)

    if smoke:
        ids, attn, c, lab = batch(2)
        l = model(ids, attn, c, labels=lab).loss; l.backward(); opt.zero_grad()
        print(f"  smoke OK (loss {float(l):.3f})", flush=True)

    model.train(); cx.eval(); t0 = time.time()
    for it in range(STEPS):
        ids, attn, c, lab = batch(BS)
        opt.zero_grad(); loss = model(ids, attn, c, labels=lab).loss; loss.backward(); opt.step()
        if it % 200 == 0: print(f"  seed {seed} step {it}/{STEPS} loss {loss.item():.3f} ({time.time()-t0:.0f}s)", flush=True)

    @torch.no_grad()
    def evaluate(ablate, n=400):
        model.eval(); enc = tok(PROMPT, return_tensors="pt")
        ids = enc["input_ids"].to(dev); attn = enc["attention_mask"].to(dev)
        what_ok = when_ok = when_w1 = tot = 0
        for i in range(0, n, BS):
            m = min(BS, n - i); ev = torch.randint(K, (m,), device=dev); probe = torch.randint(T // 5, T, (m,), device=dev)
            out = model.gen(ids.repeat(m, 1), attn.repeat(m, 1), code(ev, probe), ablate=ablate, max_new=5)
            tb = bin_of(probe)
            for j in range(m):
                pe, pt = parse(tok.decode(out[j], skip_special_tokens=True))
                tot += 1
                what_ok += int(pe == int(ev[j]))
                if pt is not None:
                    when_ok += int(pt == tb[j]); when_w1 += int(abs(pt - tb[j]) <= 1)
        return what_ok / tot, when_ok / tot, when_w1 / tot

    won, wonE, wonW1 = evaluate(False); woff, woffE, woffW1 = evaluate(True)
    return {"what_on": won, "what_off": woff, "when_on": wonE, "when_off": woffE,
            "when_w1_on": wonW1, "when_w1_off": woffW1}

OUT = "results_what_when_llm"; os.makedirs(OUT, exist_ok=True)          # resumable
res = []
for s in SEEDS:
    f = f"{OUT}/seed{s}.json"
    if os.path.exists(f):
        r = json.load(open(f)); print(f"===== seed {s}: cached =====", flush=True)
    else:
        print(f"\n===== WHAT-WHEN READOUT  seed {s} =====", flush=True)
        r = run_seed(s, smoke=(s == SEEDS[0])); json.dump(r, open(f, "w"))
    res.append(r)
    print(f"  seed {s}: WHAT ON {r['what_on']:.0%}/OFF {r['what_off']:.0%} | WHEN ON {r['when_on']:.0%}"
          f"(w1 {r['when_w1_on']:.0%})/OFF {r['when_off']:.0%}", flush=True)

def ci95(xs):
    n = len(xs); m = sum(xs) / n
    sd = (sum((x - m) ** 2 for x in xs) / (n - 1)) ** 0.5 if n > 1 else 0.0
    return m, 1.96 * sd / math.sqrt(n)
def paired_p(d, iters=20000):
    n = len(d); m = sum(d) / n; rng = random.Random(0)
    return sum(abs(sum(x * (1 if rng.random() < 0.5 else -1) for x in d) / n) >= abs(m) - 1e-12 for _ in range(iters)) / iters

print("\n================ WHAT-HAPPENED-WHEN through a frozen LLM ================")
print(f"  n={len(res)} seeds | WHAT chance ~ 1/{K} = {1/K:.0%} | WHEN chance ~ 1/{C} = {1/C:.0%}")
for key, name in [("what", "WHAT (event)"), ("when", "WHEN exact"), ("when_w1", "WHEN within-1")]:
    on = [r[f"{key}_on"] for r in res]; off = [r[f"{key}_off"] for r in res]
    mo, co = ci95(on); mf, cf = ci95(off); d = [on[i] - off[i] for i in range(len(res))]
    p = paired_p(d) if len(res) >= 2 else float("nan")
    print(f"  {name:14} ON {mo:.0%} +/-{co:.0%}   OFF {mf:.0%} +/-{cf:.0%}   Delta {sum(d)/len(d):+.0%}   p={p:.4f}")
print("  ON >> OFF on BOTH => the LLM reads the emergent what-where-when code (neither in the prompt).")
json.dump({"n_seeds": len(res), "K": K, "C": C, "per_seed": res}, open("results_what_when_llm.json", "w"), indent=2)
print("\nwrote results_what_when_llm.json -- paste the table back")
