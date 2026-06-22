# =====================================================================================
# M3 — THE TEMPORAL MAP, end to end on a Kaggle T4.  Two experiments, run cells top to bottom.
#
#   A) EMERGENCE (cell 3): a generic recurrent substrate (TemporalCortex: leaky rectified
#      rate-RNN, ONE uniform time-constant, learned recurrence, private noise — nothing
#      timing-specific) is trained on ONE task, "report elapsed time when probed," with a
#      metabolic cost.  We then MEASURE what appeared — time cells, field widening, scalar
#      (Weber) timing — none of it in the loss.  vs an untrained control.
#
#   B) LANGUAGE READOUT (cell 4): freeze that emergent temporal code and let a LoRA-Qwen
#      answer "how much time has elapsed?" reading the cortex ONLY (the elapsed time is
#      never in the text).  cortex-ON vs text-only-OFF = a causal, leakage-proof statement
#      that the LLM reads the emergent time code.  Multi-seed, EXACT + WITHIN-1, paired test.
#
# FIRST enable the GPU: Settings -> Accelerator -> GPU T4 x1.  ~5-8 min (cell 3) + ~25 min/seed (cell 4).
# Nothing is hard-coded: the neuroscience EMERGES and is measured.
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


# %% [cell 2] cache the base LLM (only needed for cell 4)
!python -u -c "from huggingface_hub import snapshot_download; snapshot_download('Qwen/Qwen2.5-1.5B')"
print("model cached")


# %% [cell 3] EXPERIMENT A — do time cells + scalar (Weber) timing EMERGE?  (~5-8 min)
import math, time, torch, torch.nn as nn
import matplotlib.pyplot as plt
from src.models.neuro.temporal_cortex import TemporalCortex      # the generic substrate (committed)

dev = "cuda"
T = 50; HIDDEN = 128; NOISE = 0.06; ACT_COST = 1e-3; SEEDS = 4; ITERS = 2000; BATCH = 96

def make_trial(B):
    x = torch.zeros(B, T, 2, device=dev); x[:, 0, 0] = 1.0           # start pulse
    probe = torch.randint(T // 5, T, (B,), device=dev)
    x[torch.arange(B, device=dev), probe, 1] = 1.0                   # "report elapsed time now"
    return x, probe

def ridge(A, y, lam=1.0):
    Ab = torch.cat([A, torch.ones(A.shape[0], 1, device=A.device)], 1)
    return torch.linalg.solve(Ab.t() @ Ab + lam * torch.eye(Ab.shape[1], device=A.device), Ab.t() @ y)

def corr(a, b):
    a = a - a.mean(); b = b - b.mean(); return (a @ b / (a.norm() * b.norm() + 1e-9)).item()

@torch.no_grad()
def measure(net, n=800):
    x, _ = make_trial(n); R = net.dynamics(x, noise=NOISE); A = R.mean(0)
    ts = torch.arange(T, device=dev).float()
    W = ridge(A, ts); that = torch.cat([R, torch.ones(n, T, 1, device=dev)], -1) @ W
    mae = (that.mean(0) - ts).abs().mean().item(); sigma = that.std(0)
    mid = (ts > 5) & (ts < T - 5); cv = sigma[mid] / ts[mid]
    weber_cv = (cv.std(unbiased=True) / (cv.mean() + 1e-9)).item()
    Ar = A / (A.max(0).values + 1e-6); peak = Ar.argmax(0).float(); width = (Ar > 0.5).float().sum(0)
    w = int(0.1 * T)
    near = torch.stack([Ar[max(0, int(p) - w):int(p) + w + 1, u].sum() for u, p in enumerate(peak)])
    is_tc = (A.max(0).values > 0.05 * A.max()) & (near / (Ar.sum(0) + 1e-6) > 0.5) & (width < T * 0.5) & (peak > 1) & (peak < T - 2)
    tc = is_tc.nonzero().squeeze(-1)
    wcorr = corr(peak[tc], width[tc]) if len(tc) > 5 else float("nan")
    early = (peak[tc] < T / 2).float().mean().item() if len(tc) else float("nan")
    return dict(mae=mae, frac=is_tc.float().mean().item(), wcorr=wcorr, weber_cv=weber_cv, early=early,
                arr=dict(Ar=Ar.cpu(), tc=tc.cpu(), peak=peak.cpu(), width=width.cpu(), sigma=sigma.cpu(), ts=ts.cpu()))

rows = []; arr0 = carr0 = None; t0 = time.time()
for s in range(SEEDS):
    torch.manual_seed(s)
    net = TemporalCortex(hidden=HIDDEN).to(dev); opt = torch.optim.Adam(net.parameters(), 3e-3)
    for it in range(ITERS):
        x, probe = make_trial(BATCH); pred, R = net(x, noise=NOISE)
        pred = pred[torch.arange(BATCH, device=dev), probe].squeeze(-1)
        loss = ((pred - probe.float() / T) ** 2).mean() + ACT_COST * R.pow(2).mean()
        opt.zero_grad(); loss.backward(); opt.step()
        if it % 500 == 0: print(f"  seed {s} iter {it}/{ITERS} loss {loss.item():.4f} ({time.time()-t0:.0f}s)", flush=True)
    tr = measure(net); ct = measure(TemporalCortex(hidden=HIDDEN).to(dev))    # untrained control
    tr["ctrl_mae"], tr["ctrl_weber_cv"], tr["ctrl_frac"] = ct["mae"], ct["weber_cv"], ct["frac"]
    if s == 0: arr0, carr0 = tr["arr"], ct["arr"]
    rows.append(tr)
    print(f"seed {s}: time-cells {tr['frac']:.0%} (untrained {tr['ctrl_frac']:.0%}) | widen {tr['wcorr']:+.2f} | "
          f"WeberCV {tr['weber_cv']:.2f} | MAE {tr['mae']:.2f} vs ctrl {tr['ctrl_mae']:.1f}", flush=True)

def agg(k):
    v = torch.tensor([r[k] for r in rows if r[k] == r[k]])
    return v.mean().item(), (1.96 * v.std(unbiased=True) / math.sqrt(len(v))).item() if len(v) > 1 else 0.0
print(f"\n=== EMERGENT TIME CODE (n={SEEDS}; mean +/- 95% CI) ===")
for k, name in [("mae", "decode MAE steps (precise timer EMERGED)"), ("ctrl_mae", "  untrained: cannot time"),
                ("frac", "time-cell fraction"), ("ctrl_frac", "  untrained time-cell fraction"),
                ("early", "  peaking in first half (denser-early; Mau 2018)"),
                ("wcorr", "FIELD WIDENING corr (emergent)"), ("weber_cv", "Weber-fraction CV (LOW=Weber's law)"),
                ("ctrl_weber_cv", "  untrained Weber CV")]:
    m, c = agg(k); print(f"  {name:46} {m:+.3f} +/- {c:.3f}")

fig, ax = plt.subplots(1, 3, figsize=(16, 4.2))
order = arr0["tc"][arr0["peak"][arr0["tc"]].argsort()]
ax[0].imshow(arr0["Ar"][:, order].T.numpy(), aspect="auto", cmap="magma", origin="lower", extent=[0, T, 0, len(order)])
ax[0].set_title(f"Emergent time cells ({len(order)}), sorted by peak"); ax[0].set_xlabel("elapsed time"); ax[0].set_ylabel("cell")
ax[1].scatter(arr0["peak"][arr0["tc"]].numpy(), arr0["width"][arr0["tc"]].numpy(), s=14, c="#2ca25f")
ax[1].set_title(f"Fields WIDEN with latency (corr {agg('wcorr')[0]:+.2f})"); ax[1].set_xlabel("peak time"); ax[1].set_ylabel("field width")
ax[2].plot(arr0["ts"][2:-2].numpy(), arr0["sigma"][2:-2].numpy(), c="#2ca25f", lw=2.4, label=f"trained (MAE {agg('mae')[0]:.2f})")
ax[2].plot(carr0["ts"][2:-2].numpy(), carr0["sigma"][2:-2].numpy(), c="#9aa5b8", lw=2.0, label=f"untrained (MAE {agg('ctrl_mae')[0]:.1f})")
ax[2].set_title("Scalar (Weber) timing"); ax[2].set_xlabel("elapsed time"); ax[2].set_ylabel("decoded-time SD"); ax[2].legend()
plt.tight_layout(); plt.savefig("emergent_time.png", dpi=130); plt.show()
print("saved emergent_time.png")


# %% [cell 4] EXPERIMENT B — read the EMERGENT time code through a frozen LLM (multi-seed)
# cortex-ON vs text-only-OFF, EXACT + WITHIN-1 (elapsed time is a scalar -> within-1 is the
# natural metric), n seeds with 95% CI + a paired test. Mirrors the proven torus-QA path.
# ~25-30 min PER SEED on a T4. Set SEEDS=[0] for a quick single-seed pass.
import os, math, time, json, random, torch, torch.nn as nn
from transformers import AutoModelForCausalLM, AutoTokenizer, set_seed
from peft import LoraConfig, TaskType, get_peft_model
from src.models.fusion import MultiScaleSpatialFusion
from src.models.llm_wrapper import _get_embed_layer
from src.models.neuro.temporal_cortex import TemporalCortex

dev = "cuda"
T = 50; HIDDEN = 128; NOISE = 0.06; ACT_COST = 1e-3; C = 6        # 6 elapsed-time bins (answer 0..5)
BASE = "Qwen/Qwen2.5-1.5B"
PROMPT = "[INTERVAL] Time has passed since a start signal.\n[QUESTION] How much time has elapsed? Answer 0 to 5.\n[ANSWER]"
SEEDS = [0, 1, 2]; CORTEX_ITERS = 2000; STEPS = 1500; BS = 4   # resumable; use list(range(6)) to
# clear the paired-permutation p-floor (n=3 floors at 2/2^3=0.25; n=6 -> 2/2^6=0.03, like torus-QA)

tok = AutoTokenizer.from_pretrained(BASE, use_fast=True)
if tok.pad_token is None: tok.pad_token = tok.eos_token

class TemporalReadoutLLM(nn.Module):
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
    def gen(self, input_ids, attn, code, ablate=False, max_new=3):
        text = self.emb()(input_ids); sp = self._tok(code, ablate).to(text.dtype)
        return self.llm.generate(inputs_embeds=self.fusion(text, sp), attention_mask=attn,
                                 max_new_tokens=max_new, do_sample=False)

def make_pulse(B):
    x = torch.zeros(B, T, 2, device=dev); x[:, 0, 0] = 1.0
    probe = torch.randint(T // 5, T, (B,), device=dev); x[torch.arange(B, device=dev), probe, 1] = 1.0
    return x, probe
def bin_of(probe): return [min(int(p.item() / T * C), C - 1) for p in probe]

def run_seed(seed, smoke=False):
    set_seed(seed); torch.manual_seed(seed)
    # 1. train + FREEZE the temporal cortex on the elapsed-time task (emergent code)
    cortex = TemporalCortex(hidden=HIDDEN).to(dev); copt = torch.optim.Adam(cortex.parameters(), 3e-3)
    for it in range(CORTEX_ITERS):
        x, probe = make_pulse(96); pred, R = cortex(x, noise=NOISE)
        pred = pred[torch.arange(96, device=dev), probe].squeeze(-1)
        loss = ((pred - probe.float() / T) ** 2).mean() + ACT_COST * R.pow(2).mean()
        copt.zero_grad(); loss.backward(); copt.step()
    for p in cortex.parameters(): p.requires_grad_(False)
    cortex.eval()

    @torch.no_grad()
    def temporal_code(probe):                                    # (B,) -> (B,HIDDEN) noisy emergent code
        B = probe.shape[0]; x = torch.zeros(B, T, 2, device=dev); x[:, 0, 0] = 1.0
        R = cortex.dynamics(x, noise=NOISE); return R[torch.arange(B, device=dev), probe]

    # 2. frozen Qwen + LoRA + gated fusion of the cortex tokens
    model = TemporalReadoutLLM(BASE, HIDDEN).to(dev)
    if hasattr(model.llm, "gradient_checkpointing_enable"):
        model.llm.gradient_checkpointing_enable(); model.llm.enable_input_require_grads(); model.llm.config.use_cache = False
    opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=2e-4)

    def train_batch(bs):
        probe = torch.randint(T // 5, T, (bs,), device=dev); code = temporal_code(probe)
        fulls = [PROMPT + " " + str(b) for b in bin_of(probe)]
        enc = tok(fulls, max_length=48, padding="max_length", truncation=True, return_tensors="pt")
        labels = enc["input_ids"].clone(); plen = len(tok(PROMPT)["input_ids"])
        labels[:, :plen] = -100; labels[enc["attention_mask"] == 0] = -100
        return enc["input_ids"].to(dev), enc["attention_mask"].to(dev), code, labels.to(dev)

    if smoke:                                                    # one tiny step to catch bugs in seconds
        ids, attn, code, lab = train_batch(2)
        l = model(ids, attn, code, labels=lab).loss; l.backward(); opt.zero_grad()
        print(f"  smoke OK (loss {float(l):.3f})", flush=True)

    model.train(); cortex.eval(); t0 = time.time()
    for it in range(STEPS):
        ids, attn, code, lab = train_batch(BS)
        opt.zero_grad(); loss = model(ids, attn, code, labels=lab).loss; loss.backward(); opt.step()
        if it % 200 == 0: print(f"  seed {seed} step {it}/{STEPS} loss {loss.item():.3f} ({time.time()-t0:.0f}s)", flush=True)

    @torch.no_grad()
    def evaluate(ablate, n=400):
        model.eval(); enc = tok(PROMPT, return_tensors="pt")
        ids = enc["input_ids"].to(dev); attn = enc["attention_mask"].to(dev)
        exact = within1 = tot = 0
        for i in range(0, n, BS):
            m = min(BS, n - i); probe = torch.randint(T // 5, T, (m,), device=dev)
            out = model.gen(ids.repeat(m, 1), attn.repeat(m, 1), temporal_code(probe), ablate=ablate, max_new=3)
            truth = bin_of(probe)
            for j in range(m):
                txt = tok.decode(out[j], skip_special_tokens=True)
                pc = next((ch for ch in txt if ch in "012345"), None)
                tot += 1
                if pc is not None:
                    pi = int(pc)
                    exact += int(pi == truth[j]); within1 += int(abs(pi - truth[j]) <= 1)
        return exact / max(tot, 1), within1 / max(tot, 1)

    on_e, on_w = evaluate(False); off_e, off_w = evaluate(True)
    return {"on_exact": on_e, "on_w1": on_w, "off_exact": off_e, "off_w1": off_w}

OUT = "results_elapsed_llm"; os.makedirs(OUT, exist_ok=True)        # resumable: skip seeds already done
res = []
for s in SEEDS:
    f = f"{OUT}/seed{s}.json"
    if os.path.exists(f):
        r = json.load(open(f)); print(f"===== seed {s}: cached =====", flush=True)
    else:
        print(f"\n===== ELAPSED-TIME READOUT  seed {s} =====", flush=True)
        r = run_seed(s, smoke=(s == SEEDS[0])); json.dump(r, open(f, "w"))
    res.append(r)
    print(f"  seed {s}: ON exact {r['on_exact']:.0%} (w1 {r['on_w1']:.0%})  |  OFF exact {r['off_exact']:.0%} (w1 {r['off_w1']:.0%})", flush=True)

def ci95(xs):
    n = len(xs); m = sum(xs) / n
    sd = (sum((x - m) ** 2 for x in xs) / (n - 1)) ** 0.5 if n > 1 else 0.0
    return m, 1.96 * sd / math.sqrt(n)
def paired_p(d, iters=20000):
    n = len(d); m = sum(d) / n; rng = random.Random(0)
    return sum(abs(sum(x * (1 if rng.random() < 0.5 else -1) for x in d) / n) >= abs(m) - 1e-12 for _ in range(iters)) / iters

print("\n================ ELAPSED-TIME READOUT through a frozen LLM ================")
print(f"  n={len(res)} seeds | chance ~ 1/{C} = {1/C:.0%}")
for metric in ("exact", "w1"):
    on = [r[f"on_{metric}"] for r in res]; off = [r[f"off_{metric}"] for r in res]
    mo, co = ci95(on); mf, cf = ci95(off); d = [on[i] - off[i] for i in range(len(res))]
    p = paired_p(d) if len(res) >= 2 else float("nan")
    tag = "EXACT  " if metric == "exact" else "WITHIN-1"
    print(f"  {tag}  ON {mo:.0%} +/-{co:.0%}   OFF {mf:.0%} +/-{cf:.0%}   Delta {sum(d)/len(d):+.0%}   p={p:.4f}")
print("  ON >> OFF => the LLM names elapsed time by READING the emergent temporal code")
print("  (elapsed time was never in the prompt) -- the temporal analogue of torus-QA.")
json.dump({"n_seeds": len(res), "chance": 1 / C, "per_seed": res}, open("results_elapsed_time_llm.json", "w"), indent=2)
print("\nwrote results_elapsed_time_llm.json -- paste the table back")
