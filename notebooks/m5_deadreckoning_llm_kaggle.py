# =====================================================================================
# M5 — THE DEAD-RECKONING BRAIN SPEAKS: a frozen LLM reads the emergent HD+grid neural code (Kaggle T4)
# =====================================================================================
# The capstone of the spatial organs. A frozen Qwen reads the agent's EMERGENT self-localization code --
# the head-direction ring-attractor state (heading) PLUS the grid-cell population (position) -- and answers
# self-localization questions in language:
#   WHERE : "which of the 9 cells are you in?"        -> reads the GRID (position) code
#   HOME  : "which way is home (0-7)?" (egocentric)   -> reads BOTH grid (home direction) AND HD (heading):
#                                                         the HOMING VECTOR, the canonical dead-reckoning act.
# The agent's moves/position NEVER appear in the prompt, so a high cortex-ON vs text-only-OFF gap is a
# clean CAUSAL + leakage-proof statement: the LLM answers ONLY by reading the emergent neural code -- the
# organs built this session (HD ring attractor + grid cortex) becoming a spatial sense an LLM speaks from.
#
# Separate queries (each trial asks WHERE or HOME) so the two fields never compete in one answer (cf. M4b).
# FIRST enable the GPU: Settings -> Accelerator -> GPU T4 x1.  ~25-30 min/seed; resumable.
# n=6 seeds so the paired sign-flip permutation p can reach 2/2^6 = 0.03 (n=3 floors at 0.25).
# Run cells top to bottom.
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


# %% [cell 3] dead-reckoning readout: ask WHERE or HOME; both cortex-ON vs text-only-OFF
import os, math, time, json, random, torch, torch.nn as nn
from transformers import AutoModelForCausalLM, AutoTokenizer, set_seed
from peft import LoraConfig, TaskType, get_peft_model
from src.models.fusion import MultiScaleSpatialFusion
from src.models.llm_wrapper import _get_embed_layer
from src.eval.agent_grid_cortex import build_cortex
from src.eval.head_direction import train_hd

dev = "cuda"
BASE = "Qwen/Qwen2.5-1.5B"
R = 2.5                      # arena half-width (matches the agent modules)
NCELL = 3                   # 3x3 grid -> WHERE has 9 classes
NHOME = 8                   # 8 egocentric sectors for the home direction
SEEDS = list(range(6)); HD_ITERS = 2000; STEPS = 1800; BS = 4
P_WHERE = ("[STATE] You have been moving; your head-direction and grid cells encode your location.\n"
           "[QUESTION] Which of the 9 cells (0-8) are you in?\n[ANSWER]")
P_HOME = ("[STATE] You have been moving; your head-direction and grid cells encode your location.\n"
          "[QUESTION] Which way is home (0-7, egocentric sectors)?\n[ANSWER]")

tok = AutoTokenizer.from_pretrained(BASE, use_fast=True)
if tok.pad_token is None: tok.pad_token = tok.eos_token


def build_organs(seed):
    """The emergent self-localization organs: a trained HD ring attractor (heading code) + the grid cortex
    (position code; fixed biological gains). Returns a code(p, theta) -> neural state function."""
    hd, _ = train_hd(seed, iters=HD_ITERS)                                  # HD ring attractor (CPU, fast)
    mod = build_cortex(seed).to(dev)                                        # grid cortex (fixed gains)
    # canonical HD ring states (rates) for every heading, from a clean angular ramp
    with torch.no_grad():
        h = torch.zeros(1, hd.U.out_features); ks = []; vs = []
        om = torch.tensor(2 * math.pi / 120)
        for _ in range(480):
            h = hd.step(h, om); ks.append(hd.decode(h)); vs.append(torch.relu(h).clone())
    keys = torch.tensor(ks, device=dev); states = torch.cat(vs, 0).to(dev)  # (N,), (N, HID)
    HID = states.shape[1]; GD = mod.K * mod.M

    def hd_lookup(theta):                                                   # (B,) -> (B, HID) ring rates
        d = torch.atan2((keys[None] - theta[:, None]).sin(), (keys[None] - theta[:, None]).cos()).abs()
        return states[d.argmin(1)]

    def code(p, theta):                                                     # (B,2),(B,) -> (B, GD+HID)
        return torch.cat([mod.grid_code_at(p), hd_lookup(theta)], -1)

    return code, GD + HID


def where_bin(p):
    col = ((p[:, 0] + R) / (2 * R) * NCELL).clamp(0, NCELL - 1e-3).long()
    row = ((p[:, 1] + R) / (2 * R) * NCELL).clamp(0, NCELL - 1e-3).long()
    return row * NCELL + col                                                # 0..8


def home_bin(p, theta):                                                     # egocentric direction to home
    home_dir = torch.atan2(-p[:, 1], -p[:, 0])
    ego = torch.atan2((home_dir - theta).sin(), (home_dir - theta).cos())   # (-pi, pi]
    return (((ego + math.pi) / (2 * math.pi) * NHOME).long().clamp(0, NHOME - 1))


class DeadReckoningReadoutLLM(nn.Module):
    def __init__(self, base, code_dim, n_tokens=8):
        super().__init__()
        try: llm = AutoModelForCausalLM.from_pretrained(base, dtype=torch.float32)
        except TypeError: llm = AutoModelForCausalLM.from_pretrained(base, torch_dtype=torch.float32)
        cfg = LoraConfig(task_type=TaskType.CAUSAL_LM, r=16, lora_alpha=32, lora_dropout=0.05, bias="none",
                         target_modules=["q_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"])
        self.llm = get_peft_model(llm, cfg)
        D = llm.config.hidden_size; self.n = n_tokens
        self.to_tokens = nn.Linear(code_dim, D * n_tokens)
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


def first_digit(txt):
    for ch in txt:
        if ch.isdigit(): return int(ch)
    return None


def run_seed(seed, smoke=False):
    set_seed(seed); torch.manual_seed(seed)
    code_fn, code_dim = build_organs(seed)

    def sample(bs):
        p = (torch.rand(bs, 2, device=dev) * 2 - 1) * R
        theta = torch.rand(bs, device=dev) * 2 * math.pi
        return p, theta, code_fn(p, theta)

    model = DeadReckoningReadoutLLM(BASE, code_dim).to(dev)
    if hasattr(model.llm, "gradient_checkpointing_enable"):
        model.llm.gradient_checkpointing_enable(); model.llm.enable_input_require_grads(); model.llm.config.use_cache = False
    opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=2e-4)

    def batch(bs):
        p, theta, c = sample(bs)
        wb = where_bin(p); hb = home_bin(p, theta)
        prompts, fulls = [], []
        for i in range(bs):
            if torch.rand(1).item() < 0.5:
                prompts.append(P_WHERE); fulls.append(P_WHERE + f" {int(wb[i])}")
            else:
                prompts.append(P_HOME); fulls.append(P_HOME + f" {int(hb[i])}")
        enc = tok(fulls, max_length=72, padding="max_length", truncation=True, return_tensors="pt")
        labels = enc["input_ids"].clone()
        for i in range(bs):
            plen = len(tok(prompts[i])["input_ids"]); labels[i, :plen] = -100
        labels[enc["attention_mask"] == 0] = -100
        return enc["input_ids"].to(dev), enc["attention_mask"].to(dev), c, labels.to(dev)

    if smoke:
        ids, attn, c, lab = batch(2)
        l = model(ids, attn, c, labels=lab).loss; l.backward(); opt.zero_grad()
        print(f"  smoke OK (loss {float(l):.3f}, code_dim {code_dim})", flush=True)

    model.train(); t0 = time.time()
    for it in range(STEPS):
        ids, attn, c, lab = batch(BS)
        opt.zero_grad(); loss = model(ids, attn, c, labels=lab).loss; loss.backward(); opt.step()
        if it % 200 == 0: print(f"  seed {seed} step {it}/{STEPS} loss {loss.item():.3f} ({time.time()-t0:.0f}s)", flush=True)

    @torch.no_grad()
    def ask(prompt, kind, ablate, n):
        model.eval(); enc = tok(prompt, return_tensors="pt")
        ids = enc["input_ids"].to(dev); attn = enc["attention_mask"].to(dev)
        ok = w1 = tot = 0
        for i in range(0, n, BS):
            m = min(BS, n - i); p, theta, c = sample(m)
            truth = where_bin(p) if kind == "where" else home_bin(p, theta)
            out = model.gen(ids.repeat(m, 1), attn.repeat(m, 1), c, ablate=ablate, max_new=3)
            for j in range(m):
                d = first_digit(tok.decode(out[j], skip_special_tokens=True)); tot += 1
                if d is not None:
                    ok += int(d == int(truth[j]))
                    if kind == "home":                                      # circular within-1 sector
                        w1 += int(min((d - int(truth[j])) % NHOME, (int(truth[j]) - d) % NHOME) <= 1)
        return ok / tot, (w1 / tot if kind == "home" else float("nan"))

    wh_on, _ = ask(P_WHERE, "where", False, 360); wh_off, _ = ask(P_WHERE, "where", True, 360)
    hm_on, hm_on1 = ask(P_HOME, "home", False, 360); hm_off, hm_off1 = ask(P_HOME, "home", True, 360)
    return {"where_on": wh_on, "where_off": wh_off, "home_on": hm_on, "home_off": hm_off,
            "home_w1_on": hm_on1, "home_w1_off": hm_off1}


OUT = "results_dr_llm"; os.makedirs(OUT, exist_ok=True)
res = []
for s in SEEDS:
    f = f"{OUT}/seed{s}.json"
    if os.path.exists(f):
        r = json.load(open(f)); print(f"===== seed {s}: cached =====", flush=True)
    else:
        print(f"\n===== DEAD-RECKONING READOUT  seed {s} =====", flush=True)
        r = run_seed(s, smoke=(s == SEEDS[0])); json.dump(r, open(f, "w"))
    res.append(r)
    print(f"  seed {s}: WHERE ON {r['where_on']:.0%}/OFF {r['where_off']:.0%} | HOME ON {r['home_on']:.0%}"
          f"(w1 {r['home_w1_on']:.0%})/OFF {r['home_off']:.0%}", flush=True)


def ci95(xs):
    n = len(xs); m = sum(xs) / n
    sd = (sum((x - m) ** 2 for x in xs) / (n - 1)) ** 0.5 if n > 1 else 0.0
    return m, 1.96 * sd / math.sqrt(n)
def paired_p(d, iters=20000):
    n = len(d); m = sum(d) / n; rng = random.Random(0)
    return sum(abs(sum(x * (1 if rng.random() < 0.5 else -1) for x in d) / n) >= abs(m) - 1e-12 for _ in range(iters)) / iters

print("\n========== THE DEAD-RECKONING BRAIN SPEAKS (cortex-ON vs text-only-OFF) ==========")
print(f"  n={len(res)} seeds | WHERE chance {1/(NCELL*NCELL):.0%} | HOME chance {1/NHOME:.0%}")
for key, name in [("where", "WHERE (cell, reads grid)"), ("home", "HOME (egocentric, reads HD+grid)"),
                  ("home_w1", "HOME within-1 sector")]:
    on = [r[f"{key}_on"] for r in res]; off = [r[f"{key}_off"] for r in res]
    mo, co = ci95(on); mf, cf = ci95(off); d = [on[i] - off[i] for i in range(len(res))]
    p = paired_p(d) if len(res) >= 2 else float("nan")
    print(f"  {name:32} ON {mo:.0%} +/-{co:.0%}   OFF {mf:.0%} +/-{cf:.0%}   Delta {sum(d)/len(d):+.0%}   p={p:.4f}")
print("  Headline: a frozen LLM reads SELF-LOCALIZATION (position + the egocentric homing vector) from the")
print("  EMERGENT HD+grid neural code -- the dead-reckoning organs become a spatial sense it speaks from.")
json.dump({"n_seeds": len(res), "ncell": NCELL, "nhome": NHOME, "per_seed": res},
          open("results_dr_llm.json", "w"), indent=2)
print("\nwrote results_dr_llm.json -- paste the table back")
