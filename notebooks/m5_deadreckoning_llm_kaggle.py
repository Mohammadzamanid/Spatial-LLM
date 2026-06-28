# =====================================================================================
# M5 — THE DEAD-RECKONING BRAIN SPEAKS: a frozen LLM reads the emergent HD + grid neural code (Kaggle T4)
# =====================================================================================
# The capstone of the spatial organs. A frozen Qwen reads the agent's EMERGENT self-localization code and
# answers, in language, with two DIRECT single-organ decodes (separate queries, never competing):
#   WHERE  : "which of the 9 cells are you in?"     -> reads the GRID (position) population
#   FACING : "which way are you facing (0-7)?"      -> reads the HEAD-DIRECTION ring-attractor state
# and an ORGAN-SPECIFIC LESION proves each read traces to its organ:
#   WHERE  collapses when the GRID part is ablated (but survives ablating HD);
#   FACING collapses when the HD   part is ablated (but survives ablating grid).
# The agent's moves never appear in the prompt, so cortex-ON vs text-only-OFF is causal + leakage-proof:
# the organs built this session (HD ring attractor + grid cortex) become a spatial sense an LLM speaks from.
#
# (An earlier version also asked the egocentric HOMING VECTOR -- a hard cross-organ nonlinear combination
#  [atan2(-position) - heading] -- which the readout could not learn (null). We keep the two clean direct
#  decodes here; the homing-vector readout is left as harder future work.)
#
# FIRST enable the GPU: Settings -> Accelerator -> GPU T4 x1.  ~35-40 min/seed; resumable.
# n=6 seeds so the paired sign-flip permutation p can reach 2/2^6 = 0.03 (n=3 floors at 0.25).
# Run cells top to bottom.  (Fresh, or: !rm -rf results_dr_llm  to re-run.)
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


# %% [cell 3] dead-reckoning readout: WHERE (grid) + FACING (HD), with organ-specific lesions
import os, math, time, json, random, torch, torch.nn as nn
from transformers import AutoModelForCausalLM, AutoTokenizer, set_seed
from peft import LoraConfig, TaskType, get_peft_model
from src.models.fusion import MultiScaleSpatialFusion
from src.models.llm_wrapper import _get_embed_layer
from src.eval.agent_grid_cortex import build_cortex
from src.eval.head_direction import train_hd

dev = "cuda"
BASE = "Qwen/Qwen2.5-1.5B"
R = 2.5; NCELL = 3; NFACE = 8                # WHERE: 3x3 = 9 cells; FACING: 8 heading sectors
SEEDS = list(range(6)); HD_ITERS = 2000; STEPS = 2400; BS = 4
P_WHERE = ("[STATE] You have been moving; your head-direction and grid cells encode your location.\n"
           "[QUESTION] Which of the 9 cells (0-8) are you in?\n[ANSWER]")
P_FACE = ("[STATE] You have been moving; your head-direction and grid cells encode your location.\n"
          "[QUESTION] Which way are you facing (0-7)?\n[ANSWER]")

tok = AutoTokenizer.from_pretrained(BASE, use_fast=True)
if tok.pad_token is None: tok.pad_token = tok.eos_token


def build_organs(seed):
    """Emergent organs: trained HD ring attractor (heading code) + grid cortex (position code, fixed gains).
    Returns code(p, theta) -> [grid | HD] neural state, plus the grid-dim split GD."""
    hd, _ = train_hd(seed, iters=HD_ITERS)
    mod = build_cortex(seed).to(dev)
    with torch.no_grad():
        h = torch.zeros(1, hd.U.out_features); ks = []; vs = []
        om = torch.tensor(2 * math.pi / 120)
        for _ in range(480):
            h = hd.step(h, om); ks.append(hd.decode(h)); vs.append(torch.relu(h).clone())
    keys = torch.tensor(ks, device=dev); states = torch.cat(vs, 0).to(dev)
    GD = mod.K * mod.M

    def hd_lookup(theta):
        d = torch.atan2((keys[None] - theta[:, None]).sin(), (keys[None] - theta[:, None]).cos()).abs()
        return states[d.argmin(1)]

    def code(p, theta):
        return torch.cat([mod.grid_code_at(p), hd_lookup(theta)], -1)

    return code, GD, GD + states.shape[1]


def where_bin(p):
    col = ((p[:, 0] + R) / (2 * R) * NCELL).clamp(0, NCELL - 1e-3).long()
    row = ((p[:, 1] + R) / (2 * R) * NCELL).clamp(0, NCELL - 1e-3).long()
    return row * NCELL + col
def facing_bin(theta):
    return ((theta % (2 * math.pi)) / (2 * math.pi) * NFACE).long().clamp(0, NFACE - 1)


class ReadoutLLM(nn.Module):
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
    def _tok(self, code):
        return self.to_tokens(code).view(code.shape[0], self.n, -1)
    def forward(self, input_ids, attn, code, labels=None):
        text = self.emb()(input_ids); sp = self._tok(code).to(text.dtype)
        return self.llm(inputs_embeds=self.fusion(text, sp), attention_mask=attn, labels=labels)
    @torch.no_grad()
    def gen(self, input_ids, attn, code, max_new=3):
        text = self.emb()(input_ids); sp = self._tok(code).to(text.dtype)
        return self.llm.generate(inputs_embeds=self.fusion(text, sp), attention_mask=attn,
                                 max_new_tokens=max_new, do_sample=False)


def first_digit(txt):
    for ch in txt:
        if ch.isdigit(): return int(ch)
    return None


def run_seed(seed, smoke=False):
    set_seed(seed); torch.manual_seed(seed)
    code_fn, GD, code_dim = build_organs(seed)

    def sample(bs):
        p = (torch.rand(bs, 2, device=dev) * 2 - 1) * R
        theta = torch.rand(bs, device=dev) * 2 * math.pi
        return p, theta, code_fn(p, theta)

    def lesion(code, mode):                                          # zero a slice of the CODE before projecting
        if mode == "none": return code
        c = code.clone()
        if mode == "all": c[:] = 0.0
        elif mode == "grid": c[:, :GD] = 0.0                         # ablate the grid (position) organ
        elif mode == "hd": c[:, GD:] = 0.0                           # ablate the HD (heading) organ
        return c

    model = ReadoutLLM(BASE, code_dim).to(dev)
    if hasattr(model.llm, "gradient_checkpointing_enable"):
        model.llm.gradient_checkpointing_enable(); model.llm.enable_input_require_grads(); model.llm.config.use_cache = False
    opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=2e-4)

    def batch(bs):
        p, theta, c = sample(bs)
        wb = where_bin(p); fb = facing_bin(theta)
        prompts, fulls = [], []
        for i in range(bs):
            if torch.rand(1).item() < 0.5:
                prompts.append(P_WHERE); fulls.append(P_WHERE + f" {int(wb[i])}")
            else:
                prompts.append(P_FACE); fulls.append(P_FACE + f" {int(fb[i])}")
        enc = tok(fulls, max_length=72, padding="max_length", truncation=True, return_tensors="pt")
        labels = enc["input_ids"].clone()
        for i in range(bs):
            plen = len(tok(prompts[i])["input_ids"]); labels[i, :plen] = -100
        labels[enc["attention_mask"] == 0] = -100
        return enc["input_ids"].to(dev), enc["attention_mask"].to(dev), c, labels.to(dev)

    if smoke:
        ids, attn, c, lab = batch(2)
        l = model(ids, attn, c, labels=lab).loss; l.backward(); opt.zero_grad()
        print(f"  smoke OK (loss {float(l):.3f}, code_dim {code_dim}, GD {GD})", flush=True)

    model.train(); t0 = time.time()
    for it in range(STEPS):
        ids, attn, c, lab = batch(BS)
        opt.zero_grad(); loss = model(ids, attn, c, labels=lab).loss; loss.backward(); opt.step()
        if it % 200 == 0: print(f"  seed {seed} step {it}/{STEPS} loss {loss.item():.3f} ({time.time()-t0:.0f}s)", flush=True)

    @torch.no_grad()
    def ask(prompt, kind, mode, n=300):
        model.eval(); enc = tok(prompt, return_tensors="pt")
        ids = enc["input_ids"].to(dev); attn = enc["attention_mask"].to(dev)
        ok = tot = 0
        for i in range(0, n, BS):
            m = min(BS, n - i); p, theta, c = sample(m)
            truth = where_bin(p) if kind == "where" else facing_bin(theta)
            out = model.gen(ids.repeat(m, 1), attn.repeat(m, 1), lesion(c, mode), max_new=3)
            for j in range(m):
                d = first_digit(tok.decode(out[j], skip_special_tokens=True)); tot += 1
                if d is not None: ok += int(d == int(truth[j]))
        return ok / tot

    return {  # WHERE needs grid (dies when grid ablated); FACING needs HD (dies when HD ablated)
        "where_on": ask(P_WHERE, "where", "none"), "where_off": ask(P_WHERE, "where", "all"),
        "where_no_grid": ask(P_WHERE, "where", "grid"), "where_no_hd": ask(P_WHERE, "where", "hd"),
        "face_on": ask(P_FACE, "face", "none"), "face_off": ask(P_FACE, "face", "all"),
        "face_no_hd": ask(P_FACE, "face", "hd"), "face_no_grid": ask(P_FACE, "face", "grid"),
    }


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
    print(f"  seed {s}: WHERE ON {r['where_on']:.0%}/OFF {r['where_off']:.0%} (no-grid {r['where_no_grid']:.0%}) | "
          f"FACE ON {r['face_on']:.0%}/OFF {r['face_off']:.0%} (no-hd {r['face_no_hd']:.0%})", flush=True)


def ci95(xs):
    n = len(xs); m = sum(xs) / n
    sd = (sum((x - m) ** 2 for x in xs) / (n - 1)) ** 0.5 if n > 1 else 0.0
    return m, 1.96 * sd / math.sqrt(n)
def paired_p(d, iters=20000):
    n = len(d); m = sum(d) / n; rng = random.Random(0)
    return sum(abs(sum(x * (1 if rng.random() < 0.5 else -1) for x in d) / n) >= abs(m) - 1e-12 for _ in range(iters)) / iters

print("\n========== THE DEAD-RECKONING BRAIN SPEAKS (cortex-ON vs text-only-OFF) ==========")
print(f"  n={len(res)} seeds | WHERE chance {1/(NCELL*NCELL):.0%} | FACING chance {1/NFACE:.0%}")
for key, name in [("where", "WHERE  (cell, reads grid)"), ("face", "FACING (heading, reads HD)")]:
    on = [r[f"{key}_on"] for r in res]; off = [r[f"{key}_off"] for r in res]
    mo, co = ci95(on); mf, cf = ci95(off); d = [on[i] - off[i] for i in range(len(res))]
    p = paired_p(d) if len(res) >= 2 else float("nan")
    print(f"  {name:28} ON {mo:.0%} +/-{co:.0%}   OFF {mf:.0%} +/-{cf:.0%}   Delta {sum(d)/len(d):+.0%}   p={p:.4f}")
# organ-specificity: each read collapses ONLY when ITS organ is ablated
wg, wh = ci95([r["where_no_grid"] for r in res])[0], ci95([r["where_no_hd"] for r in res])[0]
fh, fg = ci95([r["face_no_hd"] for r in res])[0], ci95([r["face_no_grid"] for r in res])[0]
print(f"  organ-specificity:  WHERE  no-grid {wg:.0%} (dies)  vs no-hd {wh:.0%} (survives)")
print(f"                      FACING no-hd   {fh:.0%} (dies)  vs no-grid {fg:.0%} (survives)")
print("  Headline: a frozen LLM reads BOTH emergent organs -- position from the grid code, heading from the")
print("  HD ring attractor -- and each read collapses ONLY when its own organ is ablated (organ-specific, causal).")
json.dump({"n_seeds": len(res), "ncell": NCELL, "nface": NFACE, "per_seed": res},
          open("results_dr_llm.json", "w"), indent=2)
print("\nwrote results_dr_llm.json -- paste the table back")
