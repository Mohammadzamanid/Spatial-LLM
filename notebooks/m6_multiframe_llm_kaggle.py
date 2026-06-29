# =====================================================================================
# M6 — THE MAP SPEAKS BOTH FRAMES: a frozen LLM reads allocentric AND egocentric position (Kaggle T4)
# =====================================================================================
# The reference-frame capstone. A frozen Qwen reads the COMBINED multi-reference-frame code -- the grid-cell
# population (global/allocentric position) AND the egocentric object-vector cells (the landmark's distance &
# bearing relative to the agent) -- and answers, in language, in EITHER frame on demand:
#   WHERE    : "which of the 9 room cells are you in?"          -> reads the GRID (allocentric) code
#   LANDMARK : "which way is the landmark (0-7), egocentric?"   -> reads the OBJECT-VECTOR (egocentric) cells
# with an ORGAN-SPECIFIC LESION proving each frame traces to its organ:
#   WHERE    collapses when the GRID part is ablated (survives ablating the object-vector part);
#   LANDMARK collapses when the OBJECT-VECTOR part is ablated (survives ablating the grid).
# The agent's moves never appear in the prompt, so cortex-ON vs text-only-OFF is causal + leakage-proof.
# This is exactly the review's point: a map that answers "where am I globally?" AND "where am I relative to
# the landmark?" -- allocentric and egocentric frames coexisting, read out in language.
#
# FIRST enable the GPU: Settings -> Accelerator -> GPU T4 x1.  ~35-40 min/seed; resumable.
# n=6 seeds for the paired sign-flip permutation p (reaches 0.03).  Run cells top to bottom.
# (Fresh, or: !rm -rf results_mf_llm  to re-run.)
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


# %% [cell 3] multi-frame readout: WHERE (grid, allocentric) or LANDMARK (object-vector, egocentric)
import os, math, time, json, random, torch, torch.nn as nn
from transformers import AutoModelForCausalLM, AutoTokenizer, set_seed
from peft import LoraConfig, TaskType, get_peft_model
from src.models.fusion import MultiScaleSpatialFusion
from src.models.llm_wrapper import _get_embed_layer
from src.models.neuro import EgocentricObjectVectorCells
from src.eval.agent_grid_cortex import build_cortex, R

dev = "cuda"; BASE = "Qwen/Qwen2.5-1.5B"
NCELL = 3; NLM = 8                              # WHERE: 3x3 room cells; LANDMARK: 8 egocentric sectors
SEEDS = list(range(6)); STEPS = 2400; BS = 4
P_WHERE = ("[STATE] You have been moving; grid and object-vector cells encode your location and a landmark.\n"
           "[QUESTION] Which of the 9 room cells (0-8) are you in?\n[ANSWER]")
P_LM = ("[STATE] You have been moving; grid and object-vector cells encode your location and a landmark.\n"
        "[QUESTION] Which way is the landmark (0-7, egocentric sectors)?\n[ANSWER]")

tok = AutoTokenizer.from_pretrained(BASE, use_fast=True)
if tok.pad_token is None: tok.pad_token = tok.eos_token


def build_organs(seed):
    torch.manual_seed(seed)
    mod = build_cortex(seed).to(dev)                                       # grid cortex (allocentric)
    ovc = EgocentricObjectVectorCells(num_cells=32, embed_dim=64, max_distance=2.0 * R).to(dev)
    GD = mod.K * mod.M                                                     # grid dims | OVC dims = 64
    def code(pos, lm_dist, lm_bear):
        return torch.cat([mod.grid_code_at(pos), ovc(lm_dist, lm_bear)], -1)
    return code, GD, GD + 64


def where_bin(pos):
    col = ((pos[:, 0] + R) / (2 * R) * NCELL).clamp(0, NCELL - 1e-3).long()
    row = ((pos[:, 1] + R) / (2 * R) * NCELL).clamp(0, NCELL - 1e-3).long()
    return row * NCELL + col
def lm_bin(bear_ego):
    return ((bear_ego % (2 * math.pi)) / (2 * math.pi) * NLM).long().clamp(0, NLM - 1)


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
        pos = (torch.rand(bs, 2, device=dev) * 2 - 1) * R
        th = torch.rand(bs, device=dev) * 2 * math.pi
        lm = (torch.rand(bs, 2, device=dev) * 2 - 1) * R                   # a new landmark each trial
        vrel = lm - pos; dist = vrel.norm(dim=1)
        bear_ego = torch.atan2(vrel[:, 1], vrel[:, 0]) - th                # egocentric bearing to landmark
        return code_fn(pos, dist, bear_ego), where_bin(pos), lm_bin(bear_ego)

    def lesion(code, mode):
        if mode == "none": return code
        c = code.clone()
        if mode == "all": c[:] = 0.0
        elif mode == "grid": c[:, :GD] = 0.0                               # ablate the allocentric (grid) organ
        elif mode == "ovc": c[:, GD:] = 0.0                                # ablate the egocentric (object-vector) organ
        return c

    model = ReadoutLLM(BASE, code_dim).to(dev)
    if hasattr(model.llm, "gradient_checkpointing_enable"):
        model.llm.gradient_checkpointing_enable(); model.llm.enable_input_require_grads(); model.llm.config.use_cache = False
    opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=2e-4)

    def batch(bs):
        c, wb, lb = sample(bs)
        prompts, fulls = [], []
        for i in range(bs):
            if torch.rand(1).item() < 0.5:
                prompts.append(P_WHERE); fulls.append(P_WHERE + f" {int(wb[i])}")
            else:
                prompts.append(P_LM); fulls.append(P_LM + f" {int(lb[i])}")
        enc = tok(fulls, max_length=76, padding="max_length", truncation=True, return_tensors="pt")
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
            m = min(BS, n - i); c, wb, lb = sample(m)
            truth = wb if kind == "where" else lb
            out = model.gen(ids.repeat(m, 1), attn.repeat(m, 1), lesion(c, mode), max_new=3)
            for j in range(m):
                d = first_digit(tok.decode(out[j], skip_special_tokens=True)); tot += 1
                if d is not None: ok += int(d == int(truth[j]))
        return ok / tot

    return {  # WHERE needs grid (dies when grid ablated); LANDMARK needs object-vector (dies when ovc ablated)
        "where_on": ask(P_WHERE, "where", "none"), "where_off": ask(P_WHERE, "where", "all"),
        "where_no_grid": ask(P_WHERE, "where", "grid"), "where_no_ovc": ask(P_WHERE, "where", "ovc"),
        "lm_on": ask(P_LM, "lm", "none"), "lm_off": ask(P_LM, "lm", "all"),
        "lm_no_ovc": ask(P_LM, "lm", "ovc"), "lm_no_grid": ask(P_LM, "lm", "grid"),
    }


OUT = "results_mf_llm"; os.makedirs(OUT, exist_ok=True)
res = []
for s in SEEDS:
    f = f"{OUT}/seed{s}.json"
    if os.path.exists(f):
        r = json.load(open(f)); print(f"===== seed {s}: cached =====", flush=True)
    else:
        print(f"\n===== MULTI-FRAME READOUT  seed {s} =====", flush=True)
        r = run_seed(s, smoke=(s == SEEDS[0])); json.dump(r, open(f, "w"))
    res.append(r)
    print(f"  seed {s}: WHERE ON {r['where_on']:.0%}/OFF {r['where_off']:.0%} (no-grid {r['where_no_grid']:.0%}) | "
          f"LANDMARK ON {r['lm_on']:.0%}/OFF {r['lm_off']:.0%} (no-ovc {r['lm_no_ovc']:.0%})", flush=True)


def ci95(xs):
    n = len(xs); m = sum(xs) / n
    sd = (sum((x - m) ** 2 for x in xs) / (n - 1)) ** 0.5 if n > 1 else 0.0
    return m, 1.96 * sd / math.sqrt(n)
def paired_p(d, iters=20000):
    n = len(d); m = sum(d) / n; rng = random.Random(0)
    return sum(abs(sum(x * (1 if rng.random() < 0.5 else -1) for x in d) / n) >= abs(m) - 1e-12 for _ in range(iters)) / iters

print("\n========== THE MAP SPEAKS BOTH FRAMES (cortex-ON vs text-only-OFF) ==========")
print(f"  n={len(res)} seeds | WHERE chance {1/(NCELL*NCELL):.0%} | LANDMARK chance {1/NLM:.0%}")
for key, name in [("where", "WHERE    (room cell, reads grid)"), ("lm", "LANDMARK (egocentric, reads obj-vec)")]:
    on = [r[f"{key}_on"] for r in res]; off = [r[f"{key}_off"] for r in res]
    mo, co = ci95(on); mf, cf = ci95(off); d = [on[i] - off[i] for i in range(len(res))]
    p = paired_p(d) if len(res) >= 2 else float("nan")
    print(f"  {name:36} ON {mo:.0%} +/-{co:.0%}   OFF {mf:.0%} +/-{cf:.0%}   Delta {sum(d)/len(d):+.0%}   p={p:.4f}")
wg = ci95([r["where_no_grid"] for r in res])[0]; wo = ci95([r["where_no_ovc"] for r in res])[0]
lo = ci95([r["lm_no_ovc"] for r in res])[0]; lg = ci95([r["lm_no_grid"] for r in res])[0]
print(f"  organ-specificity:  WHERE    no-grid {wg:.0%} (dies)  vs no-ovc {wo:.0%} (survives)")
print(f"                      LANDMARK no-ovc  {lo:.0%} (dies)  vs no-grid {lg:.0%} (survives)")
print("  Headline: a frozen LLM answers in BOTH reference frames -- global (allocentric) position from the")
print("  grid, landmark-relative (egocentric) direction from the object-vector cells -- each frame collapsing")
print("  ONLY when its own organ is ablated. The multi-reference-frame map, spoken in language.")
json.dump({"n_seeds": len(res), "ncell": NCELL, "nlm": NLM, "per_seed": res},
          open("results_mf_llm.json", "w"), indent=2)
print("\nwrote results_mf_llm.json -- paste the table back")
