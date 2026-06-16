"""
src/data/trajectory_qa.py

Trajectory question-answering data for Milestone 2.

Each example is a PATH (a sequence of moves) + a natural-language question + answer. The
moves are NEVER written in the text — they reach the model only through the trajectory
cortex — so the LLM must use the cortex's path integration to answer.

Three question types (``task=``), in increasing difficulty:
  - "return"   : "Are you back where you started?"     -> Yes./No.   (binary; forgiving)
  - "distance" : "How far are you from where you started?" -> a quantized bucket 0..5
                 (multi-class MAGNITUDE — must read |displacement|, not just near-origin)
  - "bearing"  : "Which direction is the start from here?"  -> an 8-way compass word
                 (DIRECTION; scale-invariant, so the cleanest length-extrapolation probe)

distance/bearing are much harder than the binary return question: the model must decode the
actual displacement VECTOR (how far / which way), which stresses the integrator far more.
"""
import math
import random

import torch
from torch.utils.data import Dataset

QUESTIONS = {
    "return":   "Are you back where you started?",
    "distance": "How far are you from where you started?",
    "bearing":  "Which direction is the start from here?",
    "torus":    "On a board that wraps around at its edges, which cell (0-8) are you in?",
}
QUESTION = QUESTIONS["return"]                 # backward-compatible default
PROMPT = "[NAVIGATION] You walked a path through space.\n[QUESTION] {q}\n[ANSWER]"
RETURN_TOL = 0.5      # final displacement below this counts as "back at start"
DIST_MAX_BUCKET = 5   # distance bucket saturates at "5" (= 5 or more)
TORUS_L = 1.6         # torus circumference (matches the grid cortex base period); world wraps mod L
TORUS_G = 3           # G x G toroidal cells -> TORUS_G**2 classes (answer 0..8); a wrap-aware question
# 8 compass points, in increasing angle from +x (East) CCW. Index = sector.
COMPASS = ["east", "northeast", "north", "northwest",
           "west", "southwest", "south", "southeast"]


def _gen_moves(T, rng, loop):
    """Out-and-back loop (returns home) or open walk. Used by the 'return' task."""
    heading = [rng.uniform(0, 2 * math.pi) for _ in range(T)]
    speed = [rng.uniform(0.3, 1.0) for _ in range(T)]
    vz = [rng.uniform(-0.5, 0.5) for _ in range(T)]
    if loop:
        half = T // 2
        for i in range(half):                       # second half undoes the first -> return home
            j = half + i
            heading[j] = heading[i] + math.pi
            speed[j] = speed[i]
            vz[j] = -vz[i]
    return torch.tensor(heading), torch.tensor(speed), torch.tensor(vz)


def _walk(T, rng):
    """Plain random walk with modest step size, so endpoints stay within a bounded
    region (keeps the self-supervised place-cell environment from saturating on the
    magnitude tasks). Random heading per step -> distance grows ~sqrt(T), bearing uniform."""
    heading = [rng.uniform(0, 2 * math.pi) for _ in range(T)]
    speed = [rng.uniform(0.2, 0.8) for _ in range(T)]
    vz = [rng.uniform(-0.4, 0.4) for _ in range(T)]
    return torch.tensor(heading), torch.tensor(speed), torch.tensor(vz)


def _final_xyz(h, s, v):
    return (s * h.cos()).sum().item(), (s * h.sin()).sum().item(), v.sum().item()


def answer_for(task, dx, dy, dz):
    """Ground-truth language answer for a path ending at displacement (dx, dy, dz)."""
    dist = math.sqrt(dx * dx + dy * dy + dz * dz)
    if task == "return":
        return "Yes." if dist < RETURN_TOL else "No."
    if task == "distance":
        return str(min(int(round(dist)), DIST_MAX_BUCKET))
    if task == "bearing":
        ang = math.atan2(-dy, -dx)                  # direction from here BACK to the start (xy)
        sector = int(round(ang / (math.pi / 4))) % 8
        return COMPASS[sector]
    if task == "torus":
        # toroidal cell of the wrapped xy position: requires the WRAP (Euclidean reading fails past 1 lap)
        gx = min(int((dx % TORUS_L) / TORUS_L * TORUS_G), TORUS_G - 1)
        gy = min(int((dy % TORUS_L) / TORUS_L * TORUS_G), TORUS_G - 1)
        return str(gx * TORUS_G + gy)
    raise ValueError(task)


def parse_answer(task, text):
    """Extract the predicted answer from generated text (None if unparseable)."""
    t = text.strip().lower()
    if task == "return":
        yi, ni = t.find("yes"), t.find("no")
        if yi == -1 and ni == -1:
            return None
        if yi == -1:
            return "no"
        if ni == -1:
            return "yes"
        return "yes" if yi < ni else "no"
    if task == "distance":
        for ch in t:
            if ch in "012345":
                return ch
        return None
    if task == "bearing":
        for w in ("northeast", "northwest", "southeast", "southwest"):  # compounds first
            if w in t:
                return w
        for w in ("north", "south", "east", "west"):
            if w in t:
                return w
        return None
    if task == "torus":
        for ch in t:
            if ch in "012345678":
                return ch
        return None
    raise ValueError(task)


def answer_index(task, parsed):
    """Integer index of a parsed answer, for exact / within-1 scoring (None if unparseable)."""
    if parsed is None:
        return None
    if task == "return":
        return {"yes": 1, "no": 0}.get(parsed)
    if task == "distance":
        return int(parsed)
    if task == "bearing":
        return COMPASS.index(parsed)
    if task == "torus":
        return int(parsed)
    raise ValueError(task)


def num_classes(task):
    return {"return": 2, "distance": DIST_MAX_BUCKET + 1, "bearing": 8, "torus": TORUS_G * TORUS_G}[task]


def is_circular(task):
    return task == "bearing"


def make_trajectory_qa(n, T=8, seed=0, task="return"):
    """Returns heading,speed,vz (n,T) tensors and a list of language answers for ``task``."""
    rng = random.Random(seed)
    H, S, V, ans = [], [], [], []
    for _ in range(n):
        if task == "return":
            h, s, v = _gen_moves(T, rng, loop=rng.random() < 0.5)
        elif task == "distance":
            # ~30% loops (bucket 0) + random walks (spread the magnitude across buckets)
            h, s, v = _gen_moves(T, rng, loop=True) if rng.random() < 0.3 else _walk(T, rng)
        elif task == "bearing":
            while True:                              # need a well-defined heading-home
                h, s, v = _walk(T, rng)
                dx, dy, _ = _final_xyz(h, s, v)
                if dx * dx + dy * dy > 0.64:         # |xy displacement| > 0.8
                    break
        elif task == "torus":
            h, s, v = _walk(T, rng)                  # random walk; position wraps mod TORUS_L
        else:
            raise ValueError(task)
        dx, dy, dz = _final_xyz(h, s, v)
        H.append(h); S.append(s); V.append(v)
        ans.append(answer_for(task, dx, dy, dz))
    return torch.stack(H), torch.stack(S), torch.stack(V), ans


class TrajectoryQADataset(Dataset):
    def __init__(self, H, S, V, ans):
        self.H, self.S, self.V, self.ans = H, S, V, ans

    def __len__(self):
        return len(self.ans)

    def __getitem__(self, i):
        return {"heading": self.H[i], "speed": self.S[i], "vz": self.V[i], "answer": self.ans[i]}


def collate(batch, tokenizer, question: str = QUESTION, max_length: int = 64):
    """Tokenize question(+answer) with answer-only label masking; stack the moves."""
    H = torch.stack([b["heading"] for b in batch])
    S = torch.stack([b["speed"] for b in batch])
    V = torch.stack([b["vz"] for b in batch])
    prompt = PROMPT.format(q=question)
    fulls = [prompt + " " + b["answer"] for b in batch]
    enc = tokenizer(fulls, max_length=max_length, padding="max_length",
                    truncation=True, return_tensors="pt")
    labels = enc["input_ids"].clone()
    plen = len(tokenizer(prompt)["input_ids"])           # supervise only the answer tokens
    labels[:, :plen] = -100
    labels[enc["attention_mask"] == 0] = -100
    return {"input_ids": enc["input_ids"], "attention_mask": enc["attention_mask"],
            "labels": labels, "heading": H, "speed": S, "vz": V}
