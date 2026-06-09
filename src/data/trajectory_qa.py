"""
src/data/trajectory_qa.py

Trajectory question-answering data for Milestone 2.

Each example is a PATH (a sequence of moves) + a natural-language yes/no question
("Are you back where you started?") + answer. The moves are NEVER written in the text —
they reach the model only through the trajectory cortex — so the LLM must use the
cortex's path integration to answer.

Balanced by construction: ~half the paths are loops (out-and-back, return near start ->
"Yes."), half are open walks (end far -> "No."). The label is the ground truth from the
actual final displacement, so it's always correct even if a random walk happens to close.
"""
import math
import random

import torch
from torch.utils.data import Dataset

QUESTION = "Are you back where you started?"
PROMPT = "[NAVIGATION] You walked a path through space.\n[QUESTION] {q}\n[ANSWER]"
RETURN_TOL = 0.5   # final displacement below this counts as "back at start"


def _gen_moves(T, rng, loop):
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


def make_trajectory_qa(n, T=8, seed=0):
    """Returns heading,speed,vz (n,T) tensors and a list of "Yes."/"No." answers."""
    rng = random.Random(seed)
    H, S, V, ans = [], [], [], []
    for _ in range(n):
        h, s, v = _gen_moves(T, rng, loop=rng.random() < 0.5)
        dx = (s * h.cos()).sum(); dy = (s * h.sin()).sum(); dz = v.sum()
        dist = torch.sqrt(dx * dx + dy * dy + dz * dz).item()
        H.append(h); S.append(s); V.append(v)
        ans.append("Yes." if dist < RETURN_TOL else "No.")
    return torch.stack(H), torch.stack(S), torch.stack(V), ans


class TrajectoryQADataset(Dataset):
    def __init__(self, H, S, V, ans):
        self.H, self.S, self.V, self.ans = H, S, V, ans

    def __len__(self):
        return len(self.ans)

    def __getitem__(self, i):
        return {"heading": self.H[i], "speed": self.S[i], "vz": self.V[i], "answer": self.ans[i]}


def collate(batch, tokenizer, max_length: int = 64):
    """Tokenize question(+answer) with answer-only label masking; stack the moves."""
    H = torch.stack([b["heading"] for b in batch])
    S = torch.stack([b["speed"] for b in batch])
    V = torch.stack([b["vz"] for b in batch])
    prompt = PROMPT.format(q=QUESTION)
    fulls = [prompt + " " + b["answer"] for b in batch]
    enc = tokenizer(fulls, max_length=max_length, padding="max_length",
                    truncation=True, return_tensors="pt")
    labels = enc["input_ids"].clone()
    plen = len(tokenizer(prompt)["input_ids"])           # supervise only the answer tokens
    labels[:, :plen] = -100
    labels[enc["attention_mask"] == 0] = -100
    return {"input_ids": enc["input_ids"], "attention_mask": enc["attention_mask"],
            "labels": labels, "heading": H, "speed": S, "vz": V}
