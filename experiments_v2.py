"""
Experiment v2 — two harder, discriminative tasks:

TASK A: Coordinate regression. Given a coordinate corrupted by large noise,
        reconstruct the true location. Measured by Haversine error (km).
        This rewards encoders that preserve fine metric structure.

TASK B: Fine-grained classification into a 1°×1° grid of cells over a region
        (many nearby classes). Rewards spatial resolution, not just
        "which continent".
"""
import sys, time, random, json
import torch
import torch.nn as nn
import numpy as np
sys.path.insert(0, ".")

from src.models.coord_embedder import CoordinateEmbedder
from src.models.grid_cell_encoder import GridCellEncoder
from src.models.neuro.brain_spatial_cortex import BrainSpatialCortex
from src.eval.metrics import haversine_km

torch.manual_seed(1); random.seed(1); np.random.seed(1)
EMBED = 64


# ── Encoders ────────────────────────────────────────────────────────────
def raw_mlp():
    return nn.Sequential(nn.Linear(2, EMBED), nn.GELU(),
                         nn.Linear(EMBED, EMBED), nn.GELU())

class Wrap(nn.Module):
    def __init__(self, enc, squeeze=False):
        super().__init__(); self.enc = enc; self.squeeze = squeeze
    def forward(self, x):
        o = self.enc(x)
        return o.squeeze(1) if self.squeeze else o

def make_encoders():
    return [
        ("Raw MLP (baseline)", raw_mlp()),
        ("Fourier", CoordinateEmbedder(embed_dim=EMBED, num_freqs=32)),
        ("Grid cells (FIXED)", GridCellEncoder(embed_dim=EMBED, num_modules=6,
                                               base_scale=1.0, scale_factor=2.0)),
        ("BrainSpatialCortex", Wrap(BrainSpatialCortex(embed_dim=EMBED, num_tokens=1),
                                    squeeze=True)),
    ]


# ── TASK A: coordinate regression ──────────────────────────────────────
def task_a():
    # Sample real-world coordinates over a region (Europe-ish), add noise
    def gen(n):
        lat = torch.empty(n).uniform_(35, 60)
        lon = torch.empty(n).uniform_(-10, 30)
        true = torch.stack([lat, lon], dim=1)
        noisy = true + torch.randn_like(true) * 2.0      # 2° noise
        return noisy, true
    Xtr, Ytr = gen(8000)
    Xte, Yte = gen(2000)

    rows = []
    for name, enc in make_encoders():
        model = nn.Sequential(enc, nn.Linear(EMBED, 2))
        opt = torch.optim.Adam(model.parameters(), lr=3e-3)
        lossf = nn.MSELoss()
        model.train()
        for ep in range(80):
            perm = torch.randperm(len(Xtr))
            for i in range(0, len(Xtr), 256):
                idx = perm[i:i+256]
                opt.zero_grad()
                loss = lossf(model(Xtr[idx]), Ytr[idx])
                loss.backward(); opt.step()
        model.eval()
        with torch.no_grad():
            pred = model(Xte)
        errs = [haversine_km(pred[i,0].item(), pred[i,1].item(),
                             Yte[i,0].item(), Yte[i,1].item()) for i in range(len(Xte))]
        rows.append((name, float(np.mean(errs)), float(np.median(errs))))
    return rows


# ── TASK B: fine-grained 1° grid classification ────────────────────────
def task_b():
    # Region 40-50 lat, 0-10 lon → 10x10 = 100 cells
    def cell_id(lat, lon):
        return int((lat - 40)) * 10 + int((lon - 0))
    def gen(n):
        X, y = [], []
        for _ in range(n):
            lat = random.uniform(40, 49.999); lon = random.uniform(0, 9.999)
            X.append([lat, lon]); y.append(cell_id(lat, lon))
        return torch.tensor(X), torch.tensor(y)
    Xtr, ytr = gen(20000); Xte, yte = gen(4000)
    N_CLS = 100

    rows = []
    for name, enc in make_encoders():
        model = nn.Sequential(enc, nn.Linear(EMBED, N_CLS))
        opt = torch.optim.Adam(model.parameters(), lr=3e-3)
        lossf = nn.CrossEntropyLoss()
        model.train()
        for ep in range(60):
            perm = torch.randperm(len(Xtr))
            for i in range(0, len(Xtr), 512):
                idx = perm[i:i+512]
                opt.zero_grad()
                loss = lossf(model(Xtr[idx]), ytr[idx])
                loss.backward(); opt.step()
        model.eval()
        with torch.no_grad():
            acc = (model(Xte).argmax(-1) == yte).float().mean().item()
        rows.append((name, acc))
    return rows


if __name__ == "__main__":
    print("="*70)
    print("TASK A — Coordinate regression (Haversine error, km, LOWER better)")
    print("="*70)
    a = task_a()
    print(f"{'Encoder':<26}{'Mean km':>12}{'Median km':>12}")
    for name, mean, med in a:
        print(f"{name:<26}{mean:>12.1f}{med:>12.1f}")

    print()
    print("="*70)
    print("TASK B — Fine 1° grid classification, 100 classes (HIGHER better)")
    print(f"Chance = {1/100:.2f}")
    print("="*70)
    b = task_b()
    print(f"{'Encoder':<26}{'Test Acc':>12}")
    for name, acc in b:
        print(f"{name:<26}{acc:>11.1%}")

    json.dump({"task_a": a, "task_b": b}, open("experiment_results_v2.json","w"), indent=2)
