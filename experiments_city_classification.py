"""
Controlled experiment: classify which of 20 cities a (jittered) coordinate
belongs to. This isolates and measures SPATIAL ENCODING quality.

Compares 4 encoders with identical classification heads, training budget,
and data. Reports real test accuracy — no projections.
"""
import sys, time, random
import torch
import torch.nn as nn
import numpy as np
sys.path.insert(0, ".")

from src.data.synthetic import CITIES
from src.models.coord_embedder import CoordinateEmbedder
from src.models.grid_cell_encoder import GridCellEncoder
from src.models.neuro.brain_spatial_cortex import BrainSpatialCortex

torch.manual_seed(0); random.seed(0); np.random.seed(0)
DEVICE = "cpu"
N_CLASSES = len(CITIES)
EMBED = 64


def make_data(n_per_city, jitter=0.15):
    X, y = [], []
    for ci, city in enumerate(CITIES):
        for _ in range(n_per_city):
            lat = city["lat"] + random.gauss(0, jitter)
            lon = city["lon"] + random.gauss(0, jitter)
            X.append([lat, lon]); y.append(ci)
    X = torch.tensor(X, dtype=torch.float32)
    y = torch.tensor(y, dtype=torch.long)
    perm = torch.randperm(len(X))
    return X[perm], y[perm]


# Encoders ---------------------------------------------------------------
class RawMLP(nn.Module):
    """Baseline: raw lat/lon → MLP (no spatial inductive bias)."""
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(2, EMBED), nn.GELU(),
            nn.Linear(EMBED, EMBED), nn.GELU())
    def forward(self, x): return self.net(x)


class FourierEnc(nn.Module):
    def __init__(self):
        super().__init__()
        self.enc = CoordinateEmbedder(embed_dim=EMBED, num_freqs=32)
    def forward(self, x): return self.enc(x)


class GridEnc(nn.Module):
    def __init__(self):
        super().__init__()
        self.enc = GridCellEncoder(embed_dim=EMBED, num_modules=5)
    def forward(self, x): return self.enc(x)


class BrainEnc(nn.Module):
    def __init__(self):
        super().__init__()
        self.enc = BrainSpatialCortex(embed_dim=EMBED, num_tokens=1)
    def forward(self, x): return self.enc(x).squeeze(1)   # (B, EMBED)


class Classifier(nn.Module):
    def __init__(self, encoder):
        super().__init__()
        self.encoder = encoder
        self.head = nn.Linear(EMBED, N_CLASSES)
    def forward(self, x): return self.head(self.encoder(x))


def train_eval(name, encoder, Xtr, ytr, Xte, yte, epochs=60, lr=3e-3):
    model = Classifier(encoder).to(DEVICE)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    lossf = nn.CrossEntropyLoss()
    n_params = sum(p.numel() for p in model.parameters())

    t0 = time.time()
    model.train()
    bs = 256
    for ep in range(epochs):
        perm = torch.randperm(len(Xtr))
        for i in range(0, len(Xtr), bs):
            idx = perm[i:i+bs]
            opt.zero_grad()
            out = model(Xtr[idx])
            loss = lossf(out, ytr[idx])
            loss.backward(); opt.step()
    train_time = time.time() - t0

    model.eval()
    with torch.no_grad():
        pred_te = model(Xte).argmax(-1)
        pred_tr = model(Xtr).argmax(-1)
        test_acc = (pred_te == yte).float().mean().item()
        train_acc = (pred_tr == ytr).float().mean().item()
    return {
        "name": name, "test_acc": test_acc, "train_acc": train_acc,
        "params": n_params, "time_s": train_time,
    }


if __name__ == "__main__":
    print("Generating data...")
    Xtr, ytr = make_data(300)      # 6000 train
    Xte, yte = make_data(80)       # 1600 test
    print(f"Train: {len(Xtr)}  Test: {len(Xte)}  Classes: {N_CLASSES}")
    print(f"Chance accuracy: {1/N_CLASSES:.3f}\n")

    encoders = [
        ("Raw MLP (baseline)", RawMLP()),
        ("Fourier embedding", FourierEnc()),
        ("Grid cell encoder", GridEnc()),
        ("BrainSpatialCortex (full stack)", BrainEnc()),
    ]

    results = []
    for name, enc in encoders:
        print(f"Training: {name} ...")
        r = train_eval(name, enc, Xtr, ytr, Xte, yte)
        results.append(r)
        print(f"  test_acc={r['test_acc']:.3f}  train_acc={r['train_acc']:.3f}  "
              f"params={r['params']:,}  time={r['time_s']:.1f}s\n")

    print("="*72)
    print(f"{'Encoder':<34}{'Test Acc':>10}{'Train Acc':>11}{'Params':>12}")
    print("-"*72)
    for r in results:
        print(f"{r['name']:<34}{r['test_acc']:>9.1%}{r['train_acc']:>10.1%}{r['params']:>12,}")
    print("="*72)

    # Save for README
    import json
    with open("experiment_results.json", "w") as f:
        json.dump(results, f, indent=2)
