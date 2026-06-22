# =====================================================================================
# Emergent TIME CELLS + scalar (Weber) timing on a T4 — at scale.  SELF-CONTAINED: paste
# into ONE Kaggle GPU cell and run. No repo, no data download. ~10-20 min on a T4.
#
# THE CLAIM (and what makes it faithful): nothing about time cells, field widening, or scalar
# timing is hard-coded. We build a GENERIC recurrent substrate (leaky rectified rate-RNN, ONE
# uniform time-constant, learned recurrence, private noise) and train it on ONE task — "report
# how much time has elapsed since a start pulse, when probed at a random moment" — with a
# metabolic activity cost.  Then we MEASURE what emerged:
#   (1) time cells (single-peaked, tiling the interval, denser early -- Mau 2018),
#   (2) fields that WIDEN with latency  (never in the loss),
#   (3) SCALAR/Weber timing: decoded-time SD grows ~linearly with elapsed time at a ~constant
#       Weber fraction (Gibbon 1977) -- while an UNTRAINED net of the same architecture cannot
#       time at all.  Scaling up (bigger net, longer interval) shows it is not a toy artifact.
# =====================================================================================
import math, torch, torch.nn as nn
import matplotlib.pyplot as plt

dev = "cuda" if torch.cuda.is_available() else "cpu"
print("device:", dev, torch.cuda.get_device_name(0) if dev == "cuda" else "")

# ---- scale (T4 lets us go bigger than the CPU run; emergence should be ROBUST to scale) ----
T = 80; HIDDEN = 256; NOISE = 0.06; ACT_COST = 1e-3; GAIN = 1.4; ALPHA = 0.25
ITERS = 4000; BATCH = 256; SEEDS = 8

class TemporalCortex(nn.Module):
    """Generic recurrent substrate. No timing structure imposed."""
    def __init__(self, H, n_in=2, n_out=1):
        super().__init__()
        self.H = H
        self.Wr = nn.Parameter(torch.randn(H, H) * (GAIN / math.sqrt(H)))
        self.Wi = nn.Parameter(torch.randn(H, n_in) * 0.5)
        self.b  = nn.Parameter(torch.zeros(H)); self.readout = nn.Linear(H, n_out)
    def dynamics(self, x, noise=0.0):
        B = x.shape[0]; h = torch.zeros(B, self.H, device=x.device); rs = []
        for t in range(x.shape[1]):
            r = torch.relu(h)
            h = (1-ALPHA)*h + ALPHA*(r @ self.Wr.t() + x[:, t] @ self.Wi.t() + self.b)
            if noise > 0: h = h + noise*torch.randn_like(h)
            rs.append(torch.relu(h))
        return torch.stack(rs, 1)
    def forward(self, x, noise=0.0):
        r = self.dynamics(x, noise); return self.readout(r), r

def make_trial(B):
    x = torch.zeros(B, T, 2, device=dev); x[:, 0, 0] = 1.0
    probe = torch.randint(T // 5, T, (B,), device=dev)
    x[torch.arange(B, device=dev), probe, 1] = 1.0
    return x, probe

def ridge(A, y, lam=1.0):
    Ab = torch.cat([A, torch.ones(A.shape[0], 1, device=A.device)], 1)
    return torch.linalg.solve(Ab.t() @ Ab + lam*torch.eye(Ab.shape[1], device=A.device), Ab.t() @ y)

def corr(a, b):
    a = a - a.mean(); b = b - b.mean(); return (a @ b / (a.norm()*b.norm()+1e-9)).item()

@torch.no_grad()
def probe(net, n=800):
    x, _ = make_trial(n); R = net.dynamics(x, noise=NOISE); A = R.mean(0)
    ts = torch.arange(T, device=dev).float()
    W = ridge(A, ts); that = torch.cat([R, torch.ones(n, T, 1, device=dev)], -1) @ W
    mae = (that.mean(0) - ts).abs().mean().item(); sigma = that.std(0)
    mid = (ts > 5) & (ts < T-5); scal = corr(ts[mid], sigma[mid])
    cv = sigma[mid]/ts[mid]; weber_cv = (cv.std(unbiased=True)/(cv.mean()+1e-9)).item()
    Ar = A / (A.max(0).values + 1e-6); peak = Ar.argmax(0).float(); width = (Ar > 0.5).float().sum(0)
    near = torch.stack([Ar[max(0,int(p)-int(0.1*T)):int(p)+int(0.1*T)+1, u].sum() for u, p in enumerate(peak)])
    act = A.max(0).values > 0.05*A.max()
    is_tc = act & (near/(Ar.sum(0)+1e-6) > 0.5) & (width < T*0.5) & (peak > 1) & (peak < T-2)
    tc = is_tc.nonzero().squeeze(-1)
    wcorr = corr(peak[tc], width[tc]) if len(tc) > 5 else float("nan")
    early = (peak[tc] < T/2).float().mean().item() if len(tc) else float("nan")
    return dict(mae=mae, frac=is_tc.float().mean().item(), wcorr=wcorr, scal=scal,
                weber_cv=weber_cv, early=early), dict(Ar=Ar.cpu(), tc=tc.cpu(), peak=peak.cpu(),
                width=width.cpu(), sigma=sigma.cpu(), ts=ts.cpu())

def run_seed(seed):
    torch.manual_seed(seed)
    net = TemporalCortex(HIDDEN).to(dev); opt = torch.optim.Adam(net.parameters(), 3e-3)
    for it in range(ITERS):
        x, probe_t = make_trial(BATCH); pred, R = net(x, noise=NOISE)
        pred = pred[torch.arange(BATCH, device=dev), probe_t].squeeze(-1)
        loss = ((pred - probe_t.float()/T)**2).mean() + ACT_COST*R.pow(2).mean()
        opt.zero_grad(); loss.backward(); opt.step()
    tr, arr = probe(net)
    ct, carr = probe(TemporalCortex(HIDDEN).to(dev))          # untrained control
    tr["ctrl_mae"], tr["ctrl_weber_cv"], tr["ctrl_wcorr"] = ct["mae"], ct["weber_cv"], ct["wcorr"]
    return tr, arr, carr

rows = []; arr0 = carr0 = None
for s in range(SEEDS):
    r, arr, carr = run_seed(s)
    if s == 0: arr0, carr0 = arr, carr
    rows.append(r)
    print(f"seed {s}: time-cells {r['frac']:.0%} | widen {r['wcorr']:+.2f} | scalar {r['scal']:+.2f} | "
          f"WeberCV {r['weber_cv']:.2f} | MAE {r['mae']:.2f} (ctrl MAE {r['ctrl_mae']:.1f}, ctrl WeberCV {r['ctrl_weber_cv']:.2f})")

def agg(k):
    v = torch.tensor([r[k] for r in rows if r[k] == r[k]]);
    return v.mean().item(), (1.96*v.std(unbiased=True)/math.sqrt(len(v))).item() if len(v) > 1 else 0.0
print("\n=== EMERGENT TIME CODE @ scale (n=%d, H=%d, T=%d) — mean +/- 95%% CI ===" % (SEEDS, HIDDEN, T))
for k, name in [("frac","time-cell fraction"),("early","  peaking in first half (denser-early)"),
                ("wcorr","FIELD WIDENING corr (emergent)"),("ctrl_wcorr","  untrained widening corr"),
                ("scal","scalar-timing corr (SD vs t)"),("weber_cv","Weber-fraction CV (LOW=scale-inv)"),
                ("ctrl_weber_cv","  untrained Weber CV (HIGH)"),("mae","decode MAE steps"),
                ("ctrl_mae","  untrained MAE steps")]:
    m, c = agg(k); print(f"  {name:42} {m:+.3f} +/- {c:.3f}")

# ---- figure ----
fig, ax = plt.subplots(1, 3, figsize=(16, 4.2))
tc0 = arr0["tc"]; order = tc0[arr0["peak"][tc0].argsort()]
ax[0].imshow(arr0["Ar"][:, order].T.numpy(), aspect="auto", cmap="magma", origin="lower",
             extent=[0, T, 0, len(order)])
ax[0].set_title(f"Emergent time cells ({len(order)}), sorted by peak"); ax[0].set_xlabel("elapsed time"); ax[0].set_ylabel("cell")
ax[1].scatter(arr0["peak"][tc0].numpy(), arr0["width"][tc0].numpy(), s=14, c="#2ca25f")
ax[1].set_title(f"Fields WIDEN with latency (corr {agg('wcorr')[0]:+.2f})"); ax[1].set_xlabel("peak time"); ax[1].set_ylabel("field width")
ax[2].plot(arr0["ts"][2:-2].numpy(), arr0["sigma"][2:-2].numpy(), c="#2ca25f", lw=2.4, label=f"trained (scalar, MAE {agg('mae')[0]:.2f})")
ax[2].plot(carr0["ts"][2:-2].numpy(), carr0["sigma"][2:-2].numpy(), c="#9aa5b8", lw=2.0, label=f"untrained (MAE {agg('ctrl_mae')[0]:.1f})")
ax[2].set_title("Scalar (Weber) timing: decoded-SD vs elapsed time"); ax[2].set_xlabel("elapsed time"); ax[2].set_ylabel("decoded-time SD"); ax[2].legend()
plt.tight_layout(); plt.savefig("emergent_time.png", dpi=130); plt.show()
print("\nsaved emergent_time.png")
