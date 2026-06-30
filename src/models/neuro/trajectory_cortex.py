"""
src/models/neuro/trajectory_cortex.py

Recurrent, ablatable spatial cortex for 4D navigation (x, y, z over time t).

Three tasks:
  - "pathint":   integrate moves -> FINAL (x,y,z). Commutative sum: only needs the
                 velocity encoder + integration.
  - "recall":    "where were you at step k?" with FULL-sequence attention. Order-
                 dependent, so the recurrent integrator (running position) is essential.
  - "memrecall": same query, but through a fixed-size MEMORY BOTTLENECK — the trajectory
                 is multiplexed into ONE vector and the answer is read back from that.
                 This needs an ORDER-preserving memory (theta-gamma, ~7 slots); a mean-
                 pool bottleneck collapses order and fails. Makes the whole stack
                 (velocity encoder + integrator + theta-gamma memory) load-bearing.

With gated=True each OPTIONAL module gets a learned gate (+L1) so the network turns
down modules it doesn't need (task-dependent complexity).
"""
import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from .spatial_cells import (ConjunctiveSpatialCells, BoundaryVectorCells, EgocentricObjectVectorCells,
                            LocalOrder3DGrid)
from .oscillations import ThetaGammaCoupling, ThetaGammaMemory
from .microcircuits import CorticalColumn, LateralInhibition


TRAJ_DEFAULT_CONFIG = {
    "conjunctive": True,        # head-direction x speed -> per-step velocity code (structural)
    "grid_attractor": True,     # recurrent path integrator over the move sequence (structural)
    "theta_gamma": True,        # theta-gamma ordered sequence memory
    "cortical_column": True,    # canonical microcircuit (gateable add-on)
    "lateral_inhibition": True, # surround inhibition (gateable add-on)
}
GATEABLE = ("theta_gamma", "cortical_column", "lateral_inhibition")


class _AttractorIntegrator(nn.Module):
    """Recurrent continuous-attractor path integrator. With return_sequence=True it
    returns the running position at every step (needed for recall/memrecall).

    ``length_norm`` controls how the accumulated sheet ``u`` is read out:
      - True  (default): ``readout(u / T)`` — the original behaviour. Calibrates the
        readout to the training length; the generalization stress-test
        (``src/eval/generalize_trajectory.py``) showed this LOCKS the model to that
        length (predictions scale by train_T/test_T on unseen lengths).
      - False (scale-free): ``readout(u)`` — accumulation is read out directly. Paired
        with MIXED-length training this is what extrapolates to unseen path lengths.
    """

    def __init__(self, embed_dim: int, grid_size: int = 16, settle: int = 2,
                 length_norm: bool = True, topology: str = "square"):
        super().__init__()
        self.N = grid_size * grid_size
        self.settle = settle
        self.length_norm = length_norm
        self.vel_to_sheet = nn.Linear(embed_dim, self.N)
        g = grid_size
        ii, jj = torch.meshgrid(torch.arange(g), torch.arange(g), indexing="ij")
        ii = ii.reshape(-1).float(); jj = jj.reshape(-1).float()
        if topology == "hex":
            # TWISTED (rhombic 60°) torus -> hexagonal grid fields (Guanella et al. 2007).
            # Shear the sheet to a 60° lattice and wrap on its two lattice vectors; the bump's
            # periodic images then tile hexagonally instead of in a square.
            cells = torch.stack([ii + 0.5 * jj, jj * (math.sqrt(3) / 2.0)], dim=-1)
            a1 = torch.tensor([float(g), 0.0]); a2 = torch.tensor([g * 0.5, g * math.sqrt(3) / 2.0])
            d = cells.unsqueeze(0) - cells.unsqueeze(1)
            dist_sq = None
            for m in (-1, 0, 1):
                for n in (-1, 0, 1):                    # min-image over the rhombic lattice
                    ds = ((d - (m * a1 + n * a2)) ** 2).sum(-1)
                    dist_sq = ds if dist_sq is None else torch.minimum(dist_sq, ds)
        else:
            # SQUARE torus (default; original behaviour) — independent per-axis wrap.
            cells = torch.stack([ii, jj], dim=-1)
            d = cells.unsqueeze(0) - cells.unsqueeze(1)
            d = torch.minimum(d.abs(), g - d.abs())
            dist_sq = (d ** 2).sum(-1)
        self.register_buffer("W", torch.exp(-dist_sq / 8.0) - 0.6 * torch.exp(-dist_sq / 72.0))
        self.readout = nn.Linear(self.N, embed_dim)

    def forward(self, vel_seq: torch.Tensor, return_sequence: bool = False) -> torch.Tensor:
        B, T, _ = vel_seq.shape
        u = torch.zeros(B, self.N, device=vel_seq.device, dtype=vel_seq.dtype)
        outs = []
        for t in range(T):
            u = u + self.vel_to_sheet(vel_seq[:, t])
            for _ in range(self.settle):
                u = u + 0.1 * F.linear(torch.tanh(u), self.W)
            if return_sequence:
                outs.append(self.readout(u / T if self.length_norm else u))
        if return_sequence:
            return torch.stack(outs, dim=1)
        return self.readout(u / T if self.length_norm else u)


class _HexGridModules(nn.Module):
    """Biologically-CONSTRAINED velocity-driven grid modules (Burak & Fiete 2009; Guanella 2007).

    Self-motion velocity drives a PHASE that is integrated and wrapped on a hexagonal (twisted)
    torus, so each module's cells fire on a HEXAGONAL lattice in real space — grid cells by
    construction. Several modules at geometric scale ratios (the real entorhinal organisation;
    Stensola 2012) make the population code unambiguous. Only the velocity GAINS are fixed
    (faithful); the readout is LEARNED (grid code -> downstream), mirroring entorhinal->hippocampal
    flow. The phase depends on the FINAL position, not the path length -> path integration is
    length-invariant by construction.
    """

    def __init__(self, embed_dim: int, n_modules: int = 4, side: int = 8,
                 base_spacing: float = 1.0, ratio: float = 1.42, sigma: float = 1.0,
                 z_cells: int = 16, z_range: float = 3.0, z_sigma: float = 0.6,
                 noise_std: float = 0.0, boundary_anchor: bool = False, learned_loc: bool = False,
                 object_anchor: bool = False, obj_reanchor_w: float = 0.7,
                 grid_3d: bool = False, grid3d_seed: int = 0):
        super().__init__()
        self.K, self.side, self.M, self.sigma = n_modules, side, side * side, sigma
        ii, jj = torch.meshgrid(torch.arange(side), torch.arange(side), indexing="ij")
        ii = ii.reshape(-1).float(); jj = jj.reshape(-1).float()
        self.register_buffer("cell_pos", torch.stack([ii + 0.5 * jj, jj * (math.sqrt(3) / 2)], -1))  # (M,2)
        a1 = torch.tensor([float(side), 0.0]); a2 = torch.tensor([side * 0.5, side * math.sqrt(3) / 2])
        self.register_buffer("shifts", torch.stack([m * a1 + n * a2
                                                    for m in (-1, 0, 1) for n in (-1, 0, 1)]))     # (9,2)
        spacings = base_spacing * (ratio ** torch.arange(n_modules).float())
        self.register_buffer("gains", side / spacings)                                            # (K,) FIXED
        # VERTICAL / 3D code. Default: a simple 1D place code over integrated z (height not grid-coded).
        # grid_3d=True swaps that stub for a biologically-grounded 3D grid code with LOCAL order but NO global
        # lattice (the bat MEC regime; Ginosar et al. 2021) — it path-integrates full 3D self-motion and
        # localizes in 3D, rather than treating height as a 1-D place stub.
        self.grid_3d = grid_3d
        if grid_3d:
            self.grid3d = LocalOrder3DGrid(embed_dim=embed_dim, box=3.0, seed=grid3d_seed)
            self.readout = nn.Linear(self.K * self.M + self.grid3d.n_cells, embed_dim)
        else:
            self.z_sigma = z_sigma
            self.register_buffer("z_centers", torch.linspace(-z_range, z_range, z_cells))
            self.readout = nn.Linear(self.K * self.M + z_cells, embed_dim)
        # integration noise (real path integration is noisy -> drift) and BOUNDARY anchoring:
        # boundary cells provide an allothetic position fix that gated-resets the grid phase
        # (Hardcastle, Ganguli & Giocomo 2015 — boundaries correct accumulated grid error).
        self.noise_std = noise_std
        self.boundary_anchor = boundary_anchor
        self.learned_loc = learned_loc
        self.arena_R = 3.0
        self.anchor_scale = 0.8                     # boundary correction acts within ~1 unit of a wall
        if boundary_anchor:
            self.bvc = BoundaryVectorCells(num_cells=24, embed_dim=32, max_distance=3.0)
            self.bvc_gate = nn.Linear(32, 2)        # PER-AXIS learned trust (a wall fixes only the
            if learned_loc:                         # perpendicular axis)
                # LEARN to localize from boundary cells (no hard-coded arena geometry); trained
                # by an auxiliary experiential signal in src/eval/boundary_anchoring.py
                self.bvc_to_pos = nn.Sequential(nn.Linear(32, 32), nn.ReLU(), nn.Linear(32, 2))
            else:
                nn.init.constant_(self.bvc_gate.bias, 2.0)   # geometric fix: start with anchoring ON
        # OBJECT / LANDMARK / CENTRE reanchoring (Nature Neurosci 2025: grid cells reanchor to a
        # task-relevant object by TRANSLATING the pattern; Nat Commun 2025: ego & allo codes coexist).
        # Egocentric object-vector cells supply an allothetic fix ANYWHERE the anchor is seen (not only at
        # walls), via the SAME egocentric->allocentric transform as boundaries (see _ego_to_allo /
        # _apply_phase_fix). The object's world position can move between trials -> the grid reanchors to it.
        self.object_anchor = object_anchor
        self.obj_reanchor_w = obj_reanchor_w
        if object_anchor:
            self.ovc = EgocentricObjectVectorCells(num_cells=24, embed_dim=32, max_distance=6.0)
            self.ovc_gate = nn.Linear(32, 1)        # learned scalar trust (down-weight a far/ambiguous object)
            nn.init.constant_(self.ovc_gate.bias, 2.0)   # start with object anchoring ON

    def _grid_code(self, phi):                       # phi (K,B,2) -> (B, K*M)
        K, B, _ = phi.shape
        d0 = self.cell_pos.view(1, 1, self.M, 2) - phi.view(K, B, 1, 2)        # (K,B,M,2)
        best = None
        for s in self.shifts:                        # min-image distance on the hex lattice
            ds = ((d0 - s) ** 2).sum(-1)             # (K,B,M)
            best = ds if best is None else torch.minimum(best, ds)
        bump = torch.exp(-best / (2 * self.sigma ** 2))                        # (K,B,M)
        return bump.permute(1, 0, 2).reshape(B, self.K * self.M)

    def grid_code_at(self, pos):
        """Grid-cell population code for absolute 2D position(s) ``pos`` (B,2), via the module's FIXED
        gains (exact, noiseless path-integration from the origin: phi = gains * pos). Returns (B, K*M).
        Lets a closed-loop agent read this cortex's grid code at any position without re-running the
        recurrent integrator step by step (used by src/eval/agent_grid_cortex.py)."""
        phi = self.gains.view(self.K, 1, 1) * pos.unsqueeze(0)                 # (K,B,2)
        return self._grid_code(phi)

    def _z_code(self, z):                            # z (B,) -> (B, z_cells)
        return torch.exp(-((z.unsqueeze(1) - self.z_centers.unsqueeze(0)) ** 2) / (2 * self.z_sigma ** 2))

    @staticmethod
    def _ego_to_allo(ego_dist, ego_bearing, heading):
        """General egocentric->allocentric transform. An anchor seen at egocentric (distance, bearing)
        relative to the agent's `heading` -> its WORLD-frame offset from the agent: R(heading)·(d·cos b, d·sin b).
        This is the single bridge used for every anchor (boundary, object, centre): the agent's implied world
        position is then `anchor_world_pos - ego_to_allo(...)`. ego_dist/ego_bearing/heading are (B,)."""
        ex = ego_dist * torch.cos(ego_bearing); ey = ego_dist * torch.sin(ego_bearing)   # agent frame
        c, s = torch.cos(heading), torch.sin(heading)
        return torch.stack([c * ex - s * ey, s * ex + c * ey], dim=-1)                    # (B,2) world offset

    def _apply_phase_fix(self, phi, p_hat, w):
        """Gated reanchoring of the grid phase to an allothetic position estimate `p_hat` (B,2):
        phi <- (1-w)·phi + w·gains·p_hat. `w` broadcasts over (K,B,2). Shared by boundary & object anchoring."""
        target = self.gains.view(self.K, 1, 1) * p_hat.unsqueeze(0)                       # (K,B,2)
        return (1 - w) * phi + w * target

    def forward(self, v3d, boundary_obs=None, object_obs=None, heading=None,
                return_sequence: bool = False, return_cells: bool = False, return_grid_seq: bool = False):
        """v3d (B,T,3) = (vx, vy, vz). boundary_obs (B,T,2) = (dist, bearing) to nearest wall.
        object_obs (B,T,4|5) = (ego_dist, ego_bearing, anchor_x, anchor_y[, reliability]) to a task-relevant
        OBJECT / LANDMARK / CENTRE — the grid phase reanchors to it via object-vector cells (anywhere it is
        seen, not only at walls). heading (B,T) is the agent's heading for the ego->allo transform; if None it
        is taken from the instantaneous self-motion direction. Returns the integrated rep; return_cells also
        yields the final GRID-cell population (B, K*M); return_grid_seq yields the per-step grid code (B,T,K*M)."""
        B, T, _ = v3d.shape
        phi = torch.zeros(self.K, B, 2, device=v3d.device, dtype=v3d.dtype)
        z = torch.zeros(B, device=v3d.device, dtype=v3d.dtype)
        p_xy = torch.zeros(B, 2, device=v3d.device, dtype=v3d.dtype)                 # integrated xy (for the 3D code)
        outs, grids, last_grid, last_full = [], [], None, None
        for t in range(T):
            vxy = v3d[:, t, :2]
            head_t = heading[:, t] if heading is not None else torch.atan2(vxy[:, 1], vxy[:, 0])
            if self.noise_std > 0:                                                   # noisy path integration -> drift
                vxy = vxy + torch.randn_like(vxy) * self.noise_std
            phi = phi + self.gains.view(self.K, 1, 1) * vxy.unsqueeze(0)             # integrate xy velocity
            z = z + v3d[:, t, 2]                                                     # integrate height
            if self.grid_3d:
                p_xy = p_xy + vxy                                                    # integrate xy position
            if self.boundary_anchor and boundary_obs is not None:                    # phase reset at walls
                dist = boundary_obs[:, t, 0]; bearing = boundary_obs[:, t, 1]
                emb = self.bvc(dist, bearing)                                        # boundary-vector cells
                g = torch.sigmoid(self.bvc_gate(emb))                                # (B,2) learned per-axis trust
                if self.learned_loc:
                    p_hat = self.bvc_to_pos(emb)                                     # (B,2) LEARNED localization
                    w = g.unsqueeze(0)                                               # gate already encodes proximity/axis
                else:
                    prox = torch.exp(-dist / self.anchor_scale)                      # strong only near a wall
                    # boundary-implied coordinate: you're at (R - dist) along the wall's normal
                    p_hat = torch.stack([bearing.cos() * (self.arena_R - dist),
                                         bearing.sin() * (self.arena_R - dist)], dim=-1)
                    w = (torch.stack([bearing.cos().abs(), bearing.sin().abs()], -1)  # constrained axis only
                         * prox.unsqueeze(1) * g).unsqueeze(0)                        # (1,B,2)
                phi = self._apply_phase_fix(phi, p_hat, w)                            # correct perpendicular drift
            if self.object_anchor and object_obs is not None:                        # reanchor to an object/landmark
                od = object_obs[:, t, 0]; ob = object_obs[:, t, 1]                    # egocentric distance, bearing
                apos = object_obs[:, t, 2:4]                                          # anchor's WORLD position
                rel = object_obs[:, t, 4] if object_obs.shape[-1] > 4 else torch.ones_like(od)
                emb = self.ovc(od, ob)                                                # egocentric object-vector cells
                g = torch.sigmoid(self.ovc_gate(emb)).squeeze(-1)                     # (B,) learned trust in the object
                p_hat = apos - self._ego_to_allo(od, ob, head_t)                      # anchor - R(heading)·ego
                w = (self.obj_reanchor_w * rel * g).clamp(0, 1).view(1, -1, 1)        # (1,B,1) reliability-gated
                phi = self._apply_phase_fix(phi, p_hat, w)                            # TRANSLATE the grid to the object frame
            last_grid = self._grid_code(phi)
            if self.grid_3d:
                p3d = torch.cat([p_xy, z.unsqueeze(1)], dim=-1)                       # integrated 3D position
                last_full = torch.cat([last_grid, self.grid3d.code_at(p3d)], dim=-1)  # 3D local-order grid code
            else:
                last_full = torch.cat([last_grid, self._z_code(z)], dim=-1)
            if return_grid_seq:
                grids.append(last_grid)
            if return_sequence:
                outs.append(self.readout(last_full))
        if return_grid_seq:
            grid_seq = torch.stack(grids, dim=1)                                      # (B,T,K*M)
            return (torch.stack(outs, dim=1), grid_seq) if return_sequence else grid_seq
        if return_sequence:
            seq = torch.stack(outs, dim=1)
            return (seq, last_grid) if return_cells else seq
        out = self.readout(last_full)
        return (out, last_grid) if return_cells else out


class TrajectoryCortex(nn.Module):
    def __init__(self, embed_dim: int = 64, config: dict | None = None,
                 aux_heads: bool = False, dims: int = 3,
                 task: str = "pathint", gated: bool = False, max_T: int = 64,
                 mem_slots: int = 8, length_norm: bool = True, out_norm: bool = True,
                 topology: str = "square", constrained_velocity: bool = False):
        super().__init__()
        self.embed_dim = embed_dim
        self.dims = dims
        self.task = task
        self.gated = gated
        self.cfg = {**TRAJ_DEFAULT_CONFIG, **(config or {})}
        self.aux_heads = aux_heads

        if self.cfg["conjunctive"]:
            self.conjunctive = ConjunctiveSpatialCells(embed_dim=embed_dim)
            self.vert = nn.Linear(1, embed_dim)
        if self.cfg["grid_attractor"]:
            if constrained_velocity:
                # velocity-driven hexagonal grid modules (fixed gains, learned readout).
                # 6 modules over a wide scale range so the residue code stays UNAMBIGUOUS over the
                # task's displacement range (too few/narrow modules alias on long paths -> bearing errors).
                self.integrator = _HexGridModules(embed_dim, n_modules=6, base_spacing=1.6)
            else:
                self.integrator = _AttractorIntegrator(embed_dim, length_norm=length_norm,
                                                       topology=topology)
        else:
            self.pool_proj = nn.Linear(embed_dim, embed_dim)
        self.constrained = constrained_velocity

        # theta-gamma plays two roles depending on task:
        #  - pathint/recall: optional additive sequence-summary (ThetaGammaCoupling)
        #  - memrecall:      the fixed-size ORDERED memory bottleneck (ThetaGammaMemory)
        self.tg_addon = (ThetaGammaCoupling(embed_dim, num_slots=7)
                         if self.cfg["theta_gamma"] and task != "memrecall" else None)
        self.tg_mem = (ThetaGammaMemory(embed_dim, num_slots=mem_slots)
                       if self.cfg["theta_gamma"] and task == "memrecall" else None)

        self.column = CorticalColumn(dim=embed_dim) if self.cfg["cortical_column"] else None
        self.lateral = LateralInhibition(dim=embed_dim) if self.cfg["lateral_inhibition"] else None

        if task == "recall":
            self.step_key = nn.Embedding(max_T, embed_dim)   # tied key/query positional retrieval
        if task == "memrecall":
            self.q_embed = nn.Embedding(max_T, embed_dim)              # query for the mean-pool fallback
            self.bottleneck_read = nn.Linear(embed_dim * 2, embed_dim)  # read item k from order-less mem

        # out_norm=False bypasses the final LayerNorm. The LayerNorm stabilises training
        # but normalises away the rep's MAGNITUDE — fine for direction/binary tasks, but it
        # discards the scale a magnitude question ("how far?") needs. Bypassing lets the
        # scale-free integrator's growing activity survive into the readout.
        self.out_norm = nn.LayerNorm(embed_dim) if out_norm else nn.Identity()
        self.readout = nn.Linear(embed_dim, dims)

        # theta_gamma is structural (not gated) on memrecall; gateable add-on otherwise
        self.gate_names = [m for m in GATEABLE if self.cfg[m]
                           and not (task == "memrecall" and m == "theta_gamma")]
        if gated and self.gate_names:
            self.gates = nn.Parameter(torch.full((len(self.gate_names),), 2.0))

        if aux_heads:
            names = [n for n in ("conjunctive", "grid_attractor", "theta_gamma") if self.cfg[n]]
            self.aux = nn.ModuleDict({n: nn.Linear(embed_dim, dims) for n in names})

    def _g(self, name: str):
        if not self.gated or name not in self.gate_names:
            return 1.0
        return torch.sigmoid(self.gates[self.gate_names.index(name)])

    def gate_values(self) -> dict:
        if not self.gated or not self.gate_names:
            return {}
        return {n: round(torch.sigmoid(self.gates[i]).item(), 3) for i, n in enumerate(self.gate_names)}

    def gate_l1(self):
        if not self.gated or not self.gate_names:
            return torch.tensor(0.0)
        return torch.sigmoid(self.gates).sum()

    def _hidden(self, heading, speed, vz, k=None):
        """Compute the integrated hidden representation h (B, embed_dim) plus the
        intermediates needed for aux heads. Shared by encode() and forward()."""
        B, T = heading.shape
        device = heading.device
        step = torch.zeros(B, T, self.embed_dim, device=device)
        if self.cfg["conjunctive"]:
            step = step + (self.conjunctive(heading.reshape(B * T), speed.reshape(B * T)).view(B, T, -1)
                           + self.vert(vz.reshape(B * T, 1)).view(B, T, -1))
        # constrained mode: the grid modules integrate the RAW self-motion velocity (the signal
        # the conjunctive head-direction×speed cells encode), not the learned embedding.
        v3d = (torch.stack([speed * heading.cos(), speed * heading.sin(), vz], dim=-1)
               if self.constrained else None)

        if self.task == "pathint":
            if self.constrained:
                position = self.integrator(v3d)
            elif self.cfg["grid_attractor"]:
                position = self.integrator(step)
            else:
                position = self.pool_proj(step.mean(1))
        else:
            # recall / memrecall need per-step running positions
            if self.constrained:
                states = self.integrator(v3d, return_sequence=True)
            elif self.cfg["grid_attractor"]:
                states = self.integrator(step, return_sequence=True)
            else:
                states = step
            if self.task == "recall":
                idx = torch.arange(T, device=device)
                attn = torch.softmax(self.step_key(k) @ self.step_key(idx).t()
                                     / math.sqrt(self.embed_dim), dim=-1)
                position = (attn.unsqueeze(-1) * states).sum(dim=1)
            else:  # memrecall — through a fixed-size memory bottleneck
                if self.tg_mem is not None:
                    m = self.tg_mem.store(states)                 # ordered multiplex
                    position = self.tg_mem.retrieve(m, k)         # read slot k
                else:
                    m = states.mean(dim=1)                        # order-less bottleneck
                    position = self.bottleneck_read(torch.cat([m, self.q_embed(k)], dim=-1))

        h = position
        tg = None
        if self.tg_addon is not None:
            tg = self.tg_addon(step)
            h = h + self._g("theta_gamma") * tg
        if self.column is not None:
            h = h + self._g("cortical_column") * self.column(h)
        if self.lateral is not None:
            h = h + self._g("lateral_inhibition") * self.lateral(h)
        h = self.out_norm(h)
        return h, step, position, tg

    def encode(self, heading, speed, vz, k=None):
        """Integrated trajectory representation (B, embed_dim) — the spatial summary
        the LLM consumes in Milestone 2 (TrajectoryLLM)."""
        return self._hidden(heading, speed, vz, k)[0]

    def forward(self, heading, speed, vz, k=None, return_aux=False):
        h, step, position, tg = self._hidden(heading, speed, vz, k)
        pred = self.readout(h)
        if return_aux and self.aux_heads:
            aux_out = {}
            if "conjunctive" in self.aux:
                aux_out["conjunctive"] = self.aux["conjunctive"](step.mean(dim=1))
            if "grid_attractor" in self.aux:
                aux_out["grid_attractor"] = self.aux["grid_attractor"](position)
            if tg is not None and "theta_gamma" in self.aux:
                aux_out["theta_gamma"] = self.aux["theta_gamma"](tg)
            return pred, aux_out
        return pred
