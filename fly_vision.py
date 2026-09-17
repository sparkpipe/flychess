"""V1: THE FLY SEES — raw piece-planes -> reach-map perception.
Operator program 2026-09-17: vision -> eval loop -> move. This module is
the perception layer: no piece-eyes, no precomputed attacks — 13 raw planes
(12 piece x color bitmaps + 1 query marker) enter the fly's sensory
epithelium, propagate through the FROZEN connectome, and a reach-perception
readout must reproduce the piece-centered direction x depth reach map
(labels: exact, from reach_map()).
Protocol: same laws as the movement curriculum — stable seeds, held-out
batteries, 0.98 gate on the informative bins, replay mix of passed lessons,
scaffold-free by construction.
"""
import os, sys, json, time, random
import numpy as np
import torch
import chess

_LAB = os.path.expanduser("~") + "/chess-lab"
sys.path.insert(0, _LAB)
import fly_curriculum as cur
import flyfeat_cb

DEV = cur.DEV
STATE = os.environ.get(
    "VSTATE", _LAB + "/fly_sees_current.pt")
LOGF = _LAB + "/vision_log.jsonl"
PLANES = 12
INDIM = (PLANES + 1) * 64                     # + query-marker plane
VORDER = ["rook", "bishop", "queen", "knight", "pawn", "king"]
PTY = {"rook": chess.ROOK, "bishop": chess.BISHOP, "queen": chess.QUEEN,
       "knight": chess.KNIGHT, "pawn": chess.PAWN, "king": chess.KING}
HARD_BIN_W = {}                               # bin index -> loss weight


def planes_of(b, qs):
    """(13, 64) float: 12 piece x color occupancy planes + query marker."""
    p = np.zeros((PLANES + 1, 64), np.float32)
    for s_ in chess.SQUARES:
        pc = b.piece_at(s_)
        if pc is None:
            continue
        p[(pc.piece_type - 1) * 2 + (0 if pc.color == chess.WHITE else 1),
          s_] = 1.0
    if qs is not None:
        p[PLANES, qs] = 1.0
    return p


def board_input(b, qs):
    return planes_of(b, qs).reshape(-1)


class VisionFly(torch.nn.Module):
    """Raw planes -> sensory epithelium (trainable) -> FROZEN connectome
    -> reach-perception readout."""

    def __init__(self, readout_n=2048):
        super().__init__()
        z = np.load(cur.BRAIN)
        m = torch.sparse_csr_tensor(
            torch.from_numpy(z["indptr"].astype(np.int64)),
            torch.from_numpy(z["indices"].astype(np.int64)),
            torch.from_numpy(z["data"].astype(np.float32)),
            size=tuple(z["shape"]), device=DEV).t().to_sparse_csr()
        self.WT = m
        self.N = m.shape[0]
        sens = np.load(cur.SENS).astype(np.int64)
        self.register_buffer("inj_idx", torch.from_numpy(sens).to(DEV))
        self.W_sens = torch.nn.Linear(INDIM, len(sens))
        rng = np.random.default_rng(20260917)
        free = np.setdiff1d(np.arange(self.N), sens)
        self.register_buffer(
            "head_idx", torch.from_numpy(
                rng.permutation(free)[:readout_n].astype(np.int64)).to(DEV))
        self.reach_head = torch.nn.Sequential(
            torch.nn.Linear(readout_n, 512), torch.nn.GELU(),
            torch.nn.Linear(512, 64))

    def see(self, planes):
        s = torch.clamp(self.W_sens(planes), -6, 6)
        a = torch.zeros(self.N, planes.shape[0], device=DEV)
        a[self.inj_idx] = s.T
        max_steps = int(os.environ.get("VPROP", str(cur.PROP_STEPS)))
        eps = float(os.environ.get("VEPS", "0.02"))
        _prev = None
        for _ in range(max_steps):
            a = torch.clamp((1 - cur.LEAK) * a + cur.LEAK * (self.WT @ a),
                            -cur.CAP, cur.CAP)
            if _prev is not None and float(
                    (a - _prev).abs().mean()) < eps:
                break
            _prev = a
        return a

    def forward(self, planes):
        a = self.see(planes)
        return self.reach_head(a[self.head_idx].T)


def build_battery(rng, piece, n=96):
    """Held-out (planes, target) pairs from the stage-1 board family."""
    ins, tg = [], []
    prng = rng if isinstance(rng, random.Random) else random.Random(rng)
    for _ in range(n):
        bs = cur.gen_stage(prng, 1, 1, piece=PTY[piece])
        if not bs:
            continue
        b, spec = bs[0]
        qs = spec.get("leg_sq")
        ins.append(board_input(b, qs))
        tg.append(cur.reach_map(b, qs))
    return np.stack(ins), np.stack(tg)


def binacc(pred, tgt):
    """(full, nonzero-target) bin accuracies at code tolerance 0.25."""
    ok = (np.abs(pred - tgt) < 0.25)
    nz = np.abs(tgt) > 1e-6
    return float(ok.mean()), float(ok[nz].mean()) if nz.any() else 1.0


def eval_battery(model, piece, n=96, seed=4200):
    rng = random.Random(seed + cur._phash(piece))
    ins, tg = build_battery(rng, piece, n)
    model.eval()
    with torch.no_grad():
        pred = model(torch.from_numpy(ins).to(DEV)).cpu().numpy()
    full, nz = binacc(pred, tg)
    return full, nz


VLAST_ACC = {}                        # piece -> last full binacc


def train_step(model, opt, rng, piece, hard=None):
    bs = cur.gen_stage(rng, 1, cur.B, piece=PTY[piece])
    if not bs:
        return 0.0
    ins = np.stack([board_input(b, spec.get("leg_sq")) for b, spec in bs])
    tg = np.stack([cur.reach_map(b, spec.get("leg_sq")) for b, spec in bs])
    w = np.where(np.abs(tg) == 1.0, 25.0,
                 np.where(np.abs(tg) > 1e-6, 3.0, 1.0)).astype(np.float32)
    for (sb, k), v in (hard or {}).items():
        pass
    xt = torch.from_numpy(ins).to(DEV)
    yt = torch.from_numpy(tg).to(DEV)
    wt = torch.from_numpy(w).to(DEV)
    pred = model(xt)
    loss = (((pred - yt) ** 2) * wt).mean()
    opt.zero_grad()
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    opt.step()
    return float(loss.item())


def main():
    torch.manual_seed(0)
    model = VisionFly().to(DEV)
    if os.path.exists(STATE):
        model.load_state_dict(torch.load(STATE, weights_only=True))
        print("resumed", flush=True)
    else:
        print("FRESH vision fly", flush=True)
    opt = torch.optim.Adam(model.parameters(), lr=3e-4)
    passed = []
    for piece in VORDER:
        print(f"=== V1 MILESTONE {piece}-vision ===", flush=True)
        rng = random.Random(5000 + cur._phash(PTY[piece]))
        best = -1.0
        stall = 0
        for step in range(1, 6001):
            model.binding_scale = 0.0
            train_step(model, opt, rng, piece)
            _w = [max(0.02, 0.98 - VLAST_ACC.get(p, 0.9)) for p in passed]
            rp = rng.choices(passed, weights=_w)[0]
            train_step(model, opt, rng, rp)
            if step % 50:
                continue
            full, nz = eval_battery(model, piece)
            VLAST_ACC[piece] = full
            print(json.dumps({"v1_milestone": piece, "step": step,
                              "full": round(full, 4), "nz": round(nz, 4)}),
                  flush=True)
            if full >= 0.98 and nz >= 0.98:
                print(f"V1 MILESTONE {piece}-vision PASS at step {step} "
                      f"(full={full:.3f} nz={nz:.3f})", flush=True)
                torch.save(model.state_dict(), STATE)
                passed.append(piece)
                break
            if step >= 100 and full <= best + 0.001:
                stall += 1
            else:
                stall = 0
                best = max(best, full)
            if stall >= 8:
                print(f"V1 {piece} STOP at step {step} (best {best:.3f})",
                      flush=True)
                stall = 0
                best = max(best, full)
            if step == 6000:
                print(f"V1 {piece} EXHAUSTED (best {best:.3f})", flush=True)
                torch.save(model.state_dict(), STATE)
                sys.exit(1)
        else:
            continue
    torch.save(model.state_dict(), STATE)
    print("V1 ALL MILESTONES PASSED", flush=True)


if __name__ == "__main__":
    main()
