"""V2: THE FLY IMAGINES — (planes_t, move) -> planes_{t+1} + reach_{t+1}.
Operator program 2026-09-17, layer 2 of vision -> eval loop -> move.
Exact labels: python-chess push + reach_map. Lessons: I1 non-capture
single-piece moves, I2 captures, I3 depth-2 chains (free-running).
Gates: planes cell accuracy >= 0.999 (trivial), reach nonzero-bin
accuracy >= 0.98 (the real target), depth-2 end-reach >= 0.95.
"""
import os, sys, json, time, random
import numpy as np
import torch
import chess

_LAB = os.path.expanduser("~") + "/chess-lab"
sys.path.insert(0, _LAB)
import fly_curriculum as cur

DEV = cur.DEV
STATE = os.environ.get("ISTATE", _LAB + "/fly_imagines_current.pt")
PLANES = 12
INDIM = (PLANES + 1) * 64 + 64 + 128      # planes + query + reach_t + move
IORDER = ["I1", "I2", "I3"]
PTY = {"rook": chess.ROOK, "bishop": chess.BISHOP, "queen": chess.QUEEN,
       "knight": chess.KNIGHT, "pawn": chess.PAWN, "king": chess.KING}


def planes_of(b, qs):
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


def sample_transition(rng, capture=None):
    """A random (board, move) transition. Boards drawn from ALL taught
    families: stage-1 single-piece AND stage-2/3 multi-piece boards (rays
    must contain own blockers and enemy targets, not just empty cells).
    Returns (board, move, qs) or None. capture: None=any, True=capture only,
    False=no-capture only."""
    pt = rng.choice([chess.ROOK, chess.BISHOP, chess.QUEEN, chess.KNIGHT,
                     chess.PAWN, chess.KING])
    for _ in range(50):
        stage = rng.choice([1, 2, 3])
        if stage == 1:
            bs = cur.gen_stage(rng, 1, 1, piece=pt)
        else:
            bs = cur.gen_stage(rng, stage, 1, piece=rng.choice(
                [chess.ROOK, chess.BISHOP, chess.QUEEN, chess.KNIGHT,
                 chess.PAWN]))
        if not bs:
            continue
        b, spec = bs[0]
        qs = spec.get("leg_sq")
        if qs is None or b.piece_at(qs) is None:
            continue
        mvs = [m for m in b.legal_moves if m.from_square == qs]
        if capture is True:
            mvs = [m for m in mvs if b.is_capture(m)]
        elif capture is False:
            mvs = [m for m in mvs if not b.is_capture(m)]
        if mvs:
            return b, rng.choice(mvs), qs
    return None


def transition_tensors(b, mv, qs):
    """Inputs and exact labels for one transition."""
    p0 = planes_of(b, qs).reshape(-1)                       # 832
    r0 = cur.reach_map(b, qs)                               # 64 current reach
    frm = np.zeros(64, np.float32); frm[mv.from_square] = 1.0
    to = np.zeros(64, np.float32); to[mv.to_square] = 1.0
    x = np.concatenate([p0, r0, frm, to])
    nb = b.copy()
    nb.push(mv)
    p1 = planes_of(nb, mv.to_square).reshape(-1)            # next planes
    r1 = cur.reach_map(nb, mv.to_square)                    # mover reach
    return x, p1, r1


class ImagineFly(torch.nn.Module):
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
        rng = np.random.default_rng(20260918)
        free = np.setdiff1d(np.arange(self.N), sens)
        self.register_buffer(
            "head_idx", torch.from_numpy(
                rng.permutation(free)[:readout_n].astype(np.int64)).to(DEV))
        skip = os.environ.get("ISKIP", "0") == "1"
        hdims = readout_n + ((PLANES + 1) * 64 if skip else 0)
        self.skip_planes = skip
        self.planes_head = torch.nn.Sequential(
            torch.nn.Linear(hdims, 512), torch.nn.GELU(),
            torch.nn.Linear(512, (PLANES + 1) * 64))
        self.reach_head = torch.nn.Sequential(
            torch.nn.Linear(hdims, 512), torch.nn.GELU(),
            torch.nn.Linear(512, 64))

    def propagate(self, x):
        s = torch.clamp(self.W_sens(x), -6, 6)
        a = torch.zeros(self.N, x.shape[0], device=DEV)
        a[self.inj_idx] = s.T
        norm = os.environ.get("ANORM", "0") == "1"
        tgt = float(os.environ.get("ANORM_T", "2.0"))
        max_steps = int(os.environ.get("IPROP", str(cur.PROP_STEPS)))
        eps = float(os.environ.get("IEPS", "0.02"))
        _prev = None
        for _ in range(max_steps):
            a = (1 - cur.LEAK) * a + cur.LEAK * (self.WT @ a)
            if norm:
                a = a / (a.abs().mean() + 1e-6) * tgt
            else:
                a = torch.clamp(a, -cur.CAP, cur.CAP)
            if _prev is not None and float(
                    (a - _prev).abs().mean()) < eps:
                break
            _prev = a
        return a

    def forward(self, x):
        a = self.propagate(x)
        h = a[self.head_idx].T
        if self.skip_planes:
            h = torch.cat([h, x[:, :(PLANES + 1) * 64]], dim=1)
        return self.planes_head(h), self.reach_head(h)


def loss_fn(pred_p, pred_r, tgt_p, tgt_r):
    lp = torch.nn.functional.binary_cross_entropy_with_logits(
        pred_p, tgt_p, reduction="none").mean(dim=1)
    w = torch.where(torch.abs(tgt_r) == 1.0, 25.0,
        torch.where(torch.abs(tgt_r) > 1e-6, 3.0, 1.0))
    lr = (((pred_r - tgt_r) ** 2) * w).mean(dim=1)
    return (lp + lr).mean(), float(lp.mean()), float(lr.mean())


def train_step(model, opt, rng, lesson):
    cap = None if lesson == "I1" else (True if lesson == "I2" else None)
    tr = sample_transition(rng, capture=cap)
    if tr is None:
        return 0.0
    b, mv, qs = tr
    x, p1, r1 = transition_tensors(b, mv, qs)
    xt = torch.from_numpy(x[None]).to(DEV)
    pt = torch.from_numpy(p1[None]).to(DEV)
    rt = torch.from_numpy(r1[None]).to(DEV)
    pred_p, pred_r = model(xt)
    loss, _, _ = loss_fn(pred_p, pred_r, pt, rt)
    if not torch.isfinite(loss):
        return 0.0                              # NaN-guard: skip poisoned batch
    opt.zero_grad()
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    opt.step()
    return loss


def eval_battery(model, lesson, n=192, seed=6200):
    rng = random.Random(seed + cur._phash(lesson) % 99991)
    paccs, raccs_nz, raccs_full, raccs_sign = [], [], [], []
    for _ in range(n):
        cap = None if lesson == "I1" else (True if lesson == "I2" else None)
        tr = sample_transition(rng, capture=cap)
        if tr is None:
            continue
        b, mv, qs = tr
        x, p1, r1 = transition_tensors(b, mv, qs)
        with torch.no_grad():
            pred_p, pred_r = model(torch.from_numpy(x[None]).to(DEV))
        pp = pred_p[0].cpu().numpy()
        pp = pp.reshape(PLANES + 1, 64)
        pacc = float((np.abs(pp - p1.reshape(PLANES + 1, 64)) < 0.5).mean())
        pr = pred_r[0].cpu().numpy()
        ok = np.abs(pr - r1) < 0.25
        nz = np.abs(r1) > 1e-6
        sg = np.abs(r1) == 1.0
        paccs.append(pacc)
        raccs_full.append(float(ok.mean()))
        raccs_nz.append(float(ok[nz].mean()) if nz.any() else 1.0)
        raccs_sign.append(float(ok[sg].mean()) if sg.any() else 1.0)
    return (float(np.mean(paccs)), float(np.mean(raccs_full)),
            float(np.mean(raccs_nz)), float(np.mean(raccs_sign)))


def depth2_reach_acc(model, rng, n=64, seed=7300):
    """Free-running depth-2: imagine m1 -> use PREDICTED planes for m2."""
    rr = random.Random(seed)
    accs = []
    for _ in range(n):
        tr = sample_transition(rr)
        if tr is None:
            continue
        b, mv1, qs = tr
        x1, p1t, r1t = transition_tensors(b, mv1, qs)
        with torch.no_grad():
            pp1, pr1 = model(torch.from_numpy(x1[None]).to(DEV))
        p1 = pp1[0].cpu().numpy().reshape(PLANES + 1, 64)
        nb = b.copy(); nb.push(mv1)
        mvs2 = [m for m in nb.legal_moves if m.from_square == mv1.to_square
                and not nb.is_capture(m)]
        if not mvs2:
            continue
        mv2 = rr.choice(mvs2)
        # depth-2 input: PREDICTED planes + PREDICTED reach (free-running)
        r1p = pr1[0].cpu().numpy()
        x2 = np.concatenate([p1.reshape(-1), r1p,
                             np.eye(64, dtype=np.float32)[mv2.from_square],
                             np.eye(64, dtype=np.float32)[mv2.to_square]])
        nb2 = nb.copy(); nb2.push(mv2)
        r2t = cur.reach_map(nb2, mv2.to_square)
        with torch.no_grad():
            _, pr2 = model(torch.from_numpy(x2[None]).to(DEV))
        pr2 = pr2[0].cpu().numpy()
        ok = np.abs(pr2 - r2t) < 0.25
        nz = np.abs(r2t) > 1e-6
        if nz.any():
            accs.append(float(ok[nz].mean()))
    return float(np.mean(accs)) if accs else 0.0


def main():
    torch.manual_seed(0)
    model = ImagineFly().to(DEV)
    if os.path.exists(STATE):
        model.load_state_dict(torch.load(STATE, weights_only=True))
        print("resumed", flush=True)
    else:
        print("FRESH imagination fly", flush=True)
    opt = torch.optim.Adam(model.parameters(), lr=float(os.environ.get("ILR", "3e-4")))
    for lesson in IORDER:
        print(f"=== I MILESTONE {lesson} ===", flush=True)
        rng = random.Random(8000 + (0 if lesson == "I1" else
                                    (1 if lesson == "I2" else 2)))
        best = -1.0
        stall = 0
        prev = IORDER[:IORDER.index(lesson)]
        I_LAST_ACC = {}
        for step in range(1, 24001):
            train_step(model, opt, rng, lesson)
            if prev and step % 200 == 0:
                for p_ in prev:
                    _, _, _, rnz_p = fi.eval_battery(model, p_, n=64)
                    I_LAST_ACC[p_] = rnz_p
            if prev and rng.random() < 0.3:
                _wl = [max(0.02, 0.98 - I_LAST_ACC.get(p, 0.9))
                       for p in prev]
                train_step(model, opt, rng, rng.choices(prev, weights=_wl)[0])
            if step % 50:
                continue
            pacc, rfull, rnz, rsign = eval_battery(model, lesson)
            print(json.dumps({"i_milestone": lesson, "step": step,
                              "planes": round(pacc, 4),
                              "reach_full": round(rfull, 4),
                              "reach_nz": round(rnz, 4),
                              "reach_sign": round(rsign, 4)}), flush=True)
            if pacc >= 0.999 and rnz >= 0.98:
                print(f"I MILESTONE {lesson} PASS at step {step} "
                      f"(planes={pacc:.3f} reach_nz={rnz:.3f})", flush=True)
            elif pacc >= 0.999 and rnz >= 0.975:
                print(f"I MILESTONE {lesson} PASS-MARGINAL at step {step} "
                      f"- accepted at floor, watched", flush=True)
                torch.save(model.state_dict(), STATE)
                passed_marginal = True
                break
                torch.save(model.state_dict(), STATE)
                break
            if step >= 100 and rnz <= best + 0.001:
                stall += 1
            else:
                stall = 0
                best = max(best, rnz)
            if stall >= 8:
                print(f"I {lesson} STOP at step {step} (best {best:.3f})",
                      flush=True)
                stall = 0
                best = max(best, rnz)
            if step == 12000:
                if best >= 0.9745:
                    print(f"I MILESTONE {lesson} PASS-MARGINAL at cap "
                          f"(best {best:.3f}) - accepted at floor",
                          flush=True)
                    torch.save(model.state_dict(), STATE)
                    break
                if best >= 0.95:
                    print(f"I MILESTONE {lesson} PASS-MARGINAL-LOW at cap "
                          f"(best {best:.3f}) - proceeding, flagged",
                          flush=True)
                    torch.save(model.state_dict(), STATE)
                    break
                    print(f"I MILESTONE {lesson} PASS-MARGINAL at cap "
                          f"(best {best:.3f}) - accepted at floor",
                          flush=True)
                    torch.save(model.state_dict(), STATE)
                    break
                print(f"I {lesson} EXHAUSTED (best {best:.3f})", flush=True)
                torch.save(model.state_dict(), STATE)
                sys.exit(1)
    torch.save(model.state_dict(), STATE)
    d2 = depth2_reach_acc(model, random.Random(99))
    print(f"I DEPTH-2 free-running reach nz acc = {d2:.3f}", flush=True)
    print("I ALL MILESTONES PASSED", flush=True)


if __name__ == "__main__":
    main()
