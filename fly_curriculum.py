#!/usr/bin/env python3
"""Staged chess curriculum for the fly brain — ground-up, gated.

Every stage trains on PROCEDURALLY GENERATED positions (no data-quality
doubts) and must pass its held-out gate before the next stage starts.

  S1 piece-movement      legality BCE, one own piece, open board      >= 0.99
  S2 blocking/capture    legality BCE, piece + blocker/enemy piece    >= 0.99
  S3 three pieces        legality BCE + protected-capture CE           >= 0.99
  S4 king/pins/forks     their-king legality + my-king + pins (BCE);
                         forks + discovered (CE)                       >= 0.99
  S5 basic mates         KQvK mate-in-1 CE == 100% on 200 held out     == 1.00

All positions are CANONICAL (mover plays up the board; flyfeat_cb) so
training is color-blind: every pattern exists in exactly one form.

Usage: python3 fly_curriculum.py <stage> [--steps 4000]
State: fly_cb.pt  (evolving; stages build on the previous stage's weights)
"""
import sys, os, json, time, random, chess
import numpy as np
import torch

sys.path.insert(0, "/home/spec/chess-lab")
import fly_v3_full as v3
import scipy.sparse as sp
import flyfeat_cb

DEV = "cuda"
STATE = "/home/spec/chess-lab/fly_cb.pt"
LOGF = "/home/spec/chess-lab/fly_cb_log.jsonl"
BRAIN = "/home/spec/chess-lab/brain_graph.npz"
SENS = "/home/spec/chess-lab/sensory_idx.npy"
PROP_STEPS, LEAK, CAP = 2, 0.5, 20.0
B = int(os.environ.get("B", "256"))
MAXL = 256


class FlyCB(torch.nn.Module):
    def __init__(self, n_feats):
        super().__init__()
        z = np.load(BRAIN)
        m = sp.csr_matrix((z["data"], z["indices"], z["indptr"]),
                          shape=tuple(z["shape"])).astype(np.float32)
        sens = np.load(SENS)
        self.sensory_idx = torch.from_numpy(sens.astype(np.int64)).to(DEV)
        self.N = m.shape[0]
        rng = np.random.default_rng(20260912)
        free = np.setdiff1d(np.arange(m.shape[0]), sens)
        p2 = rng.permutation(len(free))
        self.readout_idx = torch.from_numpy(free[p2[:4096]].astype(np.int64)).to(DEV)
        mt = m.T.tocsr()
        self.WT = torch.sparse_csr_tensor(
            torch.from_numpy(mt.indptr.astype(np.int64)),
            torch.from_numpy(mt.indices.astype(np.int64)),
            torch.from_numpy(mt.data), size=mt.shape, device=DEV)
        self.W_sens = torch.nn.Linear(n_feats, len(sens))
        self.theta = torch.nn.Parameter(torch.zeros(4096))
        self.theta_mv = torch.nn.Parameter(torch.zeros(8))
        self.theta_cls = torch.nn.Parameter(torch.zeros(3))
        self.cls_idx = torch.from_numpy(free[p2[4096:4099]].astype(np.int64)).to(DEV)

    def propagate(self, fvb):
        x = torch.from_numpy(fvb).to(DEV)
        s = torch.clamp(self.W_sens(x), -6, 6)
        a = torch.zeros(self.N, x.shape[0], device=DEV)
        a[self.sensory_idx] = s.T
        for _ in range(PROP_STEPS):
            a = torch.clamp((1 - LEAK) * a + LEAK * (self.WT @ a), -CAP, CAP)
        return a

    def logits_all(self, a):
        """(B, 4096) slot logits — used by legality BCE and move CE."""
        return self.theta.unsqueeze(0) * a[self.readout_idx].T


# ---------------- procedural generators ----------------
def fresh_kings(rng, dmin=4):
    while True:
        a, b = rng.sample(range(64), 2)
        if chess.square_distance(a, b) >= dmin:
            return a, b


def make_board(rng, pieces):
    """pieces = [(sq, pt, color)]; returns None if invalid/dead/over."""
    b = chess.Board(None)
    for sq, pt, col in pieces:
        b.set_piece_at(sq, chess.Piece(pt, col))
    if not b.is_valid() or b.is_game_over():
        return None
    return b


def gen_stage(rng, stage, batch):
    """Yield list of (canonical_board, spec): leg_sq, target_mv, cls."""
    out = []
    while len(out) < batch:
        spec = {}
        if stage == 1:
            k1, k2 = fresh_kings(rng)
            pt = rng.choice([chess.KNIGHT, chess.BISHOP, chess.ROOK,
                             chess.QUEEN, chess.PAWN])
            cand = [s for s in range(64) if s not in (k1, k2)]
            if pt == chess.PAWN:
                cand = [s for s in cand if 8 <= s < 48]
            s = rng.choice(cand)
            b = make_board(rng, [(k1, chess.KING, chess.WHITE),
                                 (k2, chess.KING, chess.BLACK),
                                 (s, pt, chess.WHITE)])
            if b is None:
                continue
            spec = {"leg_sq": s, "cls": 1}
        elif stage == 2:
            k1, k2 = fresh_kings(rng)
            pt = rng.choice([chess.KNIGHT, chess.BISHOP, chess.ROOK,
                             chess.QUEEN, chess.PAWN])
            cand = [s for s in range(64) if s not in (k1, k2)]
            if pt == chess.PAWN:
                cand = [s for s in cand if 8 <= s < 48]
            s = rng.choice(cand)
            s2 = rng.choice([x for x in cand if x != s])
            if rng.random() < 0.5:      # friendly blocker
                bpt = rng.choice([chess.PAWN, chess.KNIGHT, chess.BISHOP,
                                  chess.ROOK])
                pieces = [(k1, chess.KING, chess.WHITE), (s, pt, chess.WHITE),
                          (s2, bpt, chess.WHITE), (k2, chess.KING, chess.BLACK)]
            else:                        # enemy piece (captures appear)
                ept = rng.choice([chess.PAWN, chess.KNIGHT, chess.BISHOP,
                                  chess.ROOK, chess.QUEEN])
                e_cand = [x for x in cand if x != s]
                if pt == chess.PAWN and not e_cand:
                    continue
                s2 = rng.choice(e_cand)
                pieces = [(k1, chess.KING, chess.WHITE), (s, pt, chess.WHITE),
                          (s2, ept, chess.BLACK), (k2, chess.KING, chess.BLACK)]
            b = make_board(rng, pieces)
            if b is None:
                continue
            spec = {"leg_sq": s, "cls": 1}
        elif stage == 3:
            k1, k2 = fresh_kings(rng)
            pt = rng.choice([chess.KNIGHT, chess.BISHOP, chess.ROOK,
                             chess.QUEEN])
            cand = [s for s in range(64) if s not in (k1, k2)]
            s = rng.choice(cand)
            rest = [x for x in cand if x != s]
            rng.shuffle(rest)
            own2, own3 = rest[0], rest[1]
            enemy = rest[2]
            ept = rng.choice([chess.PAWN, chess.KNIGHT, chess.BISHOP,
                              chess.ROOK])
            defended = rng.random() < 0.5
            pieces = [(k1, chess.KING, chess.WHITE), (s, pt, chess.WHITE),
                      (own2, rng.choice([chess.PAWN, chess.KNIGHT,
                                         chess.BISHOP, chess.ROOK]), chess.WHITE),
                      (own3, rng.choice([chess.PAWN, chess.KNIGHT,
                                         chess.BISHOP, chess.ROOK]), chess.WHITE),
                      (enemy, ept, chess.BLACK), (k2, chess.KING, chess.BLACK)]
            b = make_board(rng, pieces)
            if b is None:
                continue
            if defended:
                # place a defender adjacent-ish to the enemy piece (same color)
                near = [q for q in chess.SQUARES
                        if q not in (k1, k2, s, own2, own3, enemy)
                        and chess.square_distance(q, enemy) <= 2]
                if not near:
                    continue
                b.set_piece_at(rng.choice(near),
                               chess.Piece(rng.choice([chess.PAWN, chess.KNIGHT]),
                                           chess.BLACK))
            # CE target: capture the enemy piece IF it is legal, else any move
            cap = chess.Move(s, enemy) if chess.Move(s, enemy) in b.legal_moves \
                else None
            if cap is None:
                cap = rng.choice(list(b.legal_moves))
            spec = {"leg_sq": s, "target_mv": cap, "cls": 1}
        elif stage == 4:
            mode = rng.choice(["their_king", "my_king", "pin", "fork",
                               "discovered"])
            k1, k2 = fresh_kings(rng, 4)
            if mode in ("their_king", "my_king", "pin"):
                pt = rng.choice([chess.ROOK, chess.BISHOP, chess.QUEEN])
                cand = [s for s in range(64) if s not in (k1, k2)]
                s = rng.choice(cand)
                pieces = [(k1, chess.KING, chess.WHITE), (s, pt, chess.WHITE),
                          (k2, chess.KING, chess.BLACK)]
                if mode == "pin":
                    # own piece pinned between my king and their slider
                    slider = rng.choice([chess.ROOK, chess.BISHOP, chess.QUEEN])
                    dirs = [(1, 0), (-1, 0), (0, 1), (0, -1),
                            (1, 1), (1, -1), (-1, 1), (-1, -1)]
                    df, dr = rng.choice(dirs)
                    kf, kr = chess.square_file(k1), chess.square_rank(k1)
                    pf, pr = kf + df, kr + dr
                    sf2, sr2 = kf + 2 * df, kr + 2 * dr
                    if not (0 <= pf < 8 and 0 <= pr < 8 and 0 <= sf2 < 8
                            and 0 <= sr2 < 8):
                        continue
                    psq = chess.square(pf, pr)
                    ssq = chess.square(sf2, sr2)
                    pieces = [(k1, chess.KING, chess.WHITE),
                              (psq, rng.choice([chess.KNIGHT, chess.BISHOP,
                                                chess.ROOK, chess.QUEEN]),
                               chess.WHITE),
                              (ssq, slider, chess.BLACK),
                              (k2, chess.KING, chess.BLACK)]
                b = make_board(rng, pieces)
                if b is None:
                    continue
                if mode == "their_king":
                    spec = {"leg_sq": k2, "cls": 1}
                elif mode == "my_king":
                    spec = {"leg_sq": k1, "cls": 1}
                else:
                    spec = {"leg_sq": pieces[1][0], "cls": 1}
            elif mode == "fork":
                # my knight forks two enemy pieces via one jump
                center = rng.randrange(8, 56)
                jumps = [d for d in
                         ((17, 15, 10, 6, -17, -15, -10, -6))]
                rng.shuffle(jumps)
                hits = []
                for d in jumps[:2]:
                    t = center + d
                    if 0 <= t < 64 and abs((t % 8) - (center % 8)) <= 2:
                        hits.append(t)
                if len(hits) < 2:
                    continue
                back = center - rng.choice(jumps)
                if not (0 <= back < 64
                        and abs((back % 8) - (center % 8)) <= 2):
                    continue
                pieces = [(k1, chess.KING, chess.WHITE),
                          (back, chess.KNIGHT, chess.WHITE),
                          (k2, chess.KING, chess.BLACK),
                          (hits[0], rng.choice([chess.ROOK, chess.QUEEN]),
                           chess.BLACK),
                          (hits[1], rng.choice([chess.ROOK, chess.QUEEN]),
                           chess.BLACK)]
                b = make_board(rng, pieces)
                if b is None:
                    continue
                mv = chess.Move(back, center)
                if mv not in b.legal_moves:
                    continue
                spec = {"target_mv": mv, "cls": 2}
            else:  # discovered
                df, dr = rng.choice([(1, 0), (-1, 0), (0, 1), (0, -1),
                                     (1, 1), (1, -1), (-1, 1), (-1, -1)])
                f0 = rng.randrange(0, 8)
                r0 = rng.randrange(0, 8)
                sq_slider = chess.square(f0, r0)
                sq_front = chess.square(min(max(f0 + df, 0), 7),
                                        min(max(r0 + dr, 0), 7))
                if sq_front == sq_slider:
                    continue
                sq_t = chess.square(min(max(f0 + 3 * df, 0), 7),
                                    min(max(r0 + 3 * dr, 0), 7))
                if sq_t in (sq_slider, sq_front, k1, k2):
                    continue
                slider = rng.choice([chess.ROOK, chess.BISHOP, chess.QUEEN])
                if slider == chess.BISHOP and (df == 0 or dr == 0):
                    continue
                if slider == chess.ROOK and df != 0 and dr != 0:
                    continue
                front = rng.choice([chess.KNIGHT, chess.PAWN, chess.BISHOP])
                if front == chess.PAWN and dr == 0:
                    continue
                pieces = [(k1, chess.KING, chess.WHITE),
                          (sq_slider, slider, chess.WHITE),
                          (sq_front, front, chess.WHITE),
                          (sq_t, rng.choice([chess.KNIGHT, chess.BISHOP,
                                             chess.ROOK]), chess.BLACK),
                          (k2, chess.KING, chess.BLACK)]
                b = make_board(rng, pieces)
                if b is None:
                    continue
                cand = [m for m in b.legal_moves if m.from_square == sq_front]
                if not cand:
                    continue
                mv = cand[0] if chess.Move(sq_front, sq_t) in b.legal_moves \
                    else rng.choice(cand)
                b.push(mv)
                opens = bool(board_attacks(b, sq_slider, sq_t))
                b.pop()
                if not opens:
                    continue
                spec = {"target_mv": mv, "cls": 2}
        elif stage == 5:
            b, tgt = gen_mate1(rng)
            if b is None:
                continue
            spec = {"target_mv": tgt, "cls": 2}
        out.append((b, spec))
    return out


def board_attacks(b, from_sq, to_sq):
    pc = b.piece_at(from_sq)
    return pc is not None and bool(b.attacks_mask(from_sq)
                                   & chess.BB_SQUARES[to_sq])


def gen_mate1(rng):
    """Constructed KQvK mate-in-1: enemy king on the edge, my queen delivers
    protected by my king."""
    edge = rng.choice([0, 7])
    bk = rng.randrange(8) + edge * 8          # enemy king on an edge rank
    bf, br = chess.square_file(bk), chess.square_rank(bk)
    # my king covers the delivery square
    del_f, del_r = bf, min(max(br + (1 if br == 0 else -1), 0), 7)
    del_sq = chess.square(del_f, del_r)
    k_cands = [s for s in chess.SQUARES
               if chess.square_distance(s, del_sq) == 1
               and chess.square_distance(s, bk) >= 2]
    if not k_cands:
        return None, None
    myk = rng.choice(k_cands)
    # queen starts 2 files away on the delivery rank (not checking the king)
    made = None
    for df in (-2, 2, -1, 1):
        qf = del_f + df
        if not (0 <= qf < 8) or chess.square(qf, del_r) in (bk, myk):
            continue
        q0 = chess.square(qf, del_r)
        b = chess.Board(None)
        b.set_piece_at(bk, chess.Piece(chess.KING, chess.BLACK))
        b.set_piece_at(myk, chess.Piece(chess.KING, chess.WHITE))
        b.set_piece_at(q0, chess.Piece(chess.QUEEN, chess.WHITE))
        if not b.is_valid() or b.is_game_over():
            continue
        mv = chess.Move(q0, del_sq)
        if mv not in b.legal_moves:
            continue
        b.push(mv)
        if b.is_checkmate():
            b.pop()
            made = (b, mv)
            break
        b.pop()
    if made is None:
        return None, None
    return made


# ---------------- batch building ----------------
def build_batch(rng, stage, batch):
    boards_specs = gen_stage(rng, stage, batch)
    fvb = np.stack([flyfeat_cb.feat_vec(b)[0] for b, _ in boards_specs])
    slotb = np.zeros((len(boards_specs), MAXL), np.int64)
    mfb = np.zeros((len(boards_specs), MAXL, 8), np.float32)
    maskb = np.zeros((len(boards_specs), MAXL), bool)
    tgtb = np.full(len(boards_specs), -1, dtype=np.int64)
    clb = np.zeros(len(boards_specs), dtype=np.int64)
    leg_idx, leg_slots = [], []
    for i, (b, spec) in enumerate(boards_specs):
        mvs = list(b.legal_moves)
        K = min(len(mvs), MAXL)
        for j, mv in enumerate(mvs[:K]):
            slotb[i, j] = mv.from_square * 64 + mv.to_square
            mfb[i, j] = flyfeat_cb.move_feats(b, mv)
            maskb[i, j] = True
        if "leg_sq" in spec:
            qs = spec["leg_sq"]
            y = np.zeros(MAXL, dtype=np.float32)
            for j, mv in enumerate(mvs[:K]):
                if mv.from_square == qs:
                    y[j] = 1.0
            leg_idx.append(i)
            leg_slots.append((y, int(maskb[i].sum())))
        if "target_mv" in spec:
            tgtb[i] = mvs.index(spec["target_mv"])
        clb[i] = spec.get("cls", 1)
    return fvb, slotb, mfb, maskb, tgtb, clb, leg_idx, leg_slots


def forward(model, fvb, slotb, mfb, maskb):
    a = model.propagate(fvb)
    Bn = a.shape[1]
    cols = torch.arange(Bn, device=DEV).unsqueeze(1)
    slots = torch.from_numpy(slotb).to(DEV)
    T = model.theta[slots] * a[model.readout_idx[slots], cols] \
        + torch.from_numpy(mfb).to(DEV) @ model.theta_mv
    T = T.masked_fill(~torch.from_numpy(maskb).to(DEV), -1e9)
    T_all = model.logits_all(a)                                # (B, 4096)
    cls = model.theta_cls.unsqueeze(0) * torch.tanh(a[model.cls_idx].T / 4.0)
    return torch.log_softmax(T, dim=1), T_all, cls


def train_stage(model, opt, stage, steps, rng):
    t0 = time.time()
    for step in range(1, steps + 1):
        bb = build_batch(rng, stage, B)
        fvb, slotb, mfb, maskb, tgtb, clb, leg_idx, leg_slots = bb
        logp, T_all, cls = forward(model, fvb, slotb, mfb, maskb)
        loss = 0.0
        if leg_idx:
            yy = torch.zeros(len(leg_idx), 4096, device=DEV)
            for r, (y, K) in enumerate(leg_slots):
                idx = torch.tensor([int(s) for s in slotb[leg_idx[r], :K]],
                                   device=DEV)
                yy[r, idx] = torch.from_numpy(y[:K]).to(DEV)
            rows = T_all[leg_idx]
            loss = loss + torch.nn.functional.binary_cross_entropy_with_logits(
                rows, yy)
        mv = torch.from_numpy(tgtb).to(DEV)
        mvk = mv >= 0
        if int(mvk.sum()):
            loss = loss - logp[mvk].gather(
                1, mv[mvk][:, None]).mean() * 2.0
        clst = torch.from_numpy(clb).to(DEV)
        loss = loss + torch.nn.functional.cross_entropy(cls, clst) * 0.5
        opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        if step % 100 == 0:
            print(json.dumps({"stage": stage, "step": step,
                              "loss": round(float(loss.item()), 4),
                              "sps": round(step / (time.time() - t0), 1)}),
                  flush=True)
        if step % 500 == 0:
            torch.save(model.state_dict(), STATE + ".tmp")
            os.replace(STATE + ".tmp", STATE)


def gate_stage(model, stage, rng):
    """Held-out gate: legality pairwise >= 0.99, CE top-1 == 1.00."""
    model.eval()
    bb = build_batch(rng, stage, 400)
    fvb, slotb, mfb, maskb, tgtb, clb, leg_idx, leg_slots = bb
    with torch.no_grad():
        logp, T_all, cls = forward(model, fvb, slotb, mfb, maskb)
    ok_pair = tot_pair = 0
    for r, (y, K) in enumerate(leg_slots):
        pos = [j for j in range(K) if y[j] > 0]
        neg = [j for j in range(K) if y[j] == 0]
        if not pos or not neg:
            continue
        row = T_all[leg_idx[r]]                  # (4096,) slot space
        srow = slotb[leg_idx[r]]                 # move-list j -> slot
        for p in pos[:3]:
            for q in neg[:3]:
                tot_pair += 1
                ok_pair += int(row[srow[p]] > row[srow[q]])
    pair = ok_pair / max(tot_pair, 1)
    mv = tgtb >= 0
    if int(mv.sum()):
        pred = torch.argmax(logp, dim=1).cpu().numpy()
        top1 = float((pred[mv] == tgtb[mv]).mean())
    else:
        top1 = 1.0
    model.train()
    return pair, top1


STAGE_GATE = {1: (0.99, None), 2: (0.99, None), 3: (0.99, 1.00),
              4: (0.99, 1.00), 5: (None, 1.00), 6: (None, 0.98)}

# ---------------- stage 6: tablebase-graded endings ----------------
# operator rule: reinforce the best move, but grade every move by its state
# change — a no-progress tempo is worse than progress, progress-wasting is
# worse still, crossing the dtz>=100 counter LOSES the win (draw-level),
# losing the win outright is -0.6, and in lost positions longer resistance
# is less bad.

CAT_V = {"win": 1.0, "cursed_win": 0.5, "draw": 0.0,
         "cursed_loss": -0.5, "loss": -1.0}
CLS_MAP = {"win": 2, "cursed_win": 2, "draw": 1, "cursed_loss": 0, "loss": 0}
POOL_DIR = "/home/spec/chess-lab/tbpools"


def load_pools(names):
    import glob as _g
    rows = []
    for nm in names:
        for p in _g.glob(f"{POOL_DIR}/{nm}.jsonl"):
            with open(p) as f:
                for line in f:
                    try:
                        rows.append(json.loads(line))
                    except Exception:
                        pass
    print(f"pools {names}: {len(rows)} exact-labeled positions", flush=True)
    return rows


def graded_targets(entry, b):
    """Per-legal-move value targets from the tablebase children."""
    cat = entry["cat"]
    ch = entry.get("children", {})
    opt = {"win": "loss", "cursed_win": "loss", "draw": "draw",
           "cursed_loss": "win", "loss": "win"}[cat]
    # child category is from the OPPONENT's perspective; flip to ours
    def ours(c):
        return {"win": "loss", "loss": "win", "draw": "draw",
                "cursed_win": "cursed_loss", "cursed_loss": "cursed_win"}[c]
    vals = {}
    child_ours = {}
    for uci, c in ch.items():
        cc = c.get("cat")
        if cc is None:
            continue
        o = ours(cc)
        dtz = c.get("dtz")
        child_ours[uci] = (o, dtz if dtz is not None else 99)
    if cat in ("win", "cursed_win"):
        winners = [(u, d) for u, (o, d) in child_ours.items() if o == "loss"]
        best_dtz = min((d for _, d in winners), default=99)
        for u, (o, d) in child_ours.items():
            if o == "loss":
                if d >= 98:                    # counter cliff: win evaporates
                    vals[u] = -0.4
                else:
                    vals[u] = 1.0 - min(d - best_dtz, 30) * 0.02
            elif o == "draw":
                vals[u] = -0.6                 # lost the win
            else:
                vals[u] = -1.0                 # lost the game
    elif cat == "draw":
        for u, (o, d) in child_ours.items():
            if o == "draw":
                vals[u] = 0.6                  # hold the draw
            else:
                vals[u] = -1.0                 # drifted into loss
    else:                                      # lost: resist longest
        for u, (o, d) in child_ours.items():
            if o == "win":
                vals[u] = -1.0 + min(d, 100) / 250.0
            elif o == "draw":
                vals[u] = 0.4                  # salvation draw
            else:
                vals[u] = 0.6
    return vals


def build_tb_batch(rng, rows, batch):
    buf = []
    while len(buf) < batch:
        e = rows[rng.randrange(len(rows))]
        try:
            b = chess.Board(e["fen"])
            if b.is_game_over():
                continue
            buf.append((b, e))
        except Exception:
            continue
    fvb = np.stack([flyfeat_cb.feat_vec(b)[0] for b, _ in buf])
    slots = []
    tgts = []
    clb = np.array([CLS_MAP.get(e["cat"], 1) for _, e in buf], dtype=np.int64)
    bests = []
    for b, e in buf:
        mvs = list(b.legal_moves)
        ch = e.get("children", {})
        tv = graded_targets(e, b)
        sv, vv, best = [], [], None
        for mv in mvs:
            u = mv.uci()
            if u in tv:
                sv.append(mv.from_square * 64 + mv.to_square)
                vv.append(tv[u])
            if u == e["best"]:
                best = len(sv) - 1
        slots.append((np.array(sv, dtype=np.int64),
                      np.array(vv, dtype=np.float32)))
        tgts.append(best if best is not None else -1)
        bests.append(1)
    cl = torch.from_numpy(clb).to(DEV)
    return fvb, slots, tgts, cl


def tb_step(model, opt, rows, rng):
    fvb, slots, tgts, cl = build_tb_batch(rng, rows, B)
    a = model.propagate(fvb)
    Bn = a.shape[1]
    cols = torch.arange(Bn, device=DEV).unsqueeze(1)
    cls = model.theta_cls.unsqueeze(0) * torch.tanh(a[model.cls_idx].T / 4.0)
    loss_cls = torch.nn.functional.cross_entropy(cls, cl)
    # per-move value regression on raw slot logits
    losses = []
    ce_terms = []
    for i, (sv, vv) in enumerate(slots):
        if len(sv) < 2:
            continue
        s = torch.from_numpy(sv).to(DEV)
        t = torch.from_numpy(vv).to(DEV)
        Trow = model.theta[s] * a[model.readout_idx[s], i] \
            + torch.zeros(len(s), device=DEV)
        losses.append(torch.nn.functional.huber_loss(Trow, t, delta=0.5))
    loss_val = torch.stack(losses).mean() if losses else torch.zeros((), device=DEV)
    loss = loss_val * 2.0 + loss_cls
    opt.zero_grad()
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    opt.step()
    return float(loss.item())


def gate_tb(model, rows, rng):
    """Gate: argmax(value head) ∈ optimal-move set on held-out entries."""
    model.eval()
    ok = tot = 0
    with torch.no_grad():
        for _ in range(400):
            e = rows[rng.randrange(len(rows))]
            try:
                b = chess.Board(e["fen"])
            except Exception:
                continue
            if b.is_game_over():
                continue
            mvs = list(b.legal_moves)
            if not mvs:
                continue
            fv, _ = flyfeat_cb.feat_vec(b)
            a = model.propagate(fv[None, :])
            s = torch.tensor([m.from_square * 64 + m.to_square
                              for m in mvs], device=DEV)
            T = model.theta[s] * a[model.readout_idx[s], 0]
            pick = mvs[int(torch.argmax(T))]
            # optimal set: moves whose child category preserves the outcome
            ch = e.get("children", {})
            opt = {"win": "loss", "cursed_win": "loss", "draw": "draw",
                   "cursed_loss": "win", "loss": "win"}[e["cat"]]
            optset = {u for u, c in ch.items()
                      if {"win": "loss", "loss": "win", "draw": "draw",
                          "cursed_win": "cursed_loss",
                          "cursed_loss": "cursed_win"}.get(c.get("cat")) == opt}
            tot += 1
            ok += int(pick.uci() in optset)
    model.train()
    return ok / max(tot, 1), tot


def main():
    stage = int(sys.argv[1])
    steps = int(sys.argv[sys.argv.index("--steps") + 1]) if "--steps" in sys.argv \
        else 4000
    v3.feat_vec(chess.Board())
    flyfeat_cb.feat_vec(chess.Board())
    model = FlyCB(len(flyfeat_cb.FEATURE_KEYS)).to(DEV)
    if os.path.exists(STATE):
        model.load_state_dict(torch.load(STATE, weights_only=True))
        print("resumed", flush=True)
    opt = torch.optim.Adam(model.parameters(), lr=3e-4)
    rng = random.Random(1000 + stage)
    train_stage(model, opt, stage, steps, rng)
    while True:
        grng = random.Random(999_000 + stage)
        pair, top1 = gate_stage(model, stage, grng)
        gate_pair, gate_top = STAGE_GATE[stage]
        ok_p = gate_pair is None or pair >= gate_pair
        ok_t = gate_top is None or top1 >= gate_top
        rec = {"stage": stage, "pair": round(pair, 4), "top1": round(top1, 4),
               "pass": bool(ok_p and ok_t)}
        print(json.dumps(rec), flush=True)
        with open(LOGF, "a") as f:
            f.write(json.dumps(rec) + "\n")
        if ok_p and ok_t:
            torch.save(model.state_dict(), STATE)
            print(f"STAGE {stage} PASSED — cleared for stage {stage+1}",
                  flush=True)
            break
        train_stage(model, opt, stage, steps, rng)


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "smoke":
        import flyfeat_cb as fc
        v3.feat_vec(chess.Board())
        fc.feat_vec(chess.Board())
        v0, NAMES = fc.feat_vec(chess.Board())
        rng = random.Random(5)
        for st in (1, 2, 3, 4, 5):
            bb = gen_stage(random.Random(st), st, 60)
            print(f"stage {st}: {len(bb)}/60 generated")
        print("feature dims:", len(NAMES))
    else:
        main()
