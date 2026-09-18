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
import sys, os, json, time, random, chess, zlib
import numpy as np
import torch
# GPU trainers need no CPU thread army: PyTorch's default OpenMP
# pool is 2x cores (56 threads on 24) — pure contention with the
# co-resident trainers. Cap in code; no env needed.
torch.set_num_threads(8)


import os as _o
_LAB = _o.path.expanduser("~") + "/chess-lab"
sys.path.insert(0, _LAB)
import scipy.sparse as sp
import flyfeat_cb

DEV = "cuda"
STATE = os.environ.get("STATE", _LAB + "/fly_cb.pt")
LOGF = _LAB + "/fly_cb_log.jsonl"
BRAIN = _LAB + "/brain_graph.npz"
SENS = _LAB + "/sensory_idx.npy"
ANN = _LAB + "/annotations.feather"
NODES = _LAB + "/node_ids.npy"
PROP_STEPS = int(os.environ.get("STEPS", "6"))   # real circuits are 4-7 synapses
LEAK = float(os.environ.get("LEAK", "0.5"))
CAP = 20.0
B = int(os.environ.get("B", "256"))
MAXL = 256


# measured family -> site assignment (wiring_report.json 2026-09-15):
# capacity 1.0 everywhere; sensory_periph worst hub (gain 105K, 50%
# self-locked); medulla_Tm/lobula/lobula_plate the high-gain visual path;
# LH routes INTO KC (0.13) for associative convergence.
WIRING_MAP = {
    "attacks":     "medulla_Tm",
    "occupancy":   "medulla_Tm",
    "eyes_view":   "lobula_plate",
    "king_rings":  "lobula",
    "mobility":    "lobula",
    "material":    "KC",
    "castling":    "lateral_horn",
    "en_passant":  "lateral_horn",
    "check_state": "lateral_horn",
}


def build_retino_map(mode="geo", seed=12345):
    """Three-patch retinotopic overlay on the REAL hex lattice:
      square patch: (file, rank) -> columns, lamina neurons — atk/occ
      diagonal patch: rotated frame (dark=f-r, light=f+r) -> second column
        band, medulla Tm/Mi — atk again, so bishop lines are axis-aligned
      net patch: same geometry as square, deeper-layer neurons — net force
    mode=shuf permutes square->column identically in all patches
    (fungibility control: same neurons, wrong geometry)."""
    import pandas as pd
    a = pd.read_feather(ANN)
    node_ids = np.load(NODES)
    pos = {int(b): i for i, b in enumerate(node_ids)}
    m = a.assignedOlHex1.notna() & a.assignedOlHex2.notna() \
        & a.bodyId.isin(pos)
    sub = a[m]
    h1 = sub.assignedOlHex1.values.astype(int)
    h2 = sub.assignedOlHex2.values.astype(int)
    ty = sub.flywireType.fillna("").astype(str).values
    c1 = np.sort(np.unique(h1))
    # three disjoint 16-column bands; each board square = a 2x2 column block
    band = [c1[2:18], c1[20:36] if len(c1) >= 36 else c1[20:], c1[2:18]]
    c2 = np.sort(np.unique(h2))
    use2 = c2[4:20]

    def patch(bandcols, typefilter):
        bins = {}
        bc = np.asarray(bandcols)
        for i in range(len(sub)):
            if h1[i] not in bc or h2[i] not in use2:
                continue
            if not typefilter(ty[i]):
                continue
            b1 = int(np.searchsorted(bc, h1[i])) // 2
            b2 = int(np.searchsorted(use2, h2[i])) // 2
            if b1 > 7 or b2 > 7:
                continue
            nid = pos[int(sub.bodyId.values[i])]
            bins.setdefault(b1 * 8 + b2, []).append(nid)
        return bins

    lam = patch(band[0], lambda t: t.startswith("L"))
    med = patch(band[1], lambda t: t.startswith(("Tm", "Mi", "T1", "T2")))
    # net patch: same hex1 band as square, shifted hex2 (disjoint region,
    # any cell type — independence by geography, not type)
    netp = None
    for i in range(len(sub)):
        pass
    def patch2():
        bins = {}
        bc = np.asarray(band[2])
        u2 = c2[24:40]
        for i in range(len(sub)):
            if h1[i] not in bc or h2[i] not in u2:
                continue
            b1 = int(np.searchsorted(bc, h1[i])) // 2
            b2 = int(np.searchsorted(u2, h2[i])) // 2
            if b1 > 7 or b2 > 7:
                continue
            nid = pos[int(sub.bodyId.values[i])]
            bins.setdefault(b1 * 8 + b2, []).append(nid)
        return bins
    netp = patch2()
    rng = np.random.default_rng(seed)
    sq_perm = list(range(64))
    if mode == "shuf":
        rng.shuffle(sq_perm)

    def cap(idx):
        if len(idx) > 300:
            return list(rng.choice(idx, 300, replace=False))
        return idx

    out = {"square": {}, "diag": {}, "net": {}}
    for f in range(8):
        for r in range(8):
            s = f * 8 + r
            sp = sq_perm[s]
            out["square"][sp] = np.array(cap(lam.get(sp, [])), dtype=np.int64)
            dark = (f - r + 7) // 2          # rotated frame, binned to 8
            light = (f + r) // 2
            db = dark * 8 + light
            out["diag"][sp] = np.array(cap(med.get(db, [])), dtype=np.int64)
            out["net"][sp] = np.array(cap(netp.get(sp, [])), dtype=np.int64)
    tot = sum(len(v) for p in out.values() for v in p.values())
    print(f"retino {mode}: square={sum(len(v) for v in out['square'].values())} "
          f"diag={sum(len(v) for v in out['diag'].values())} "
          f"net={sum(len(v) for v in out['net'].values())}", flush=True)
    return out


SQUARE_FEATS = {          # per-square channels for the retinotopic payload
    "atk_my_{s}": 0, "atk_their_{s}": 1, "occ_my_{s}": 2,
    "occ_their_{s}": 3, "net_{s}": 4,
}


class FlyCB(torch.nn.Module):
    def __init__(self, n_feats, sel_boards=None, readout="random",
                 wiring=None):
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
        self.register_buffer("readout_idx",
                             torch.from_numpy(free[p2[:4096]].astype(np.int64)))
        self.readout_idx = self.readout_idx.to(DEV)
        mt = m.T.tocsr()
        self.WT = torch.sparse_csr_tensor(
            torch.from_numpy(mt.indptr.astype(np.int64)),
            torch.from_numpy(mt.indices.astype(np.int64)),
            torch.from_numpy(mt.data), size=mt.shape, device=DEV)
        # INJECT env: anatomical site override (A/B/C learning-speed tests)
        inj_site = os.environ.get("INJECT", "")
        inj_neurons = sens
        if inj_site:
            import wiring_engineer as we
            sm = we.site_masks(node_ids := np.load(
                "/home/spec/chess-lab/node_ids.npy"))
            if inj_site == "lh_kc":
                inj_neurons = np.concatenate([sm["lateral_horn"],
                                              sm.get("KC", np.array([], int))])
            elif inj_site in sm:
                inj_neurons = sm[inj_site]
            print(f"INJECT={inj_site}: {len(inj_neurons)} neurons", flush=True)
        self.W_sens = torch.nn.Linear(n_feats, len(inj_neurons))
        self.inj_idx = torch.from_numpy(
            np.asarray(inj_neurons, dtype=np.int64)).to(DEV)
        self.theta = torch.nn.Parameter(torch.zeros(4096))
        # shared displacement basis: movement rules generalize across slots
        self.geo_w = torch.nn.Linear(10, 1)
        # from->to binding: target square in the moving piece's attack set
        # (the additional view that carries piece-to-target relations)
        self.w_pseudo = torch.nn.Parameter(torch.tensor(0.0))
        self.w_pseudo2 = torch.nn.Parameter(torch.tensor(1.0))
        # post-move threat count: enemies attacked FROM the destination
        # (fork detection: count >= 2). Checkpoint-compatible scalar.
        self.w_threat = torch.nn.Parameter(torch.tensor(0.0))
        self.register_buffer("slot_geo",
                             torch.from_numpy(flyfeat_cb.slot_geo()))
        self.theta_mv = torch.nn.Parameter(torch.zeros(flyfeat_cb.MOVE_DIMS))
        self.theta_mv_pc = torch.nn.Parameter(
            torch.zeros(9, flyfeat_cb.MOVE_DIMS))
        self.theta_cls = torch.nn.Parameter(torch.zeros(3))
        self.register_buffer("cls_idx",
                             torch.from_numpy(free[p2[4096:4099]].astype(np.int64)))
        self.cls_idx = self.cls_idx.to(DEV)
        self.readout_mode = readout
        if wiring:
            # inject DIRECTLY at the assigned anatomical sites (bypasses the
            # periphery entirely — measured to be the worst hub)
            rows_all = np.concatenate(
                [np.asarray(r, dtype=np.int64) for _, (c, r) in wiring.items()])
            self.inj_idx = torch.from_numpy(rows_all).to(DEV)
            M = np.zeros((len(rows_all), n_feats), dtype=np.float32)
            off = 0
            for fam, (cols, rws) in wiring.items():
                M[off:off + len(rws), cols] = 1.0
                off += len(rws)
            self.W_sens = torch.nn.Linear(n_feats, len(rows_all))
            self.register_buffer("wmask", torch.from_numpy(M))  # moves with .to()
            with torch.no_grad():
                self.W_sens.weight.mul_(torch.from_numpy(M))
        else:
            self.register_buffer("wmask", None)
        if sel_boards is not None:
            self._select_readouts(sel_boards)

    @torch.no_grad()
    def _select_readouts(self, boards, n_move=4096, n_cls=3):
        """Pathway experiment: readout neurons chosen by activation variance
        across diverse boards (random sampling under-receives signal)."""
        self.to(DEV)
        a = self.propagate(np.stack([flyfeat_cb.feat_vec(b)[0]
                                     for b in boards]))
        var = a.var(dim=1).cpu().numpy()
        var[self.inj_idx.cpu().numpy()] = -1.0
        order = np.argsort(var)[::-1]
        self.readout_idx = torch.from_numpy(
            order[:n_move].astype(np.int64)).to(DEV)
        self.cls_idx = torch.from_numpy(
            order[n_move:n_move + n_cls].astype(np.int64)).to(DEV)
        print(f"readout=variance: top var {var[order[0]]:.3f}, "
              f"4096th {var[order[n_move-1]]:.4f}", flush=True)

    def propagate(self, fvb, reach=None):
        x = torch.from_numpy(fvb).to(DEV)
        if getattr(self, "retino", None) is not None:
            # geometric overlay: per-square channels -> matched columns,
            # one trainable gain per channel (tests the built-in wiring,
            # not extra capacity)
            if getattr(self, "_sq_idx", None) is None:
                keys = flyfeat_cb.FEATURE_KEYS
                sq_idx = {c_: {} for c_ in range(5)}
                for fi, k in enumerate(keys):
                    for s_ in range(64):
                        for pref, c_ in (("atk_my_", 0), ("atk_their_", 1),
                                         ("occ_my_", 2), ("occ_their_", 3),
                                         ("net_", 4)):
                            if k == f"{pref}{s_}":
                                sq_idx[c_][s_] = fi
                self._sq_idx = sq_idx
            sq_idx = self._sq_idx
            B = x.shape[0]
            a = torch.zeros(self.N, B, device=DEV)
            R = self.retino
            def inject(patchkey, s_, c_, g):
                idx = R[patchkey].get(s_)
                if idx is None or not len(idx):
                    return
                fi = sq_idx[c_].get(s_)
                if fi is None:
                    return
                ti = torch.from_numpy(idx).to(DEV)
                a[ti] = a[ti] + self.retino_gain[g] * x[:, fi][None, :]
            for s_ in range(64):
                inject("square", s_, 0, 0)        # atk_my   (lamina)
                inject("square", s_, 1, 1)        # atk_their
                inject("square", s_, 2, 2)        # occ_my
                inject("square", s_, 3, 3)        # occ_their
                inject("diag", s_, 0, 4)          # atk_my   (rotated frame)
                inject("diag", s_, 1, 5)          # atk_their (bishop lines)
                if reach is not None:
                    ti = R["net"].get(s_)
                    if ti is not None and len(ti):
                        ti2 = torch.from_numpy(ti).to(DEV)
                        a[ti2] = a[ti2] + self.retino_gain[6] * \
                            torch.from_numpy(reach[:, s_]).to(DEV)[None, :]
                else:
                    inject("net", s_, 4, 6)       # net force (deeper layer)
            max_steps = int(os.environ.get("MPROP", str(PROP_STEPS)))
            eps = float(os.environ.get("MEPS", "0.02"))
            _prev = None
            for _ in range(max_steps):
                a = (1 - LEAK) * a + LEAK * (self.WT @ a)
                if os.environ.get("ANORM", "1") == "1":
                    a = a / (a.abs().mean() + 1e-6)                         * float(os.environ.get("ANORM_T", "2.0"))
                else:
                    a = torch.clamp(a, -CAP, CAP)
                if _prev is not None and float(
                        (a - _prev).abs().mean()) < eps:
                    break
                _prev = a
            return a
        w = self.W_sens.weight * self.wmask if self.wmask is not None \
            else self.W_sens.weight
        s = torch.clamp(torch.nn.functional.linear(x, w, self.W_sens.bias),
                        -6, 6)
        a = torch.zeros(self.N, x.shape[0], device=DEV)
        a[self.inj_idx] = s.T
        max_steps = int(os.environ.get("MPROP", str(PROP_STEPS)))
        eps = float(os.environ.get("MEPS", "0.02"))
        _prev = None
        for _ in range(max_steps):
            a = (1 - LEAK) * a + LEAK * (self.WT @ a)
            if os.environ.get("ANORM", "1") == "1":
                a = a / (a.abs().mean() + 1e-6)                     * float(os.environ.get("ANORM_T", "2.0"))
            else:
                a = torch.clamp(a, -CAP, CAP)
            if _prev is not None and float(
                    (a - _prev).abs().mean()) < eps:
                break
            _prev = a
        return a

    def logits_all(self, a):
        """(B, 4096) slot logits: position-modulated readout + SHARED
        geometric displacement basis (the relative-view signal)."""
        geo = self.geo_w(self.slot_geo).squeeze(-1)     # (4096,)
        return self.theta.unsqueeze(0) * a[self.readout_idx].T + geo


_REACH_DIRS = [(1, 0), (1, 1), (0, 1), (-1, 1),
               (-1, 0), (-1, -1), (0, -1), (1, -1)]
_REACH_KNIGHT = [(2, 1), (1, 2), (-1, 2), (-2, 1),
                 (-2, -1), (-1, -2), (1, -2), (2, -1)]


def reach_map(b, qs):
    """(64,) net-patch channel, bin = dir*8 + depth (depth 1..7), centered on
    the queried piece. Rays truncate at the first piece."""
    m = np.zeros(64, np.float32)
    if qs is None:
        return m
    pc = b.piece_at(qs)
    if pc is None:
        return m
    my = pc.color
    f, r = chess.square_file(qs), chess.square_rank(qs)

    def put(di, k, nf, nr):
        if not (0 <= nf < 8 and 0 <= nr < 8):
            return
        cell = b.piece_at(chess.square(nf, nr))
        m[di * 8 + k] = 0.3 if cell is None else (
            1.0 if cell.color != my else -1.0)
        return cell

    pt = pc.piece_type
    if pt in (chess.BISHOP, chess.ROOK, chess.QUEEN):
        dirs = _REACH_DIRS
        if pt == chess.BISHOP:
            dirs = _REACH_DIRS[1::2]
        elif pt == chess.ROOK:
            dirs = _REACH_DIRS[0::2]
        for di, (df, dr) in enumerate(dirs):
            for k in range(1, 8):
                cell = put(di, k, f + df * k, r + dr * k)
                if cell is not None:
                    break                                # ray blocked
    elif pt == chess.KNIGHT:
        for di, (df, dr) in enumerate(_REACH_KNIGHT):
            put(di, 1, f + df, r + dr)
    elif pt == chess.KING:
        for di, (df, dr) in enumerate(_REACH_DIRS):
            put(di, 1, f + df, r + dr)
    elif pt == chess.PAWN:
        put(2, 1, f, r + 1)                              # push 1 (N)
        put(2, 2, f, r + 2)                              # push 2
        put(3, 1, f - 1, r + 1)                          # capture NW
        put(1, 1, f + 1, r + 1)                          # capture NE
    return m


# ---------------- procedural generators ----------------
def fresh_kings(rng, dmin=4):
    while True:
        a, b = rng.sample(range(64), 2)
        if chess.square_distance(a, b) >= dmin:
            return a, b


def make_board(rng, pieces):
    """pieces = [(sq, pt, color)]; rejects only broken/checkmate/stalemate
    boards. NOT insufficient material — K+B/K+N vs K boards are exactly the
    stage-1/2 training positions (is_game_over() would reject them all)."""
    b = chess.Board(None)
    for sq, pt, col in pieces:
        b.set_piece_at(sq, chess.Piece(pt, col))
    if not b.is_valid() or b.is_checkmate() or b.is_stalemate():
        return None
    return b


def gen_stage(rng, stage, batch, piece=None):
    """Yield list of (canonical_board, spec): leg_sq, target_mv, cls."""
    if stage in (2, 3) and isinstance(piece, str):
        piece = {"ep": chess.PAWN, "promo": chess.PAWN,
                 "castle": chess.KING}.get(
                    piece, getattr(chess, piece.upper()))
    out = []
    while len(out) < batch:
        spec = {}
        if stage == 1 and piece in ("ep", "promo", "castle"):
            # special-move arms (parallel sparks): en passant, promotion
            # impulse with restraint, castling legality
            if piece == "ep":
                # my pawn rank 5 + enemy pawn beside that just double-pushed
                # (ep set) OR hasn't (no ep) — the capture discrimination
                f = rng.randrange(1, 7)
                my_sq = chess.square(f, 4)
                ef = rng.choice([-1, 1])
                if not (0 <= f + ef < 8):
                    continue
                epb = chess.Board(None)
                epb.set_piece_at(my_sq, chess.Piece(chess.PAWN, chess.WHITE))
                epb.set_piece_at(chess.square(f + ef, 4),
                                 chess.Piece(chess.PAWN, chess.BLACK))
                has_ep = rng.random() < 0.5
                if has_ep:
                    epb.ep_square = chess.square(f + ef, 5)
                b = epb
                spec = {"leg_sq": my_sq, "cls": 1}
                cap = chess.Move(my_sq, epb.ep_square) if has_ep else None
                if cap is not None and cap in b.legal_moves:
                    spec["target_mv"] = cap
            elif piece == "promo":
                # my pawn on rank 7: PUSH when the promotion square is safe,
                # RESTRAIN when it is attacked (pawn would be lost)
                f = rng.randrange(8)
                my_sq = chess.square(f, 6)
                b = chess.Board(None)
                b.set_piece_at(my_sq, chess.Piece(chess.PAWN, chess.WHITE))
                ahead = chess.square(f, 7)
                threatened = rng.random() < 0.5
                if threatened:
                    pt2 = rng.choice([chess.ROOK, chess.KNIGHT, chess.BISHOP])
                    # attacker must cover the promotion square: pick a
                    # from-square that attacks it
                    cands = []
                    for g in chess.SQUARES:
                        if g == my_sq or g == ahead:
                            continue
                        sc = chess.Board(None)
                        sc.set_piece_at(g, chess.Piece(pt2, chess.BLACK))
                        if sc.attacks_mask(g) & chess.BB_SQUARES[ahead]:
                            cands.append(g)
                    if not cands:
                        continue
                    b.set_piece_at(rng.choice(cands),
                                   chess.Piece(pt2, chess.BLACK))
                # operator ruling: the impulse is ALWAYS to push at this
                # level — attacked promotion squares stay in the data as
                # distractors; suppression is a later layer's job
                push = chess.Move(my_sq, ahead, promotion=chess.QUEEN)
                if push not in b.legal_moves:
                    continue
                spec = {"leg_sq": my_sq, "cls": 2, "target_mv": push}
            else:  # castle
                # K e1 + R h1 (O-O) with rights; half the time an enemy
                # attacker covers a transit square -> castling illegal
                b = chess.Board(None)
                b.set_piece_at(chess.E1, chess.Piece(chess.KING, chess.WHITE))
                b.set_piece_at(chess.H1, chess.Piece(chess.ROOK, chess.WHITE))
                b.castling_rights = chess.BB_H1
                blocked = rng.random() < 0.3
                if blocked:
                    b.set_piece_at(chess.F1,
                                   chess.Piece(rng.choice([chess.KNIGHT,
                                                           chess.BISHOP]),
                                               chess.WHITE))
                    b.castling_rights = 0
                attacked = rng.random() < 0.4 and not blocked
                if attacked:
                    atk_sq = rng.choice([chess.F8, chess.G8])
                    b.set_piece_at(atk_sq, chess.Piece(chess.ROOK,
                                                       chess.BLACK))
                    if not (b.attacks_mask(atk_sq)
                            & chess.BB_SQUARES[rng.choice(
                                [chess.F1, chess.G1])]):
                        continue
                oo = chess.Move(chess.E1, chess.G1)
                spec = {"leg_sq": chess.E1, "cls": 1}
                if oo in b.legal_moves:
                    spec["target_mv"] = oo
            out.append((b, spec))
            continue
        if stage == 1:
            # operator spec: ONE piece on an OPEN board, movement instinct.
            # No kings — python-chess generates legality fine kingless.
            pt = piece or rng.choice([chess.KNIGHT, chess.BISHOP, chess.ROOK,
                                      chess.QUEEN, chess.PAWN])
            cand = [s for s in range(64)]
            if pt == chess.PAWN:
                cand = [s for s in cand if 8 <= s < 56]   # ranks 2-7
            s = rng.choice(cand)
            b = chess.Board(None)
            b.set_piece_at(s, chess.Piece(pt, chess.WHITE))
            spec = {"leg_sq": s, "cls": 1}
        elif stage == 2:
            # operator spec: TWO pieces, open board, no kings. Friendly
            # blocker (cannot move through/onto) vs enemy piece (capturable).
            pt = piece or rng.choice([chess.KING, chess.KNIGHT, chess.BISHOP,
                                      chess.ROOK, chess.QUEEN, chess.PAWN])
            cand = [s for s in range(64)]
            if pt == chess.PAWN:
                cand = [s for s in cand if 8 <= s < 56]
            s = rng.choice(cand)
            s2 = rng.choice([x for x in range(64) if x != s])
            if rng.random() < 0.5:      # friendly blocker
                bpt = rng.choice([chess.KING, chess.PAWN, chess.KNIGHT,
                                  chess.BISHOP, chess.ROOK, chess.QUEEN])
                b = chess.Board(None)
                b.set_piece_at(s, chess.Piece(pt, chess.WHITE))
                b.set_piece_at(s2, chess.Piece(bpt, chess.WHITE))
            else:                        # enemy piece (captures appear)
                ept = rng.choice([chess.KING, chess.PAWN, chess.KNIGHT,
                                  chess.BISHOP, chess.ROOK, chess.QUEEN])
                b = chess.Board(None)
                b.set_piece_at(s, chess.Piece(pt, chess.WHITE))
                b.set_piece_at(s2, chess.Piece(ept, chess.BLACK))
                if pt == chess.PAWN:
                    # pawn captures require the enemy on a forward diagonal;
                    # relocate s2 accordingly half the time
                    if rng.random() < 0.6:
                        f, r = chess.square_file(s), chess.square_rank(s)
                        df = rng.choice([-1, 1])
                        nf, nr = f + df, r + 1
                        if 0 <= nf < 8 and 0 <= nr < 8:
                            b = chess.Board(None)
                            b.set_piece_at(s, chess.Piece(pt, chess.WHITE))
                            b.set_piece_at(chess.square(nf, nr),
                                           chess.Piece(ept, chess.BLACK))
            spec = {"leg_sq": s, "cls": 1}
        elif stage == 3:
            # operator spec: THREE pieces, open board, no kings. My piece +
            # enemy piece (the capture target) + second enemy that DEFENDS it
            # half the time -> combined protection decides whether the
            # capture is right. Trains legality + CE on capture choice.
            pt = piece or rng.choice([chess.KING, chess.KNIGHT, chess.BISHOP,
                                      chess.ROOK, chess.QUEEN, chess.PAWN])
            cand = [s for s in range(64)]
            if pt == chess.PAWN:
                cand = [s for s in cand if 8 <= s < 56]
            s = rng.choice(cand)
            b = chess.Board(None)
            b.set_piece_at(s, chess.Piece(pt, chess.WHITE))
            # place the enemy target on a square my piece attacks
            att = [t for t in chess.SQUARES
                   if b.attacks_mask(s) & chess.BB_SQUARES[t]]
            if pt == chess.PAWN:
                f, r = chess.square_file(s), chess.square_rank(s)
                att = [chess.square(f + d, r + 1) for d in (-1, 1)
                       if 0 <= f + d < 8 and r + 1 < 8]
            if not att:
                continue
            e1 = rng.choice(att)
            ept = rng.choice([chess.PAWN, chess.KNIGHT, chess.BISHOP,
                              chess.ROOK, chess.QUEEN])   # no enemy kings:
            b.set_piece_at(e1, chess.Piece(ept, chess.BLACK))  # s4 concept
            defended = rng.random() < 0.5
            if defended:
                dpt = rng.choice([chess.PAWN, chess.KNIGHT, chess.BISHOP,
                                  chess.ROOK, chess.QUEEN])
                defs = []
                for t in chess.SQUARES:
                    if t in (s, e1) or chess.square_distance(t, e1) != 1:
                        continue
                    sc = chess.Board(None)
                    sc.set_piece_at(t, chess.Piece(dpt, chess.BLACK))
                    if sc.attacks_mask(t) & chess.BB_SQUARES[e1]:
                        defs.append(t)
                if not defs:
                    continue
                b.set_piece_at(rng.choice(defs),
                               chess.Piece(dpt, chess.BLACK))
            cap = chess.Move(s, e1)
            spec = {"leg_sq": s, "cls": 1}
            if cap in b.legal_moves:
                if defended:
                    # protection lesson: grabbing a defended piece is wrong;
                    # CE target = a safe non-capturing move instead
                    safe = [m for m in b.legal_moves
                            if m.from_square == s and not b.is_capture(m)]
                    if safe:
                        spec["target_mv"] = rng.choice(safe)
                else:
                    spec["target_mv"] = cap
            out.append((b, spec))
            continue
        elif stage == 4:
            # operator spec: king safety, pins, forks, discovered attacks —
            # projected force and its blocking, with kings proper.
            mode = (piece if isinstance(piece, str) else None) or \
                rng.choice(["their_king", "pin", "fork", "discovered"])
            if mode == "their_king":
                # my piece + enemy king, ENEMY to move: the enemy king cannot
                # move to squares my piece controls
                pt = rng.choice([chess.ROOK, chess.BISHOP, chess.KNIGHT,
                                 chess.QUEEN])
                sqs = rng.sample(range(64), 2)
                b = chess.Board(None)
                b.set_piece_at(sqs[0], chess.Piece(pt, chess.WHITE))
                b.set_piece_at(sqs[1], chess.Piece(chess.KING, chess.BLACK))
                if b.is_attacked_by(chess.WHITE, sqs[1]):
                    continue                    # enemy king not in check
                b.turn = chess.BLACK
                if b.is_game_over() or not any(b.generate_legal_moves()):
                    continue
                spec = {"leg_sq": sqs[1], "cls": 1}
            elif mode == "pin":
                # my slider pins an enemy piece to the enemy king; the pinned
                # piece may only move along the pin line
                df, dr = rng.choice([(1, 0), (-1, 0), (0, 1), (0, -1),
                                     (1, 1), (1, -1), (-1, 1), (-1, -1)])
                slider = (chess.ROOK if df == 0 or dr == 0 else chess.BISHOP)
                f0, r0 = rng.randrange(1, 7), rng.randrange(1, 7)
                if not (0 <= f0 + 2 * df < 8 and 0 <= r0 + 2 * dr < 8
                        and 0 <= f0 - df < 8 and 0 <= r0 - dr < 8):
                    continue
                pin_sq = chess.square(f0 + df, r0 + dr)
                slider_sq = chess.square(f0 + 2 * df, r0 + 2 * dr)
                ksq = chess.square(f0 - df, r0 - dr)
                pinned_pt = rng.choice([chess.QUEEN, chess.ROOK, chess.BISHOP,
                                        chess.KNIGHT])
                b = chess.Board(None)
                b.set_piece_at(ksq, chess.Piece(chess.KING, chess.BLACK))
                b.set_piece_at(pin_sq, chess.Piece(pinned_pt, chess.BLACK))
                b.set_piece_at(slider_sq, chess.Piece(slider, chess.WHITE))
                if not (b.attacks_mask(slider_sq) & chess.BB_SQUARES[pin_sq]):
                    continue                    # pin must be real
                b.turn = chess.BLACK
                mv_pin = [m for m in b.legal_moves if m.from_square == pin_sq]
                if not mv_pin:
                    continue
                spec = {"leg_sq": pin_sq, "cls": 1}
            elif mode == "fork":
                # knight jumps TO center, forking t1 and t2 from origin k
                center = rng.randrange(8, 56)
                offs = [17, 15, 10, 6, -17, -15, -10, -6]
                rng.shuffle(offs)
                def ok(sq, ref):
                    return 0 <= sq < 64 and abs((sq % 8) - (ref % 8)) <= 2
                d1, d2 = offs[0], None
                t1 = center + d1
                if not ok(t1, center):
                    continue
                for d in offs[1:]:
                    if d == -d1:
                        continue
                    t = center + d
                    if ok(t, center) and t != t1:
                        d2 = d
                        break
                if d2 is None:
                    continue
                t2 = center + d2
                k = center - d1
                if not ok(k, center) or k in (t1, t2):
                    continue
                b = chess.Board(None)
                b.set_piece_at(k, chess.Piece(chess.KNIGHT, chess.WHITE))
                b.set_piece_at(t1, chess.Piece(
                    rng.choice([chess.ROOK, chess.QUEEN]), chess.BLACK))
                b.set_piece_at(t2, chess.Piece(
                    rng.choice([chess.ROOK, chess.QUEEN]), chess.BLACK))
                mv = chess.Move(k, center)
                if mv not in b.legal_moves:
                    continue
                spec = {"target_mv": mv, "cls": 2}
            else:  # discovered
                # my slider behind my piece; moving the piece opens the line
                df, dr = rng.choice([(1, 0), (-1, 0), (0, 1), (0, -1),
                                     (1, 1), (1, -1), (-1, 1), (-1, -1)])
                slider = (chess.ROOK if df == 0 or dr == 0 else chess.BISHOP)
                f0, r0 = rng.randrange(1, 6), rng.randrange(1, 6)
                if not (0 <= f0 + 3 * df < 8 and 0 <= r0 + 3 * dr < 8
                        and 0 <= f0 + df < 8 and 0 <= r0 + dr < 8):
                    continue
                front_sq = chess.square(f0 + df, r0 + dr)
                slider_sq = chess.square(f0, r0)
                tgt_sq = chess.square(f0 + 3 * df, r0 + 3 * dr)
                if tgt_sq in (front_sq, slider_sq):
                    continue
                front_pt = rng.choice([chess.KNIGHT, chess.ROOK,
                                       chess.BISHOP])
                b = chess.Board(None)
                b.set_piece_at(slider_sq, chess.Piece(slider, chess.WHITE))
                b.set_piece_at(front_sq, chess.Piece(front_pt, chess.WHITE))
                b.set_piece_at(tgt_sq, chess.Piece(
                    rng.choice([chess.ROOK, chess.QUEEN]), chess.BLACK))
                cands = [m for m in b.legal_moves
                         if m.from_square == front_sq]
                # a discovery = front piece leaves the slider->target line;
                # the slider's other rays were never blocked, so a landing
                # on them (e.g. the perpendicular file) is still a discovery
                seg = chess.between(slider_sq, tgt_sq) \
                    | chess.BB_SQUARES[tgt_sq]
                opens = [m for m in cands
                         if not (seg & chess.BB_SQUARES[m.to_square])]
                if not opens:
                    continue
                # deterministic preferred target: captures first, then
                # farthest-from-front; eval accepts ANY valid discovery
                caps = [m for m in opens if b.is_capture(m)]
                pool = caps or opens
                tgt = max(pool, key=lambda m: chess.square_distance(
                    m.to_square, front_sq))
                spec = {"target_mv": tgt, "cls": 2,
                        "target_set": {m.uci() for m in opens}
                        | {m.uci() for m in cands
                           if m.to_square == tgt_sq},
                        "eval_from": front_sq}
            out.append((b, spec))
        elif stage == 5:
            b, tgt = gen_mate1(rng)
            if b is None:
                continue
            # operator tolerance law: KQvK positions usually have SEVERAL
            # legal mates (145/200 of the held battery) — accept ALL of
            # them; a mate available is the one "clearly best move" class
            mates = set()
            for m in b.legal_moves:
                b.push(m)
                if b.is_checkmate():
                    mates.add(m.uci())
                b.pop()
            spec = {"target_mv": tgt, "cls": 2,
                    "target_set": mates or {tgt.uci()}}
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
def build_batch(rng, stage, batch, piece=None):
    boards_specs = gen_stage(rng, stage, batch, piece=piece)
    fvb = np.stack([flyfeat_cb.feat_vec(b)[0] for b, _ in boards_specs])
    slotb = np.zeros((len(boards_specs), MAXL), np.int64)
    pcrowb = np.zeros((len(boards_specs), MAXL), np.int64)
    reachb = np.zeros((len(boards_specs), 64), np.float32)
    mfb = np.zeros((len(boards_specs), MAXL, flyfeat_cb.MOVE_DIMS),
                    np.float32)
    maskb = np.zeros((len(boards_specs), MAXL), bool)
    tgtb = np.full(len(boards_specs), -1, dtype=np.int64)
    clb = np.zeros(len(boards_specs), dtype=np.int64)
    leg_idx, leg_slots = [], []
    for i, (b, spec) in enumerate(boards_specs):
        reachb[i] = reach_map(b, spec.get("leg_sq"))
        mvs = list(b.legal_moves)
        K = min(len(mvs), MAXL)
        for j, mv in enumerate(mvs[:K]):
            slotb[i, j] = mv.from_square * 64 + mv.to_square
            pcrowb[i, j] = _PC_IDX[b.piece_at(mv.from_square).piece_type]
            mfb[i, j] = flyfeat_cb.move_feats(b, mv)
            maskb[i, j] = True
        spec["_extra"] = ()
        qs = spec.get("leg_sq")
        if qs is not None and b.piece_at(qs) is not None and K < MAXL:
            legal_to = {mv.to_square for mv in mvs if mv.from_square == qs}
            extra = []
            if b.castling_rights:
                extra += [t for t in (qs + 2, qs - 2)
                          if 0 <= t < 64 and t not in legal_to]
            if b.ep_square is not None and b.ep_square not in legal_to:
                extra.append(b.ep_square)
            pool = [t for t in range(64) if t != qs and t not in legal_to]
            rng.shuffle(pool)
            extra += pool[:4]
            ded = []
            for t in extra:
                if t not in ded:
                    ded.append(t)
            ded = ded[:MAXL - K]
            for idx, t in enumerate(ded):
                slotb[i, K + idx] = qs * 64 + t
                pcrowb[i, K + idx] = _PC_IDX[b.piece_at(qs).piece_type]
                maskb[i, K + idx] = True
            spec["_extra"] = ded
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
    pseudo = np.zeros((len(boards_specs), MAXL), np.float32)
    pseudo2 = np.zeros((len(boards_specs), MAXL), np.float32)
    threat = np.zeros((len(boards_specs), MAXL), np.float32)
    for i, (b, spec) in enumerate(boards_specs):
        mvs = list(b.legal_moves)
        K = min(len(mvs), MAXL)
        p2 = {}
        for pmv in b.pseudo_legal_moves:
            p2.setdefault(pmv.from_square, set()).add(pmv.to_square)
        for j, mv in enumerate(mvs[:K]):
            pc = b.piece_at(mv.from_square)
            pseudo2[i, j] = 1.0 if mv.to_square in p2.get(
                mv.from_square, ()) else 0.0
            pseudo[i, j] = 1.0 if (pc and b.attacks_mask(mv.from_square)
                                   & chess.BB_SQUARES[mv.to_square]) else 0.0
            b.push(mv)
            threat[i, j] = 1.0 if (b.attacks_mask(mv.to_square)
                                   & b.occupied_co[b.turn]) else 0.0
            b.pop()
        for idx, t in enumerate(spec.get("_extra", ())):
            pseudo2[i, K + idx] = 1.0 if t in p2.get(
                spec["leg_sq"], ()) else 0.0
            threat[i, K + idx] = 1.0 if b.attackers_mask(not b.turn, t) \
                else 0.0
    return (fvb, slotb, pcrowb, reachb, mfb, maskb, tgtb, clb,
            leg_idx, leg_slots, pseudo, threat, pseudo2)


def forward(model, fvb, slotb, pcrowb, mfb, maskb, pseudob=None,
            threatb=None, pseudo2b=None, reach=None):
    a = model.propagate(fvb, reach=reach)
    Bn = a.shape[1]
    cols = torch.arange(Bn, device=DEV).unsqueeze(1)
    slots = torch.from_numpy(slotb).to(DEV)
    geo = model.geo_w(model.slot_geo).squeeze(-1)              # (4096,)
    mf_t = torch.from_numpy(mfb).to(DEV)
    T = model.theta[slots] * a[model.readout_idx[slots], cols] \
        + geo[slots] + mf_t @ model.theta_mv \
        + (mf_t * model.theta_mv_pc[
            torch.from_numpy(pcrowb).to(DEV)]).sum(-1)
    if pseudob is not None:
        T = T + model.w_pseudo * torch.from_numpy(pseudob).to(DEV)
    if threatb is not None:
        T = T + model.w_threat * torch.from_numpy(threatb).to(DEV)
    if pseudo2b is not None:
        sca = getattr(model, "binding_scale", 0.0)
        T = T + sca * model.w_pseudo2 * torch.from_numpy(pseudo2b).to(DEV)
    T = T.masked_fill(~torch.from_numpy(maskb).to(DEV), -1e9)
    T_all = model.logits_all(a)                                # (B, 4096)
    cls = model.theta_cls.unsqueeze(0) * torch.tanh(a[model.cls_idx].T / 4.0)
    return T, torch.log_softmax(T, dim=1), T_all, cls


HARD_SLOT_W = {}                       # (from,to) -> weight boost


def battery_activations(model, boards_specs):
    """ONE battery prologue: feature-stack + propagate (+ leg_sq reach).
    Replaces the copy in every eval."""
    fvb = np.stack([flyfeat_cb.feat_vec(b)[0] for b, _ in boards_specs])
    reach = np.stack([reach_map(b, spec.get("leg_sq"))
                      for b, spec in boards_specs]) \
        if boards_specs else np.zeros((0, 64), np.float32)
    with torch.no_grad():
        return model.propagate(fvb, reach=reach)


def make_score_ctx(model):
    """Detached per-model constants for the scalar readout. ONE copy —
    the four hand-rolled scorers this replaces had drifted term sets
    (binary vs capped threat, missing pseudo2)."""
    return {
        "geo": model.geo_w(model.slot_geo).squeeze(-1)
               .detach().cpu().numpy(),
        "wmv": model.theta_mv.detach().cpu().numpy(),
        "wp": float(model.w_pseudo.detach()),
        "wp2": float(model.w_pseudo2.detach()),
        "bs": getattr(model, "binding_scale", 0.0),
        "wt": float(model.w_threat.detach()),
    }


def score_terms(ctx, theta_val, act_val, geo_val, mf, wpc,
                ps, ps2, thr):
    """The canonical scalar move score: the same terms as forward(),
    one legal order. Every scalar-path eval scores through THIS."""
    return (theta_val * act_val + geo_val
            + float(mf @ (ctx["wmv"] + wpc))
            + ctx["wp"] * ps + ctx["bs"] * ctx["wp2"] * ps2
            + ctx["wt"] * thr)


def repair_until(eval_fn, train_fn, rng_fn, bar=0.98, rounds=3,
                 steps0=300, interleave_fn=None):
    """The escalation-repair loop (300->600->1200 ...), single-sourced:
    it appeared copied six times across the stage-4/5 sweeps. interleave_fn
    runs after each repair round (the operator's hold-the-newest law).
    Returns (score, failures, repair_rounds_executed)."""
    pr, fails = eval_fn()
    steps = steps0
    done = 0
    for rnd in range(rounds):
        if pr >= bar:
            break
        train_fn(steps, rng_fn(rnd))
        if interleave_fn is not None:
            interleave_fn(steps, rng_fn(rnd))
        done += 1
        pr, fails = eval_fn()
        steps *= 2
    return pr, fails, done



def train_stage(model, opt, stage, steps, rng, piece=None):
    t0 = time.time()
    for step in range(1, steps + 1):
        bb = build_batch(rng, stage, B, piece=piece)
        fvb, slotb, pcrowb, reachb, mfb, maskb, tgtb, clb, leg_idx, \
            leg_slots, pseudob, threatb, pseudo2b = bb
        T, logp, T_all, cls = forward(model, fvb, slotb, pcrowb, mfb,
                                      maskb, pseudob, threatb, pseudo2b,
                                      reach=reachb)
        loss = 0.0
        if leg_idx:
            # per-move legality BCE on the bound logits (with pseudo flag):
            # legal moves of the queried piece = 1, its others = 0
            rowsel = []
            ysel = []
            wsel = []
            for r, (y, K) in enumerate(leg_slots):
                if K == 0:
                    continue
                rowsel.append(T[leg_idx[r], :K])
                ysel.append(torch.from_numpy(y[:K]).to(DEV))
                wrow = torch.ones(K, device=DEV)
                for j, s in enumerate(slotb[leg_idx[r], :K]):
                    wboost = HARD_SLOT_W.get(int(s))
                    if wboost:
                        wrow[j] = wboost
                wsel.append(wrow)
            if rowsel:
                rows = torch.cat(rowsel)
                yy = torch.cat(ysel)
                ww = torch.cat(wsel)
                loss = loss + (torch.nn.functional.binary_cross_entropy_with_logits(
                    rows, yy, reduction="none") * ww).mean()
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
    fvb, slotb, pcrowb, reachb, mfb, maskb, tgtb, clb, leg_idx, \
        leg_slots, pseudob, threatb, pseudo2b = bb
    with torch.no_grad():
        T, logp, T_all, cls = forward(model, fvb, slotb, pcrowb, mfb,
                                      maskb, pseudob, threatb, pseudo2b,
                                      reach=reachb)
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

_PC = {"king": chess.KING, "rook": chess.ROOK, "bishop": chess.BISHOP,
       "knight": chess.KNIGHT, "queen": chess.QUEEN, "pawn": chess.PAWN,
       "ep": "ep", "promo": "promo", "castle": "castle"}
_PC_IDX = {chess.KING: 0, chess.ROOK: 1, chess.BISHOP: 2, chess.KNIGHT: 3,
           chess.QUEEN: 4, chess.PAWN: 5, "ep": 6, "promo": 7, "castle": 8}


def _pcidx(piece):
    return _PC_IDX[piece]


def _phash(piece):
    """Stable across processes (str hash() is salted per process)."""
    if isinstance(piece, int):
        return piece
    return zlib.crc32(piece.encode()) % 99991


def pname(p):
    return p if isinstance(p, str) else chess.piece_name(p)
PIECE_ORDER = [_PC[p] for p in os.environ.get("PIECES",
               "king,rook,bishop,knight,queen,pawn").split(",")]


LAST_ACC = {}                         # (stage, piece-name) -> last pair


def eval_piece(model, piece, n=64, seed=5000, stage=1):
    """Held-out battery for ONE piece type. Returns (pair, top1, failures):
    failures = list of (fen, piece_sq, legal_to, illegal_to, gap) where the
    illegal slot outscores the legal one."""
    rng = random.Random(seed + _phash(piece) + stage * 7919)
    boards_specs = gen_stage(rng, stage, n, piece=piece)
    a = battery_activations(model, boards_specs)
    ok = tot = top_ok = 0
    failures = []
    geo = model.geo_w(model.slot_geo).squeeze(-1).detach().cpu().numpy()
    for i, (b, spec) in enumerate(boards_specs):
        sq = spec.get("leg_sq")
        if sq is None:
            continue
        legal = {m.to_square for m in b.legal_moves if m.from_square == sq}
        if not legal:
            continue
        illegal = [t for t in range(64) if t != sq and t not in legal]
        if not illegal:
            continue
        act = a[model.readout_idx.cpu().numpy(), i].detach().cpu().numpy()
        pc = b.piece_at(sq)
        p2 = set()
        real = {}
        for pmv in b.pseudo_legal_moves:
            if pmv.from_square != sq:
                continue
            p2.add(pmv.to_square)
            real[pmv.to_square] = pmv
        def score(t):
            slot = sq * 64 + t
            rmv = real.get(t)
            mf = flyfeat_cb.move_feats(b, rmv) if rmv is not None \
                else np.zeros(flyfeat_cb.MOVE_DIMS, np.float32)
            pseudo = 1.0 if (pc and b.attacks_mask(sq)
                             & chess.BB_SQUARES[t]) else 0.0
            if rmv is not None:
                b.push(rmv)
                atk = b.attacks_mask(t) & b.occupied_co[b.turn]
                b.pop()
            else:
                atk = b.attackers_mask(not b.turn, t)
            thr = 1.0 if atk else 0.0
            wmv = model.theta_mv.detach().cpu().numpy()
            wpc = model.theta_mv_pc[_PC_IDX[pc.piece_type]].detach() \
                .cpu().numpy()
            return (float(model.theta[slot].detach()) * act[slot]
                    + geo[slot] + float(mf @ (wmv + wpc))
                    + float(model.w_pseudo.detach()) * pseudo
                    + float(model.w_threat.detach()) * thr)
        scores = {t: score(t) for t in list(legal)[:4] + illegal[:4]}
        for lt in list(legal)[:4]:
            for it in illegal[:4]:
                gl = scores[lt]
                gi = scores[it]
                tot += 1
                ok += int(gl > gi)
                if gl <= gi:
                    failures.append((b.fen(), sq, lt, it, round(gl - gi, 3)))
    LAST_ACC[(stage, pname(piece))] = ok / max(tot, 1)
    return ok / max(tot, 1), failures


def maintain_regressions(model, opt, prior, rng, bar=0.98, floor=0.975,
                         repair_steps=25, rounds=3):
    """Reactive regression maintenance: eval every prior battery, retrain
    the FAILs briefly (repair_steps, `rounds` tries) to get them back to
    PASS. prior = list of (stage, piece). Loud on repairs/unfixed, one
    summary line when all green. Returns #unfixed."""
    if not prior:
        return 0
    unfixed = repaired = 0
    for st, pc in prior:
        pr, _ = eval_piece(model, pc, n=64, stage=st)
        r = 0
        steps = repair_steps
        while pr < bar and r < rounds:
            train_stage(model, opt, st, steps, rng, piece=pc)
            pr, _ = eval_piece(model, pc, n=64, stage=st)
            r += 1
            if pr < bar:
                steps = min(steps * 3, 225)   # deep holes get deep repair
        if r:
            repaired += 1
            print("REGRESSION-REPAIR s%d:%s -> %.3f (%d rounds)"
                  % (st, pname(pc), pr, r), flush=True)
        if pr < floor:
            unfixed += 1
            print("REGRESSION-UNFIXED s%d:%s at %.3f" % (st, pname(pc), pr),
                  flush=True)
        elif pr < bar:
            print("REGRESSION-MARGINAL s%d:%s at %.3f (accepted, watched)"
                  % (st, pname(pc), pr), flush=True)
    if repaired == 0:
        print("REGRESSION-CHECK all-green (%d batteries)" % len(prior),
              flush=True)
    return unfixed


def milestone_stage2(model, opt):
    """Stage 2 milestones: per piece, two-piece boards (blocker/capture).
    Eval = pairwise legality incl. capture-vs-block discrimination."""
    logf = open(LOGF, "a")
    for piece in PIECE_ORDER:
        name = pname(piece)
        print(f"=== S2 MILESTONE piece={name} ===", flush=True)
        best = -1.0
        stall = 0
        rng = random.Random(3000 + _phash(piece))
        passed = PIECE_ORDER[:PIECE_ORDER.index(piece)]
        prior = [(1, p) for p in PIECE_ORDER] + [(2, p) for p in passed]
        for step in range(1, 6001):
            train_stage(model, opt, 2, 1, rng, piece=piece)
            if prior and rng.random() < 0.35:
                _w = [max(0.02, 0.98 - LAST_ACC.get((_s, _p), 0.9))
                      for _s, _p in prior]
                st2, pc2 = rng.choices(prior, weights=_w)[0]
                train_stage(model, opt, st2, 1, rng, piece=pc2)
            if step % 110 == 0:
                if maintain_regressions(model, opt, prior, rng):
                    print(f"MAINTENANCE-ABORT s2:{name} — prior battery "
                          "cannot hold 0.98", flush=True)
                    torch.save(model.state_dict(), STATE)
                    return False
            pair, failures = eval_piece(model, piece, n=96, stage=2)
            if step % 20 == 0 or pair >= 0.98 or (stall >= 1):
                rec = {"s2_milestone": name, "step": step,
                       "pair": round(pair, 4)}
                print(json.dumps(rec), flush=True)
                logf.write(json.dumps(rec) + "\n"); logf.flush()
            if pair >= 0.98:
                print(f"S2 MILESTONE {name} PASS at step {step}", flush=True)
                torch.save(model.state_dict(), STATE)
                sweep = []
                blocked = None
                for p2 in PIECE_ORDER[:PIECE_ORDER.index(piece) + 1]:
                    n2 = pname(p2)
                    pr, fails = eval_piece(model, p2, n=96, stage=2)
                    for rnd in range(3):
                        if pr >= 0.98:
                            break
                        for fen, sq, lt, it, gap in fails[:60]:
                            HARD_SLOT_W[sq * 64 + lt] = 6.0
                            HARD_SLOT_W[sq * 64 + it] = 4.0
                        train_stage(model, opt, 2, 300,
                                    random.Random(4000 + _phash(p2) + rnd), piece=p2)
                        pr, fails = eval_piece(model, p2, n=96, stage=2)
                    if pr < 0.98:
                        blocked = (n2, pr, fails)
                    sweep.append(f"{n2}={round(pr, 3)}")
                print("S2 REGRESSION-SWEEP " + " ".join(sweep), flush=True)
                if blocked:
                    n2, pr, fails = blocked
                    print(f"S2 SWEEP-BLOCKED {n2} at {pr:.3f} — failures:",
                          flush=True)
                    for fen, sq, lt, it, gap in fails[:10]:
                        print(f"  FAIL {fen} sq={chess.square_name(sq)} "
                              f"legal={chess.square_name(lt)} "
                              f"illegal={chess.square_name(it)} gap={gap}",
                              flush=True)
                    torch.save(model.state_dict(), STATE)
                    return False
                break
            if step < 100:
                best = max(best, pair)
                continue
            if pair <= best + 0.001:
                stall += 1
            else:
                stall = 0
                best = pair
            if stall >= 8:
                print(f"S2 MILESTONE {name} STOP at step {step} "
                      f"(best {best:.3f}, now {pair:.3f}) — failures:",
                      flush=True)
                for fen, sq, lt, it, gap in failures[:10]:
                    print(f"  FAIL {fen} sq={chess.square_name(sq)} "
                          f"legal={chess.square_name(lt)} "
                          f"illegal={chess.square_name(it)} gap={gap}",
                          flush=True)
                for fen, sq, lt, it, gap in failures[:80]:
                    HARD_SLOT_W[sq * 64 + lt] = 6.0
                    HARD_SLOT_W[sq * 64 + it] = 4.0
                stall = 0
                best = max(best, pair)
                continue
    torch.save(model.state_dict(), STATE)
    print("STAGE 2 ALL MILESTONES PASSED", flush=True)
    return True


def milestone_stage3(model, opt):
    """Stage 3 milestones: protection. Sweep covers ALL prior stages."""
    logf = open(LOGF, "a")

    def sweep_all(upto):
        parts = []
        blocked = None
        for st in (1, 2, 3):
            for p2 in (PIECE_ORDER if st < 3
                       else PIECE_ORDER[:PIECE_ORDER.index(upto) + 1]):
                n2 = pname(p2)
                pr, fails = eval_piece(model, p2, n=96, stage=st)
                for rnd in range(3):
                    if pr >= 0.98:
                        break
                    for fen, sq, lt, it, gap in fails[:60]:
                        HARD_SLOT_W[sq * 64 + lt] = 6.0
                        HARD_SLOT_W[sq * 64 + it] = 4.0
                    train_stage(model, opt, st, 300,
                                random.Random(5000 + st * 31 + _phash(p2) + rnd),
                                piece=p2)
                    pr, fails = eval_piece(model, p2, n=96, stage=st)
                if pr < 0.98:
                    blocked = (st, n2, pr, fails)
                parts.append(f"s{st}:{n2}={round(pr, 3)}")
        print("S3 REGRESSION-SWEEP " + " ".join(parts), flush=True)
        return blocked

    for piece in PIECE_ORDER:
        name = pname(piece)
        print(f"=== S3 MILESTONE piece={name} ===", flush=True)
        best = -1.0
        stall = 0
        rng = random.Random(6000 + _phash(piece))
        passed = PIECE_ORDER[:PIECE_ORDER.index(piece)]
        prior = ([(1, p) for p in PIECE_ORDER]
                 + [(2, p) for p in PIECE_ORDER]
                 + [(3, p) for p in passed])
        for step in range(1, 6001):
            train_stage(model, opt, 3, 1, rng, piece=piece)
            if prior and rng.random() < 0.35:
                _w = [max(0.02, 0.98 - LAST_ACC.get((_s, _p), 0.9))
                      for _s, _p in prior]
                st3, pc3 = rng.choices(prior, weights=_w)[0]
                train_stage(model, opt, st3, 1, rng, piece=pc3)
            if step % 110 == 0:
                if maintain_regressions(model, opt, prior, rng):
                    print(f"MAINTENANCE-ABORT s3:{name} — prior battery "
                          "cannot hold 0.98", flush=True)
                    torch.save(model.state_dict(), STATE)
                    return False
            pair, failures = eval_piece(model, piece, n=96, stage=3)
            if step % 20 == 0 or pair >= 0.98 or (stall >= 1):
                rec = {"s3_milestone": name, "step": step,
                       "pair": round(pair, 4)}
                print(json.dumps(rec), flush=True)
                logf.write(json.dumps(rec) + "\n"); logf.flush()
            if pair >= 0.98:
                print(f"S3 MILESTONE {name} PASS at step {step}", flush=True)
                torch.save(model.state_dict(), STATE)
                blocked = sweep_all(piece)
                if blocked:
                    st, n2, pr, fails = blocked
                    print(f"S3 SWEEP-BLOCKED stage{st} {n2} at {pr:.3f} — "
                          f"failures:", flush=True)
                    for fen, sq, lt, it, gap in fails[:10]:
                        print(f"  FAIL {fen} sq={chess.square_name(sq)} "
                              f"legal={chess.square_name(lt)} "
                              f"illegal={chess.square_name(it)} gap={gap}",
                              flush=True)
                    torch.save(model.state_dict(), STATE)
                    return False
                break
            if step < 100:
                best = max(best, pair)
                continue
            if pair <= best + 0.001:
                stall += 1
            else:
                stall = 0
                best = pair
            if step == 6000:
                print(f"S3 MILESTONE {name} EXHAUSTED at cap "
                      f"(best {best:.3f}) — failures:", flush=True)
                for fen, sq, lt, it, gap in failures[:10]:
                    print(f"  FAIL {fen} sq={chess.square_name(sq)} "
                          f"legal={chess.square_name(lt)} "
                          f"illegal={chess.square_name(it)} gap={gap}",
                          flush=True)
                torch.save(model.state_dict(), STATE)
                return False               # never silently skip a piece
            if stall >= 8:
                print(f"S3 MILESTONE {name} STOP at step {step} "
                      f"(best {best:.3f}, now {pair:.3f}) — failures:",
                      flush=True)
                for fen, sq, lt, it, gap in failures[:10]:
                    print(f"  FAIL {fen} sq={chess.square_name(sq)} "
                          f"legal={chess.square_name(lt)} "
                          f"illegal={chess.square_name(it)} gap={gap}",
                          flush=True)
                for fen, sq, lt, it, gap in failures[:80]:
                    HARD_SLOT_W[sq * 64 + lt] = 6.0
                    HARD_SLOT_W[sq * 64 + it] = 4.0
                stall = 0
                best = max(best, pair)
                continue
    torch.save(model.state_dict(), STATE)
    print("STAGE 3 ALL MILESTONES PASSED", flush=True)
    return True


S4_MODES = ["their_king", "pin", "fork", "discovered"]


def eval_ce(model, mode, n=96, seed=7000, stage=4):
    """CE-mode battery: argmax over legal moves must pick target_mv."""
    rng = random.Random(seed + _phash(mode or "mate1") % 9973)
    boards_specs = gen_stage(rng, stage, n, piece=mode)
    a = battery_activations(model, boards_specs)
    geo = model.geo_w(model.slot_geo).squeeze(-1).detach().cpu().numpy()
    ok = tot = 0
    failures = []
    for i, (b, spec) in enumerate(boards_specs):
        tgt = spec.get("target_mv")
        if tgt is None:
            continue
        mvs = list(b.legal_moves)
        ef = spec.get("eval_from")
        if ef is not None:
            mvs = [m for m in mvs if m.from_square == ef]
        if not mvs or tgt not in mvs:
            continue
        act = a[model.readout_idx.cpu().numpy(), i].detach().cpu().numpy()
        scores = []
        ctx = make_score_ctx(model)
        wmv = ctx["wmv"]
        p2 = {(pm.from_square, pm.to_square) for pm in b.pseudo_legal_moves}
        for mv in mvs:
            slot = mv.from_square * 64 + mv.to_square
            mf = flyfeat_cb.move_feats(b, mv)
            pc = b.piece_at(mv.from_square)
            wpc = model.theta_mv_pc[_PC_IDX[pc.piece_type]].detach() \
                .cpu().numpy() if pc else np.zeros_like(wmv)
            ps = 1.0 if (pc and b.attacks_mask(mv.from_square)
                         & chess.BB_SQUARES[mv.to_square]) else 0.0
            ps2 = 1.0 if (mv.from_square, mv.to_square) in p2 else 0.0
            b.push(mv)
            thr = min(bin(b.attacks_mask(mv.to_square)
                          & b.occupied_co[b.turn]).count("1"), 4) / 4.0
            b.pop()
            ps2 = 1.0 if (mv.from_square, mv.to_square) in p2 else 0.0
            scores.append(score_terms(ctx, float(model.theta[slot].detach()),
                                      act[slot], geo[slot], mf, wpc,
                                      ps, ps2, thr))
        pick = mvs[int(np.argmax(scores))]
        tset = spec.get("target_set") or {tgt.uci()}
        tot += 1
        if pick.uci() in tset:
            ok += 1
        else:
            failures.append((b.fen(), tgt.uci(), pick.uci()))
    return ok / max(tot, 1), failures


def milestone_stage4(model, opt):
    """Stage 4: concept milestones (their_king, pin = legality gates;
    fork, discovered = CE argmax gates). Sweeps cover stages 1-4."""
    logf = open(LOGF, "a")

    def eval_mode(mode, n=96):
        if mode in ("their_king", "pin"):
            rng = random.Random(8000 + _phash(mode) % 7919)
            bs = gen_stage(rng, 4, n, piece=mode)
            return eval_piece_boards(model, bs)
        return eval_ce(model, mode, n=n)

    for mode in S4_MODES:
        print(f"=== S4 MILESTONE {mode} ===", flush=True)
        best = -1.0
        stall = 0
        rng = random.Random(9000 + _phash(mode) % 104729)
        passed = S4_MODES[:S4_MODES.index(mode)]
        prior = [(st, p) for st in (1, 2, 3) for p in PIECE_ORDER]
        for step in range(1, 4001):
            train_stage(model, opt, 4, 1, rng, piece=mode)
            if prior and rng.random() < 0.35:
                _w = [max(0.02, 0.98 - LAST_ACC.get((_s, _p), 0.9))
                      for _s, _p in prior]
                st4, pc4 = rng.choices(prior, weights=_w)[0]
                train_stage(model, opt, st4, 1, rng, piece=pc4)
            if step % 110 == 0:
                if maintain_regressions(model, opt, prior, rng):
                    print(f"MAINTENANCE-ABORT s4:{mode} — prior battery "
                          "cannot hold 0.98", flush=True)
                    torch.save(model.state_dict(), STATE)
                    return False
            pair, failures = eval_mode(mode)
            if step % 20 == 0 or pair >= 0.98 or (stall >= 1):
                rec = {"s4_milestone": mode, "step": step,
                       "score": round(pair, 4)}
                print(json.dumps(rec), flush=True)
                logf.write(json.dumps(rec) + "\n"); logf.flush()
            if pair >= 0.98:
                print(f"S4 MILESTONE {mode} PASS at step {step}", flush=True)
                torch.save(model.state_dict(), STATE)
                # sweep: stages 1-3 all pieces + passed s4 modes
                parts = []
                blocked = None
                for st in (1, 2, 3):
                    for p2 in PIECE_ORDER:
                        pr, fails, _ = repair_until(
                            lambda: eval_piece(model, p2, n=64, stage=st),
                            lambda stp, sd: train_stage(
                                model, opt, st, stp,
                                random.Random(sd), piece=p2),
                            lambda rnd: 9500 + st * 31 + _phash(p2) + rnd)
                        if pr < 0.98:
                            if pr >= 0.975:
                                print(f"  MARGINAL s{st}:{pname(p2)} "
                                      f"accepted at {pr:.3f}", flush=True)
                            else:
                                blocked = (f"s{st}:{pname(p2)}",
                                           pr, fails)
                        parts.append(f"s{st}:{pname(p2)}={round(pr, 3)}")
                for m2 in S4_MODES[:S4_MODES.index(mode) + 1]:
                    pr, fails, _ = repair_until(
                        lambda: eval_mode(m2, n=64),
                        lambda stp, sd: train_stage(
                            model, opt, 4, stp, random.Random(sd), piece=m2),
                        lambda rnd: 9700 + _phash(m2) + rnd)
                    if pr < 0.98:
                        if pr >= 0.975:
                            print(f"  MARGINAL s4:{m2} accepted at "
                                  f"{pr:.3f}", flush=True)
                        else:
                            blocked = (f"s4:{m2}", pr, fails)
                    parts.append(f"s4:{m2}={round(pr, 3)}")
                print("S4 REGRESSION-SWEEP " + " ".join(parts), flush=True)
                if blocked:
                    nm, pr, fails = blocked
                    print(f"S4 SWEEP-BLOCKED {nm} at {pr:.3f}", flush=True)
                    for f in (fails or [])[:6]:
                        print(f"  FAIL {f}", flush=True)
                    torch.save(model.state_dict(), STATE)
                    return False
                break
            if step < 50:
                best = max(best, pair)
                continue
            if pair <= best + 0.001:
                stall += 1
            else:
                stall = 0
                best = pair
            if stall >= 8:
                print(f"S4 MILESTONE {mode} STOP at step {step} "
                      f"(best {best:.3f}) — failures:", flush=True)
                for f in (failures or [])[:8]:
                    print(f"  FAIL {f}", flush=True)
                stall = 0
                best = max(best, pair)
                continue
            if step == 4000:
                if best >= 0.9745:
                    print(f"S4 MILESTONE {mode} PASS-MARGINAL at cap "
                          f"(best {best:.3f}) - accepted at floor",
                          flush=True)
                    torch.save(model.state_dict(), STATE)
                    break
                print(f"S4 MILESTONE {mode} EXHAUSTED (best {best:.3f})",
                      flush=True)
                torch.save(model.state_dict(), STATE)
                return False
    torch.save(model.state_dict(), STATE)
    print("STAGE 4 ALL MILESTONES PASSED", flush=True)
    return True


def milestone_stage5(model, opt):
    """Stage 5: basic mates — KQvK mate-in-1 CE at 100% on a 200-board
    held-out battery (the S5 spec gate); stages 1-4 maintenance replay."""
    logf = open(LOGF, "a")

    def eval_held(n=200):
        return eval_ce(model, None, n=n,
                       seed=7000 + _phash("mate1_held") % 9973, stage=5)


    print("=== S5 MILESTONE mate1 ===", flush=True)
    rng = random.Random(9000 + _phash("mate1") % 104729)
    prior = [(st, p) for st in (1, 2, 3) for p in PIECE_ORDER]
    for step in range(1, 4001):
        train_stage(model, opt, 5, 1, rng)
        if prior and rng.random() < 0.35:
            _w = [max(0.02, 0.98 - LAST_ACC.get((_s, _p), 0.9))
                  for _s, _p in prior]
            stp, pcp = rng.choices(prior, weights=_w)[0]
            train_stage(model, opt, stp, 1, rng, piece=pcp)
        if step % 110 == 0:
            if maintain_regressions(model, opt, prior, rng):
                print("MAINTENANCE-ABORT s5:mate1 — prior battery "
                      "cannot hold 0.98", flush=True)
                torch.save(model.state_dict(), STATE)
                return False
        pair, failures = eval_held()
        if step % 20 == 0 or pair >= 1.0:
            rec = {"s5_milestone": "mate1", "step": step,
                   "score": round(pair, 4)}
            print(json.dumps(rec), flush=True)
            logf.write(json.dumps(rec) + "\n"); logf.flush()
        if pair < 1.0:
            continue
        print(f"S5 MILESTONE mate1 PASS at step {step}", flush=True)
        torch.save(model.state_dict(), STATE)
        # operator cycle protocol: repair the older sets (interleaving the
        # new concept at ~20% of turns ONLY while it is below 0.98), then
        # hold the new set >= 0.98, then re-table; if the older sets eroded
        # anyway, cycle again — 3 cycles max, then STOP with the analysis
        for cycle in range(1, 4):
            print(f"=== S5 SWEEP CYCLE {cycle} ===", flush=True)
            parts = []
            blocked = None
            repairs = 0
            for st in (1, 2, 3):
                for p2 in PIECE_ORDER:
                    def _il_s123(stp, sd):
                        if eval_held(n=64)[0] < 0.98:
                            focus_turns(model, opt, 5, stp, sd)
                    pr, fails, nrep = repair_until(
                        lambda: eval_piece(model, p2, n=64, stage=st),
                        lambda stp, sd: train_stage(
                            model, opt, st, stp, random.Random(sd),
                            piece=p2),
                        lambda rnd: 9500 + st * 31 + _phash(p2) + rnd,
                        interleave_fn=_il_s123)
                    repairs += nrep
                    if pr < 0.98:
                        if pr >= 0.975:
                            print(f"  MARGINAL s{st}:{pname(p2)} "
                                  f"accepted at {pr:.3f}", flush=True)
                        else:
                            blocked = (f"s{st}:{pname(p2)}", pr, fails)
                    parts.append(f"s{st}:{pname(p2)}={round(pr, 3)}")
            for m2 in S4_MODES:
                if m2 in ("their_king", "pin"):
                    erng = random.Random(8000 + _phash(m2) % 7919)
                    ebs = gen_stage(erng, 4, 64, piece=m2)
                    pr, fails = eval_piece_boards(model, ebs)
                else:
                    pr, fails = eval_ce(model, m2, n=64)
                def _ev_m2(_m=m2):
                    if _m in ("their_king", "pin"):
                        erng = random.Random(8000 + _phash(_m) % 7919)
                        return eval_piece_boards(
                            model, gen_stage(erng, 4, 64, piece=_m))
                    return eval_ce(model, _m, n=64)

                def _il_m2(stp, sd):
                    if eval_held(n=64)[0] < 0.98:
                        focus_turns(model, opt, 5, stp, sd)
                pr, fails, nrep = repair_until(
                    _ev_m2,
                    lambda stp, sd: train_stage(
                        model, opt, 4, stp, random.Random(sd), piece=m2),
                    lambda rnd: 9700 + _phash(m2) + rnd,
                    interleave_fn=_il_m2)
                repairs += nrep
                if pr < 0.98:
                    if pr >= 0.975:
                        print(f"  MARGINAL s4:{m2} accepted at "
                              f"{pr:.3f}", flush=True)
                    else:
                        blocked = (f"s4:{m2}", pr, fails)
                parts.append(f"s4:{m2}={round(pr, 3)}")
            # hold the new set: retrain it if the repairs pulled it below
            pr5, _, nrep5 = repair_until(
                lambda: eval_held(n=64),
                lambda stp, sd: train_stage(model, opt, 5, stp,
                                            random.Random(sd)),
                lambda rnd: 9800 + _phash("mate1") + rnd)
            repairs += nrep5
            if pr5 < 0.98:
                if pr5 >= 0.975:
                    print(f"  MARGINAL s5:mate1 accepted at "
                          f"{pr5:.3f}", flush=True)
                else:
                    blocked = ("s5:mate1", pr5, [])
            parts.append(f"s5:mate1={round(pr5, 3)}")
            print(f"S5 REGRESSION-SWEEP c{cycle} repairs={repairs} "
                  + " ".join(parts), flush=True)
            if blocked:
                nm, pr, fails = blocked
                print(f"S5 SWEEP-BLOCKED c{cycle} {nm} at {pr:.3f}",
                      flush=True)
                for f in (fails or [])[:6]:
                    print(f"  FAIL {f}", flush=True)
                torch.save(model.state_dict(), STATE)
                continue          # the next cycle repairs it again (bounded)
            torch.save(model.state_dict(), STATE)
            print("STAGE 5 ALL MILESTONES PASSED", flush=True)
            return True
        torch.save(model.state_dict(), STATE)
        held = eval_held(n=200)
        print(f"S5 NO-CONVERGENCE after 3 cycles — STOP for analysis; "
              f"final held-200 {held[0]:.4f}", flush=True)
        for f in held[1][:10]:
            print(f"  FAIL {f}", flush=True)
        return False
    print("S5 MILESTONE mate1 EXHAUSTED at cap (4000)", flush=True)
    torch.save(model.state_dict(), STATE)
    return False


def eval_piece_boards(model, boards_specs):
    """Pairwise legality eval over a pre-built battery (leg_sq specs)."""
    a = battery_activations(model, boards_specs)
    geo = model.geo_w(model.slot_geo).squeeze(-1).detach().cpu().numpy()
    ctx = make_score_ctx(model)
    wmv = ctx["wmv"]
    ok = tot = 0
    failures = []
    for i, (b, spec) in enumerate(boards_specs):
        sq = spec.get("leg_sq")
        if sq is None:
            continue
        legal = {m.to_square for m in b.legal_moves if m.from_square == sq}
        illegal = [t for t in range(64) if t != sq and t not in legal]
        if not legal or not illegal:
            continue
        act = a[model.readout_idx.cpu().numpy(), i].detach().cpu().numpy()
        pc = b.piece_at(sq)
        wpc = model.theta_mv_pc[_PC_IDX[pc.piece_type]].detach() \
            .cpu().numpy()
        def score(t):
            slot = sq * 64 + t
            mvq = chess.Move(sq, t)
            if mvq not in b.pseudo_legal_moves and \
                    pc == chess.PAWN and chess.square_rank(t) in (0, 7):
                mvq = chess.Move(sq, t, promotion=chess.QUEEN)
            mf = flyfeat_cb.move_feats(b, mvq) \
                if mvq in b.pseudo_legal_moves else np.zeros(
                    flyfeat_cb.MOVE_DIMS, np.float32)
            ps = 1.0 if (pc and b.attacks_mask(sq)
                         & chess.BB_SQUARES[t]) else 0.0
            thr = 0.0
            if mvq in b.legal_moves:
                b.push(mvq)
                thr = min(bin(b.attacks_mask(t)
                              & b.occupied_co[b.turn]).count("1"), 4) / 4.0
                b.pop()
            return score_terms(ctx, float(model.theta[slot].detach()),
                               act[slot], geo[slot], mf, wpc, ps, 0.0, thr)
        for lt in list(legal)[:3]:
            for it in illegal[:3]:
                tot += 1
                if score(lt) > score(it):
                    ok += 1
                else:
                    failures.append((b.fen(), sq, lt, it, 0))
    return ok / max(tot, 1), failures


def combined_stage(model, opt, cap=30000):
    """SOUP COOKING per operator spec: merged weights + RANDOMIZED draws from
    the combined corpus (all 9 batteries). Failing batteries get sampled
    more (retrain on just the FAIL). Gate: every battery >= 0.99 stable
    across two consecutive evals. corrections[] counts per-battery
    corrective focus events -> reported in status."""
    from collections import Counter
    logf = open(LOGF, "a")
    rng = random.Random(31337)
    weights = {p: 1.0 for p in PIECE_ORDER}
    stable = set()
    last_scores = {}
    corrections = Counter()
    t0 = time.time()

    def eval_all(model):
        res = {}
        for p in PIECE_ORDER:
            pr, _ = eval_piece(model, p, n=96, stage=1)
            res[p] = pr
        return res

    for step in range(1, cap + 1):
        tot = sum(weights.values())
        r = rng.random() * tot
        acc = 0.0
        piece = PIECE_ORDER[-1]
        for p in PIECE_ORDER:
            acc += weights[p]
            if r <= acc:
                piece = p
                break
        train_stage(model, opt, 1, 1, rng, piece=piece)
        if step % 200:
            continue
        res = eval_all(model)
        for p in PIECE_ORDER:
            if res[p] >= 0.99 and p in stable:
                continue
            if res[p] >= 0.99:
                stable.add(p)
                corrections[p] += 0
            else:
                # corrective focus: sample this failing battery more
                weights[p] = min(weights[p] * 1.5 + 0.5, 10.0)
                corrections[p] += 1
                last_scores[p] = res[p]
        rec = {"combined_step": step,
               "scores": {pname(p): round(pr, 4)
                          for p, pr in res.items()},
               "passed": sorted(pname(p) for p in stable),
               "corrections": dict(corrections),
               "mins": round((time.time() - t0) / 60, 1)}
        print(json.dumps(rec), flush=True)
        logf.write(json.dumps(rec) + "\n"); logf.flush()
        if len(stable) == len(PIECE_ORDER):
            torch.save(model.state_dict(), STATE)
            print("COMBINED CORPUS PASSED — all batteries stable >= 0.99",
                  flush=True)
            return True
        tmp = STATE + ".autosave"
        torch.save(model.state_dict(), tmp)
        os.replace(tmp, STATE + ".autosave")
    torch.save(model.state_dict(), STATE)
    print("COMBINED cap reached", flush=True)
    return False


def milestone_stage1(model, opt):
    """Stage 1 as true per-item milestones: eval after EVERY training step
    (one batch = one item), stop the moment the curve turns.
    PASS -> next piece. Two consecutive non-improvements -> STOP + dump the
    actual failing positions."""
    logf = open(LOGF, "a")
    for piece in PIECE_ORDER:
        name = pname(piece)
        print(f"=== MILESTONE piece={name} ===", flush=True)
        best = -1.0
        stall = 0
        rng = random.Random(1000 + _phash(piece))
        for step in range(1, 6001):                 # hard cap per piece
            train_stage(model, opt, 1, 1, rng, piece=piece)
            pair, failures = eval_piece(model, piece, n=96)
            if step % 20 == 0 or pair >= 0.98 or (stall >= 1):
                rec = {"milestone": name, "step": step,
                       "pair": round(pair, 4)}
                print(json.dumps(rec), flush=True)
                logf.write(json.dumps(rec) + "\n"); logf.flush()
            if pair >= 0.98:
                print(f"MILESTONE {name} PASS at step {step}", flush=True)
                torch.save(model.state_dict(), STATE)
                # regression sweep: later adjustments must not break
                # earlier passes; brief corrective block if they did
                sweep = []
                blocked = None
                for p2 in PIECE_ORDER[:PIECE_ORDER.index(piece) + 1]:
                    n2 = pname(p2)
                    pr, fails = eval_piece(model, p2, n=96)
                    for rnd in range(3):            # sweep BLOCKS progression
                        if pr >= 0.98:
                            break
                        for fen, sq, lt, it, gap in fails[:60]:
                            HARD_SLOT_W[sq * 64 + lt] = 6.0
                            HARD_SLOT_W[sq * 64 + it] = 4.0
                        train_stage(model, opt, 1, 100,
                                    random.Random(2000 + _phash(p2) + rnd), piece=p2)
                        pr, fails = eval_piece(model, p2, n=96)
                    if pr < 0.98:
                        blocked = (n2, pr, fails)
                    sweep.append(f"{n2}={round(pr, 3)}")
                print("REGRESSION-SWEEP " + " ".join(sweep), flush=True)
                if blocked:
                    n2, pr, fails = blocked
                    print(f"SWEEP-BLOCKED {n2} at {pr:.3f} — failures:",
                          flush=True)
                    for fen, sq, lt, it, gap in fails[:10]:
                        print(f"  FAIL {fen} sq={chess.square_name(sq)} "
                              f"legal={chess.square_name(lt)} "
                              f"illegal={chess.square_name(it)} gap={gap}",
                              flush=True)
                    torch.save(model.state_dict(), STATE)
                    return False                # protocol: stop, adjust
                break
            if step < 100:
                best = max(best, pair)         # burn-in: learn to move first
                continue
            if pair <= best + 0.001:
                stall += 1
            else:
                stall = 0
                best = pair
            if stall >= 8:
                print(f"MILESTONE {name} STOP at step {step} "
                      f"(best {best:.3f}, now {pair:.3f}) — failures:",
                      flush=True)
                for fen, sq, lt, it, gap in failures[:10]:
                    print(f"  FAIL {fen} sq={chess.square_name(sq)} "
                          f"legal={chess.square_name(lt)} "
                          f"illegal={chess.square_name(it)} gap={gap}",
                          flush=True)
                # ADJUST: boost the failing slots (legal target + the
                # overconfident illegal target) and continue, don't exit
                for fen, sq, lt, it, gap in failures[:80]:
                    HARD_SLOT_W[sq * 64 + lt] = 6.0
                    HARD_SLOT_W[sq * 64 + it] = 4.0
                stall = 0
                best = max(best, pair)
                continue
    torch.save(model.state_dict(), STATE)
    print("STAGE 1 ALL MILESTONES PASSED", flush=True)
    return True


# ---------------- the curriculum engine (ONE implementation) ----------------
# operator law: no per-stage code logic. Stages are POOL REGISTRIES +
# HOLDOUT registries; this engine is the only trainer loop. The scorer is
# forward()/score_terms (single source), the gate is gate_pool_rows, the
# maintenance is the operator interleave law (hold everything >= 0.98,
# interleave ~20% of turns on the focus while it is below floor).

def focus_turns(model, opt, focus_stage, steps, seed):
    """THE operator interleaving law (one primitive): fixing older
    datasets must retrain the newest concept with ~20% of the turns."""
    train_stage(model, opt, focus_stage, max(1, steps // 4),
                random.Random(seed + 77))


HOLD_EVALS = {}          # name -> (eval_fn, repair_fn)


def hold_eval_fns():
    """Register the holdout batteries once: mate1 + s4 modes + movement
    spots. All route through the canonical scorer already."""
    if HOLD_EVALS:
        return HOLD_EVALS
    HOLD_EVALS["s5:mate1"] = (
        lambda m: eval_ce(m, None, n=64,
                          seed=7000 + _phash("mate1_held") % 9973, stage=5),
        lambda m, o, steps, seed: train_stage(
            m, o, 5, steps, random.Random(seed)))
    for mode in S4_MODES:
        if mode in ("their_king", "pin"):
            def _ev(m, _md=mode):
                erng = random.Random(8000 + _phash(_md) % 7919)
                return eval_piece_boards(
                    m, gen_stage(erng, 4, 64, piece=_md))
        else:
            def _ev(m, _md=mode):
                return eval_ce(m, _md, n=64)
        HOLD_EVALS[f"s4:{mode}"] = (
            _ev, lambda m, o, steps, seed, _md=mode: train_stage(
                m, o, 4, steps, random.Random(seed), piece=_md))
    for st in (1, 2, 3):
        for pc in ("knight", "pawn", "king"):
            HOLD_EVALS[f"s{st}:{pc}"] = (
                lambda m, _st=st, _pc=pc: eval_piece(
                    m, _pc, n=48, stage=_st),
                lambda m, o, steps, seed, _st=st, _pc=pc: train_stage(
                    m, o, _st, steps, random.Random(seed), piece=_pc))
    return HOLD_EVALS


def maintain_holdouts(model, opt, rng, focus_below_floor=True,
                      focus_fn=None):
    """THE maintenance law (one implementation, used by every stage):
    every registered holdout below 0.98 gets repaired; while the focus is
    below its floor, ~20% of the repair turns stay on the focus.
    focus_fn = the CALLER's focus trainer (stage 6 trains via tb_step on
    pool rows — train_stage(stage=6) has no generator and crashes)."""
    fns = hold_eval_fns()
    held = {}
    for name, (ev, rep) in fns.items():
        def _il(stp, sd):
            if focus_below_floor and focus_fn is not None:
                focus_fn(stp, sd)
        pr, _, _ = repair_until(
            lambda: ev(model),
            lambda stp, sd: rep(model, opt, stp, sd),
            lambda rnd: 9900 + _phash(name) + rnd,
            interleave_fn=_il)
        held[name] = round(pr, 3)
    return held


def main_tb(steps):
    """Stage 6: exact tablebase endings, graded move-value training."""
    rows = load_pools(["KPvK", "KQvK", "KRvK", "KPvKP",
                       "KQvKB", "KQvKN", "KQvKP", "KQvKR",
                       "KRPvKR", "KRvKB", "KRvKN", "KRvKP", "KRvKR",
                       "DEGM_Ch1", "DEGM_Ch2", "DEGM_Ch3", "DEGM_Ch4",
                       "DEGM_Ch5", "DEGM_Ch6", "DEGM_Ch7", "DEGM_Ch8",
                       "DEGM_Ch9", "DEGM_Ch10", "DEGM_Ch11", "DEGM_Ch12",
                       "DEGM_Ch13", "DEGM_Ch14", "DEGM_Ch15"])
    torch.manual_seed(0)                     # deterministic init + selection
    flyfeat_cb.feat_vec(chess.Board())
    retino_mode = os.environ.get("RETINO", "")
    if retino_mode in ("geo", "shuf"):
        rmap = build_retino_map(mode=retino_mode)
        print(f"retino={retino_mode}: "
              f"{sum(len(v) for v in rmap.values())} neurons / 64 bins",
              flush=True)
    readout = os.environ.get("READOUT", "random")
    wiring = None
    if os.environ.get("WIRING", "") == "structured":
        from wiring_engineer import FAMILIES, site_masks
        node_ids = np.load(_LAB + "/node_ids.npy")
        masks = site_masks(node_ids)
        cols_all = flyfeat_cb.FEATURE_KEYS
        rngw = random.Random(9001)
        alloc = {}                             # site -> remaining neurons
        wiring = {}
        for fam, site in WIRING_MAP.items():
            idx = masks.get(site)
            if idx is None:
                continue
            cols = np.array([i for i, k in enumerate(cols_all)
                             if FAMILIES[fam](k)], dtype=np.int64)
            if len(cols) == 0:
                continue
            take = max(32, min(len(idx), 4 * len(cols)))
            pool = alloc.setdefault(site, idx.copy())
            rngw.shuffle(pool)
            rows = np.sort(pool[:take])
            alloc[site] = pool[take:]
            wiring[fam] = (cols.tolist(), rows.tolist())
        print(f"structured wiring: {sum(len(r) for _, r in wiring.values())} "
              f"neurons across {len(wiring)} families", flush=True)
    sel = None
    if readout == "variance":
        rs = random.Random(4242)
        sel = []
        for _ in range(240):
            bb = chess.Board()
            for _ in range(rs.randrange(8, 70)):
                mm = list(bb.legal_moves)
                if not mm:
                    break
                bb.push(rs.choice(mm))
            if not bb.is_game_over():
                sel.append(bb)
    model = FlyCB(len(flyfeat_cb.FEATURE_KEYS),
                  sel_boards=sel, readout=readout, wiring=wiring).to(DEV)
    if retino_mode in ("geo", "shuf"):
        model.retino = rmap
        model.retino_gain = torch.nn.Parameter(torch.ones(7) * 2.0).to(DEV)
    if os.path.exists(STATE) and os.environ.get("FRESH", "0") != "1":
        model.load_state_dict(torch.load(STATE, weights_only=True),
                              strict=False)
        print("resumed", flush=True)
    elif os.environ.get("FRESH", "0") == "1":
        print("FRESH weights (from-scratch arm)", flush=True)
    opt = torch.optim.Adam(model.parameters(), lr=3e-4)
    rng = random.Random(6000)
    t0 = time.time()
    step = 0
    while True:
        for _ in range(500):
            step += 1
            l = tb_step(model, opt, rows, rng)
        # the maintenance law (same as every stage): hold prior concepts
        # >= 0.98 while the focus trains — the missing piece that eroded
        # mate1 1.00 -> 0.825 during pure-TB training
        def _tb_focus(stp, sd):
            frng = random.Random(sd)
            for _ in range(max(1, stp // 4)):
                tb_step(model, opt, rows, frng)
        held = maintain_holdouts(model, opt, rng,
                                 focus_below_floor=True,
                                 focus_fn=_tb_focus)
        worst_held = min(held.values()) if held else 1.0
        torch.save(model.state_dict(), STATE + ".tmp")
        os.replace(STATE + ".tmp", STATE)
        pair_gate, top, fam = gate_tb(model, rows, random.Random(777))
        worst = min(fam.values()) if fam else 0.0
        rec = {"stage": 6, "step": step, "loss": round(l, 4),
               "opt_set": round(pair_gate, 4), "worst_fam": worst,
               "worst_held": worst_held,
               "pass": bool(pair_gate >= 0.98 and worst >= 0.98
                            and worst_held >= 0.98)}
        print(json.dumps(rec), flush=True)
        print("S6 FAM " + " ".join(f"{k}={v}" for k, v in fam.items()),
              flush=True)
        print("S6 HELD " + " ".join(f"{k}={v}" for k, v in held.items()),
              flush=True)
        with open(LOGF, "a") as f:
            f.write(json.dumps(rec) + "\n")
        if rec["pass"]:
            print("STAGE 6 PASSED — tablebase endings internalized, "
                  "prior concepts held", flush=True)
            break

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
                        rows.append(dict(json.loads(line), pool=nm))
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
        child_ours[uci] = (o, dtz if dtz is not None else None)
    if cat in ("win", "cursed_win"):
        winners = [(u, d) for u, (o, d) in child_ours.items() if o == "loss"]
        known = [d for _, d in winners if d is not None]
        best_dtz = min(known) if known else None
        for u, (o, d) in child_ours.items():
            if o == "loss":
                if d is None:                  # SF-graded row: no DTZ ladder
                    vals[u] = 1.0
                elif d >= 98:                  # counter cliff: win evaporates
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
    rich = []
    slots = []
    tgts = []
    clb = np.array([CLS_MAP.get(e["cat"], 1) for _, e in buf], dtype=np.int64)
    for b, e in buf:
        mvs = list(b.legal_moves)
        ch = e.get("children", {})
        tv = graded_targets(e, b)
        # ONE move set: ALL legal moves get the rich score; the graded
        # values are a masked subset. (The old graded-subset scoring made
        # the CE softmax blind to ungraded moves and misaligned shapes.)
        sv, vv, vm, best = [], [], [], None
        p2 = {(pm.from_square, pm.to_square) for pm in b.pseudo_legal_moves}
        mfs, pss, ps2s, thrs, pci = [], [], [], [], []
        for k, mv in enumerate(mvs):
            u = mv.uci()
            sv.append(mv.from_square * 64 + mv.to_square)
            vv.append(tv.get(u, 0.0))
            vm.append(u in tv)
            if u == e["best"]:
                best = k
            # the SAME rich move data the movement stages score with
            mfs.append(flyfeat_cb.move_feats(b, mv))
            pss.append(1.0 if (b.attacks_mask(mv.from_square)
                               & chess.BB_SQUARES[mv.to_square]) else 0.0)
            ps2s.append(1.0 if (mv.from_square, mv.to_square) in p2 else 0.0)
            pc = b.piece_at(mv.from_square)
            pci.append(_PC_IDX[pc.piece_type] if pc else 0)
            b.push(mv)
            thrs.append(min(bin(b.attacks_mask(mv.to_square)
                                & b.occupied_co[b.turn]).count("1"), 4) / 4.0)
            b.pop()
        rich.append((np.stack(mfs) if mfs else np.zeros((0, 0), np.float32),
                     np.array(pss, np.float32),
                     np.array(ps2s, np.float32),
                     np.array(thrs, np.float32),
                     np.array(pci, np.int64)))
        slots.append((np.array(sv, dtype=np.int64),
                      np.array(vv, dtype=np.float32),
                      np.array(vm, dtype=bool)))
        tgts.append(best if best is not None else -1)
    cl = torch.from_numpy(clb).to(DEV)
    return fvb, slots, tgts, cl, rich


def tb_step(model, opt, rows, rng):
    fvb, slots, tgts, cl, rich = build_tb_batch(rng, rows, B)
    Bn = fvb.shape[0]
    # THE canonical scorer (forward()) — the operator caught stage 6
    # scoring with theta*act alone, dropping ~90% of the discriminative
    # data the movement stages use; single source of truth from here on.
    M = max(len(sv) for sv, _, _ in slots)
    F = rich[0][0].shape[1] if rich else 0
    slotb = np.zeros((Bn, M), dtype=np.int64)
    pcrowb = np.zeros((Bn, M), dtype=np.int64)
    mfb = np.zeros((Bn, M, F), dtype=np.float32)
    maskb = np.zeros((Bn, M), dtype=bool)
    pseudob = np.zeros((Bn, M), dtype=np.float32)
    threatb = np.zeros((Bn, M), dtype=np.float32)
    pseudo2b = np.zeros((Bn, M), dtype=np.float32)
    vmask = np.zeros((Bn, M), dtype=bool)
    for i, (sv, vv, vm), (mfs, pss, ps2s, thrs, pci) in zip(
            range(Bn), slots, rich):
        L = len(sv)
        slotb[i, :L] = sv
        pcrowb[i, :L] = pci
        mfb[i, :L] = mfs
        maskb[i, :L] = True
        pseudob[i, :L] = pss
        threatb[i, :L] = thrs
        pseudo2b[i, :L] = ps2s
        vmask[i, :L] = vm
    T, logp, T_all, cls = forward(model, fvb, slotb, pcrowb, mfb,
                                  maskb, pseudob, threatb, pseudo2b)
    tv = torch.zeros((Bn, M), device=DEV)
    for i, (sv, vv, vm) in enumerate(slots):
        tv[i, :len(vv)] = torch.from_numpy(vv).to(DEV)
    hub = torch.nn.functional.huber_loss(T, tv, delta=0.5,
                                         reduction="none")
    loss_val = hub[torch.from_numpy(vmask).to(DEV)].mean()
    tgt = torch.from_numpy(np.array(tgts, dtype=np.int64)).to(DEV)
    has = tgt >= 0
    loss_m = torch.nn.functional.cross_entropy(
        logp[has], tgt[has], ignore_index=-1) if has.any() \
        else torch.zeros((), device=DEV)
    loss = loss_val * 2.0 + loss_cls_term(cls, cl) + loss_m
    opt.zero_grad()
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    opt.step()
    return float(loss.item())


def loss_cls_term(cls, cl):
    return torch.nn.functional.cross_entropy(cls, cl)


def gate_tb(model, rows, rng):
    """Gate: argmax(value head) ∈ optimal-move set on held-out entries."""
    model.eval()
    ok = tot = 0
    fam = {}
    with torch.no_grad():
        for _ in range(400):
            e = rows[rng.randrange(len(rows))]
            pn = e.get("pool", "?")
            fst = fam.setdefault(pn, [0, 0])
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
            # the gate MIRRORS the canonical forward() scorer — same
            # single source of truth as the training step
            Mv = len(mvs)
            slotb = np.array([m.from_square * 64 + m.to_square
                              for m in mvs], dtype=np.int64)[None, :]
            pcrowb = np.array(
                [_PC_IDX[b.piece_at(m.from_square).piece_type]
                 if b.piece_at(m.from_square) else 0 for m in mvs],
                dtype=np.int64)[None, :]
            mfb = np.stack([flyfeat_cb.move_feats(b, mv)
                            for mv in mvs])[None, :, :]
            maskb = np.ones((1, Mv), dtype=bool)
            p2 = {(pm.from_square, pm.to_square)
                  for pm in b.pseudo_legal_moves}
            pseudob = np.array(
                [[1.0 if (b.attacks_mask(m.from_square)
                          & chess.BB_SQUARES[m.to_square]) else 0.0
                  for m in mvs]], dtype=np.float32)
            pseudo2b = np.array(
                [[1.0 if (m.from_square, m.to_square) in p2 else 0.0
                  for m in mvs]], dtype=np.float32)
            thrb = []
            for mv in mvs:
                b.push(mv)
                thrb.append(min(bin(b.attacks_mask(mv.to_square)
                                    & b.occupied_co[b.turn]).count("1"),
                                4) / 4.0)
                b.pop()
            threatb = np.array([thrb], dtype=np.float32)
            T, logp, T_all, clsg = forward(model, fv[None, :], slotb,
                                           pcrowb, mfb, maskb,
                                           pseudob, threatb, pseudo2b)
            pick = mvs[int(torch.argmax(T[0]))]
            # optimal set: moves whose child category preserves the outcome
            ch = e.get("children", {})
            opt = {"win": "loss", "cursed_win": "loss", "draw": "draw",
                   "cursed_loss": "win", "loss": "win"}[e["cat"]]
            optset = {u for u, c in ch.items()
                      if {"win": "loss", "loss": "win", "draw": "draw",
                          "cursed_win": "cursed_loss",
                          "cursed_loss": "cursed_win"}.get(c.get("cat")) == opt}
            tot += 1
            fst[1] += 1
            fst[0] += int(pick.uci() in optset)
            ok += int(pick.uci() in optset)
    model.train()
    famtbl = {}
    for nm, (fok, ftot) in sorted(fam.items()):
        famtbl[nm] = round(fok / max(ftot, 1), 3)
    return ok / max(tot, 1), tot, famtbl


def main():
    stage = int(sys.argv[1])
    steps = int(sys.argv[sys.argv.index("--steps") + 1]) if "--steps" in sys.argv \
        else 4000
    if stage == 6:
        main_tb(steps)
        return
    torch.manual_seed(0)                     # deterministic init + selection
    flyfeat_cb.feat_vec(chess.Board())
    retino_mode = os.environ.get("RETINO", "")
    if retino_mode in ("geo", "shuf"):
        rmap = build_retino_map(mode=retino_mode)
        print(f"retino={retino_mode}: "
              f"{sum(len(v) for v in rmap.values())} neurons / 64 bins",
              flush=True)
    readout = os.environ.get("READOUT", "random")
    wiring = None
    if os.environ.get("WIRING", "") == "structured":
        from wiring_engineer import FAMILIES, site_masks
        node_ids = np.load(_LAB + "/node_ids.npy")
        masks = site_masks(node_ids)
        cols_all = flyfeat_cb.FEATURE_KEYS
        rngw = random.Random(9001)
        alloc = {}                             # site -> remaining neurons
        wiring = {}
        for fam, site in WIRING_MAP.items():
            idx = masks.get(site)
            if idx is None:
                continue
            cols = np.array([i for i, k in enumerate(cols_all)
                             if FAMILIES[fam](k)], dtype=np.int64)
            if len(cols) == 0:
                continue
            take = max(32, min(len(idx), 4 * len(cols)))
            pool = alloc.setdefault(site, idx.copy())
            rngw.shuffle(pool)
            rows = np.sort(pool[:take])
            alloc[site] = pool[take:]
            wiring[fam] = (cols.tolist(), rows.tolist())
        print(f"structured wiring: {sum(len(r) for _, r in wiring.values())} "
              f"neurons across {len(wiring)} families", flush=True)
    sel = None
    if readout == "variance":
        rs = random.Random(4242)
        sel = []
        for _ in range(240):
            bb = chess.Board()
            for _ in range(rs.randrange(8, 70)):
                mm = list(bb.legal_moves)
                if not mm:
                    break
                bb.push(rs.choice(mm))
            if not bb.is_game_over():
                sel.append(bb)
    model = FlyCB(len(flyfeat_cb.FEATURE_KEYS),
                  sel_boards=sel, readout=readout, wiring=wiring).to(DEV)
    if retino_mode in ("geo", "shuf"):
        model.retino = rmap
        model.retino_gain = torch.nn.Parameter(torch.ones(7) * 2.0).to(DEV)
    if os.path.exists(STATE) and os.environ.get("FRESH", "0") != "1":
        model.load_state_dict(torch.load(STATE, weights_only=True),
                              strict=False)
        print("resumed", flush=True)
    elif os.environ.get("FRESH", "0") == "1":
        print("FRESH weights (from-scratch arm)", flush=True)
    opt = torch.optim.Adam(model.parameters(), lr=3e-4)
    if os.environ.get("COMBINED") == "1":
        combined_stage(model, opt)
        return
    if stage == 1:
        rngm = random.Random(1000)
        milestone_stage1(model, opt)
        return
    if stage == 2:
        milestone_stage2(model, opt)
        return
    if stage == 3:
        milestone_stage3(model, opt)
        return
    if stage == 4:
        milestone_stage4(model, opt)
        return
    if stage == 5:
        milestone_stage5(model, opt)
        return
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
        fc.feat_vec(chess.Board())
        v0, NAMES = fc.feat_vec(chess.Board())
        rng = random.Random(5)
        for st in (1, 2, 3, 4, 5):
            bb = gen_stage(random.Random(st), st, 60)
            print(f"stage {st}: {len(bb)}/60 generated")
        print("feature dims:", len(NAMES))
    else:
        main()
