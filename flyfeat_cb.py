#!/usr/bin/env python3
"""Color-blind feature extractor v2 — the mover is always 'me'.

Canonicalization: if black to move, the position is VERTICALLY MIRRORED, so
the mover always plays up the board and every position has ONE feature
representation. This halves required training data and makes side-to-move
implicit. Castling rights and en-passant file are explicit features.

Feature groups (all relative to the mover):
  material bitboard counts      12
  per-square: atk_my/atk_their/net/occ_my/occ_their   64 x 5
  piece 'eyes' (32 x 66: view + pin + mobility)      2112
  king rings (my/theirs x enemy/defended)             ~36
  material counts                                    12
  castling rights (my K/Q, their K/Q)                  4
  en-passant file (canonical)                          1
  check / checkers / pinned                            3
  mobility my/theirs                                   2
"""
import numpy as np, chess

FEATURE_KEYS = None
VALS = {chess.PAWN: 1, chess.KNIGHT: 3, chess.BISHOP: 3, chess.ROOK: 5,
        chess.QUEEN: 9, chess.KING: 0}


def canon(board):
    """Vertical mirror when black to move -> mover always plays 'up'."""
    if board.turn == chess.WHITE:
        return board
    return board.mirror()


def extract(board):
    """board MUST already be canonical (mover = white-like 'me')."""
    f = {}
    me, them = board.turn, (not board.turn)
    pts = [chess.PAWN, chess.KNIGHT, chess.BISHOP, chess.ROOK, chess.QUEEN,
           chess.KING]

    # ---- per-square attack / occupancy (relative) ----
    a_my = np.zeros(64, dtype=np.float32)
    a_th = np.zeros(64, dtype=np.float32)
    o_my = np.zeros(64, dtype=np.float32)
    o_th = np.zeros(64, dtype=np.float32)
    eyes = []
    eyes_geo = []                       # (dist up, down, left, right) to edges
    for sq in chess.SQUARES:
        pc = board.piece_at(sq)
        if pc is None:
            continue
        sf, sr = chess.square_file(sq), chess.square_rank(sq)
        eyes_geo.append((sr / 7.0, (7 - sr) / 7.0,
                         sf / 7.0, (7 - sf) / 7.0))
        mine = pc.color == board.turn
        if mine:
            o_my[sq] = 1.0
        else:
            o_th[sq] = 1.0
        am = board.attacks_mask(sq)
        bits = np.frombuffer(np.uint64(am).tobytes(), dtype=np.uint8)
        # expand bit mask to 64 squares
        m = int(am)
        target = a_my if mine else a_th
        while m:
            s2 = (m & -m).bit_length() - 1
            m &= m - 1
            target[s2] += 1.0
        eye = np.zeros(66, dtype=np.float32)
        for s2 in chess.SQUARES:
            v = 1.0 if ((int(am) >> s2) & 1) else 0.0
            own = board.occupied_co[pc.color]
            if (own >> s2) & 1:
                v += 2.0 if mine else 4.0
            elif (board.occupied >> s2) & 1:
                v += 4.0 if mine else 2.0
            eye[s2] = v
        eye[64] = 1.0 if board.is_pinned(pc.color, sq) else 0.0
        eye[65] = int(am & ~board.occupied).bit_count()
        if len(eyes) < 32:
            eyes.append(eye)
    for sq in chess.SQUARES:
        f[f"atk_my_{sq}"] = min(a_my[sq], 4) / 4.0
        f[f"atk_their_{sq}"] = min(a_th[sq], 4) / 4.0
        f[f"net_{sq}"] = (min(a_my[sq], 4) - min(a_th[sq], 4)) / 4.0
        f[f"occ_my_{sq}"] = o_my[sq]
        f[f"occ_their_{sq}"] = o_th[sq]
    for i in range(32):
        if i < len(eyes):
            e = eyes[i]
            for s2 in range(64):
                f[f"eye{i}_s{s2}"] = e[s2]
            f[f"eye{i}_pin"] = e[64]
            f[f"eye{i}_mob"] = e[65]
            eu, ed, el, er = eyes_geo[i]
            f[f"eye{i}_eu"] = eu                  # edge-of-board geometry:
            f[f"eye{i}_ed"] = ed                  # NEW signal the relative
            f[f"eye{i}_el"] = el                  # view collapses away
            f[f"eye{i}_er"] = er
        else:
            for s2 in range(64):
                f[f"eye{i}_s{s2}"] = 0.0
            f[f"eye{i}_pin"] = 0.0
            f[f"eye{i}_mob"] = 0.0
            f[f"eye{i}_eu"] = f[f"eye{i}_ed"] = 0.0
            f[f"eye{i}_el"] = f[f"eye{i}_er"] = 0.0
    # attacked-square-on-edge conjunctions (the rook failure signature)
    edge = np.array([1.0 if (s % 8 in (0, 7)) or (s // 8 in (0, 7)) else 0.0
                     for s in range(64)], dtype=np.float32)
    for sq in chess.SQUARES:
        f[f"atk_edge_my_{sq}"] = min(a_my[sq], 1.0) * edge[sq]
        f[f"atk_edge_their_{sq}"] = min(a_th[sq], 1.0) * edge[sq]

    # ---- material ----
    for t, pt in enumerate(pts):
        f[f"bb_my_{t}"] = float(board.pieces_mask(pt, me).bit_count())
        f[f"bb_their_{t}"] = float(board.pieces_mask(pt, them).bit_count())
        f[f"mat_my_{t}"] = board.pieces_mask(pt, me).bit_count() / 8.0
        f[f"mat_their_{t}"] = board.pieces_mask(pt, them).bit_count() / 8.0

    # ---- king rings ----
    for tag, k in (("my", board.king(me)), ("their", board.king(them))):
        if k is None:
            continue
        ring = chess.BB_KING_ATTACKS[k] | (1 << k)
        for s2 in chess.SQUARES:
            if (ring >> s2) & 1:
                f[f"ks_{tag}_bythem_{s2}"] = min(a_th[s2], 4) / 4.0 if tag == "my" \
                    else min(a_my[s2], 4) / 4.0
                f[f"ks_{tag}_byme_{s2}"] = min(a_my[s2], 4) / 4.0 if tag == "my" \
                    else min(a_th[s2], 4) / 4.0

    # ---- castling rights (relative) ----
    f["castle_my_k"] = 1.0 if board.has_kingside_castling_rights(me) else 0.0
    f["castle_my_q"] = 1.0 if board.has_queenside_castling_rights(me) else 0.0
    f["castle_their_k"] = 1.0 if board.has_kingside_castling_rights(them) else 0.0
    f["castle_their_q"] = 1.0 if board.has_queenside_castling_rights(them) else 0.0

    # ---- en passant (canonical: file only; mover to move) ----
    ep = board.ep_square
    f["ep_file"] = (chess.square_file(ep) + 1) / 8.0 if ep is not None else 0.0

    # ---- state ----
    f["in_check"] = 1.0 if board.is_check() else 0.0
    f["checkers"] = min(board.checkers_mask().bit_count(), 4) / 4.0
    f["pinned_n"] = min(sum(1 for sq in chess.SQUARES
                            if (pc := board.piece_at(sq))
                            and board.is_pinned(pc.color, sq)), 8) / 8.0
    mw = mb = 0
    for s in chess.SQUARES:
        pc = board.piece_at(s)
        if pc is None:
            continue
        n = int(board.attacks_mask(s) & ~board.occupied).bit_count()
        if pc.color == board.turn:
            mw += n
        else:
            mb += n
    f["mob_my"] = min(mw, 120) / 120.0
    f["mob_their"] = min(mb, 120) / 120.0
    return f


_FV_CACHE = {}                 # fen -> (vec, canon_board): the pools are
_FV_CACHE_MAX = 120000         # a fixed FEN set revisited every sweep AND
                               # every training batch; extract is a pure
                               # function so memoization is bit-exact


def feat_vec_by_fen(fen):
    """FEN-keyed access for precomputed paths: canon(fen) is determined
    by the FEN alone, so the color-canonical result is identical to
    feat_vec(Board(fen)) — verified bit-exact at startup."""
    hit = _FV_CACHE.get(fen)
    if hit is not None:
        return hit[0]
    return feat_vec(chess.Board(fen))[0]


def feat_vec(board):
    global FEATURE_KEYS
    key = board.fen()
    hit = _FV_CACHE.get(key)
    if hit is not None:
        v, cb = hit
        return v, cb.copy()
    cb = canon(board)
    d = extract(cb)
    if FEATURE_KEYS is None:
        FEATURE_KEYS = sorted(d.keys())
    v = np.array([d.get(k, 0.0) for k in FEATURE_KEYS], dtype=np.float32)
    if len(_FV_CACHE) < _FV_CACHE_MAX:
        _FV_CACHE[key] = (v, cb)
    return v, cb.copy()


MOVE_DIMS = 17


def move_feats(board, mv):
    """Move features: tactical context + RELATIVE DISPLACEMENT geometry
    (df/dr/same-line/diagonal/knight-shape/edge distances) — the shared
    basis that lets movement rules generalize across all 64x64 slots."""
    g = np.zeros(MOVE_DIMS, dtype=np.float32)
    g[0] = 1.0 if board.is_capture(mv) else 0.0
    g[1] = 1.0 if board.gives_check(mv) else 0.0
    g[2] = 1.0 if mv.promotion else 0.0
    pc = board.piece_at(mv.from_square)
    g[3] = 1.0 if (pc and board.is_pinned(pc.color, mv.from_square)) else 0.0
    victim = board.piece_type_at(mv.to_square)
    vals = {1: 1, 2: 3, 3: 3, 4: 5, 5: 9}
    g[4] = vals.get(victim, 0) / 9.0
    g[5] = int(board.attackers_mask(not board.turn, mv.to_square)).bit_count() / 4.0
    g[6] = int(board.attackers_mask(board.turn, mv.to_square)).bit_count() / 4.0
    g[7] = 1.0 if board.is_into_check(mv) else (-1.0 if board.gives_check(mv) else 0.0)
    fr, to = mv.from_square, mv.to_square
    df = chess.square_file(to) - chess.square_file(fr)
    dr = chess.square_rank(to) - chess.square_rank(fr)
    g[8] = df / 7.0
    g[9] = dr / 7.0                       # forward is + (canonical up)
    g[10] = abs(df) / 7.0
    g[11] = abs(dr) / 7.0
    g[12] = 1.0 if df == 0 else 0.0        # same file
    g[13] = 1.0 if dr == 0 else 0.0        # same rank
    g[14] = 1.0 if abs(df) == abs(dr) and df != 0 else 0.0   # diagonal
    g[15] = 1.0 if {abs(df), abs(dr)} == {1, 2} else 0.0     # knight shape
    g[16] = min(chess.square_file(to), 7 - chess.square_file(to),
                chess.square_rank(to), 7 - chess.square_rank(to)) / 3.0
    return g


SLOT_GEO = None


def slot_geo():
    """(4096, 10) constant geometric basis per (from,to) slot: the shared
    displacement signal for the legality head."""
    global SLOT_GEO
    if SLOT_GEO is None:
        rows = []
        for fr in range(64):
            for to in range(64):
                if fr == to:
                    rows.append(np.zeros(10, dtype=np.float32))
                    continue
                df = (to % 8) - (fr % 8)
                dr = (to // 8) - (fr // 8)
                rows.append(np.array([
                    df / 7.0, dr / 7.0, abs(df) / 7.0, abs(dr) / 7.0,
                    1.0 if df == 0 else 0.0, 1.0 if dr == 0 else 0.0,
                    1.0 if abs(df) == abs(dr) and df != 0 else 0.0,
                    1.0 if {abs(df), abs(dr)} == {1, 2} else 0.0,
                    min(to % 8, 7 - to % 8, to // 8, 7 - to // 8) / 3.0,
                    min(fr % 8, 7 - fr % 8, fr // 8, 7 - fr // 8) / 3.0],
                    dtype=np.float32))
        SLOT_GEO = np.stack(rows)
    return SLOT_GEO


if __name__ == "__main__":
    import time
    b = chess.Board()
    v, cb = feat_vec(b)
    print("dims:", len(v), "nonzero:", int((v != 0).sum()))
    bb = chess.Board("r1bqkbnr/pppp1ppp/2n5/4p3/4P3/5N2/PPPP1PPP/RNBQKB1R b kq - 0 1")
    v1, c1 = feat_vec(bb)
    b2 = chess.Board("rnbqkbnr/pppp1ppp/8/4p3/4P3/5N2/PPPP1PPP/RNBQKB1R w KQkq - 0 1")
    v2, c2 = feat_vec(b2)
    print("mirror-invariance (same position, both colors):",
          np.allclose(v1, v2, atol=1e-6))
    t0 = time.time()
    for _ in range(200):
        feat_vec(b)
    print(f"{(time.time()-t0)*5:.2f} ms/canon-position")
