#!/usr/bin/env python3
"""Fast numpy extractor for FlyV3 — bit-identical to fly_v3_full.extract()
but vectorized: per-piece attack masks computed once, all per-square features
derived from one (K,64) bit matrix. Run directly to self-verify equality
against the reference extractor on random positions.
"""
import numpy as np, chess

S64 = np.arange(64, dtype=np.uint64)
FEATURE_KEYS = None


def _bits(masks):
    """(K,) uint64 masks -> (K,64) 0/1 float32 matrix."""
    if len(masks) == 0:
        return np.zeros((0, 64), dtype=np.float32)
    m = np.asarray(masks, dtype=np.uint64)
    return ((m[:, None] >> S64[None, :]) & np.uint64(1)).astype(np.float32)


def _pop(mask):
    return int(mask).bit_count()


def extract(board, hist_dv=None):
    f = {}
    pts = [chess.PAWN, chess.KNIGHT, chess.BISHOP, chess.ROOK, chess.QUEEN, chess.KING]
    # piece tables in board-scan order (must match reference eye ordering)
    occs = []
    for sq in chess.SQUARES:
        pc = board.piece_at(sq)
        if pc is not None:
            occs.append((sq, pc))
    ams = np.zeros(len(occs), dtype=np.uint64)
    for i, (sq, pc) in enumerate(occs):
        ams[i] = board.attacks_mask(sq)
    B = _bits(ams)                                   # (K,64) attack bits
    cols = np.array([pc.color for _, pc in occs], dtype=bool)   # True=white
    wrows = B[cols]; brows = B[~cols]
    cw = wrows.sum(axis=0) if len(wrows) else np.zeros(64, dtype=np.float32)
    cb = brows.sum(axis=0) if len(brows) else np.zeros(64, dtype=np.float32)

    for t, pt in enumerate(pts):
        f[f"bb_w_{t}"] = float(_pop(board.pieces_mask(pt, chess.WHITE)))
        f[f"bb_b_{t}"] = float(_pop(board.pieces_mask(pt, chess.BLACK)))
    cap4 = lambda x: np.minimum(x, 4) / 4.0
    for sq in chess.SQUARES:
        f[f"atk_w_{sq}"] = float(cap4(cw[sq])); f[f"cnt_w_{sq}"] = float(cap4(cw[sq]))
        f[f"atk_b_{sq}"] = float(cap4(cb[sq])); f[f"cnt_b_{sq}"] = float(cap4(cb[sq]))
        f[f"net_{sq}"] = float((np.minimum(cw[sq], 4) - np.minimum(cb[sq], 4)) / 4.0)
    # eye features for up to 32 pieces
    occ_w_v = np.uint64(board.occupied_co[chess.WHITE]) if board.occupied_co[chess.WHITE] else np.uint64(0)
    occ_b_v = np.uint64(board.occupied_co[chess.BLACK]) if board.occupied_co[chess.BLACK] else np.uint64(0)
    ow = ((occ_w_v >> S64) & np.uint64(1)).astype(np.float32)
    ob = ((occ_b_v >> S64) & np.uint64(1)).astype(np.float32)
    eyes = B + 2.0 * ow[None, :] + 4.0 * ob[None, :] if len(occs) else B
    occupied = board.occupied
    for i in range(32):
        if i < len(occs):
            sq, pc = occs[i]
            eye = eyes[i]
            for s2 in range(64):
                f[f"eye{i}_s{s2}"] = float(eye[s2])
            f[f"eye{i}_pin"] = 1.0 if board.is_pinned(pc.color, sq) else 0.0
            f[f"eye{i}_mob"] = float(_pop(board.attacks_mask(sq) & ~occupied))
        else:
            for s2 in range(64):
                f[f"eye{i}_s{s2}"] = 0.0
            f[f"eye{i}_pin"] = 0.0
            f[f"eye{i}_mob"] = 0.0
    for color, ko, opp in (("w", chess.WHITE, chess.BLACK), ("b", chess.BLACK, chess.WHITE)):
        king = board.king(ko)
        if king is None: continue
        ring = chess.BB_KING_ATTACKS[king] | (1 << king)
        for s2 in chess.SQUARES:
            if (ring >> s2) & 1:
                f[f"ks_{color}_enemy_{s2}"] = float(cap4(cb[s2] if ko == chess.WHITE else cw[s2]))
                f[f"ks_{color}_defend_{s2}"] = float(cap4(cw[s2] if ko == chess.WHITE else cb[s2]))
    f["check"] = 1.0 if board.is_check() else 0.0
    f["checkers"] = min(_pop(board.checkers_mask()), 4) / 4.0
    f["pinned_count"] = min(sum(1 for sq in chess.SQUARES
        if board.piece_at(sq) and board.is_pinned(board.piece_at(sq).color, sq)), 8) / 8.0
    for t, pt in enumerate(pts):
        f[f"mat_w_{t}"] = _pop(board.pieces_mask(pt, chess.WHITE)) / 8.0
        f[f"mat_b_{t}"] = _pop(board.pieces_mask(pt, chess.BLACK)) / 8.0
    center_ctrl_w = sum(board.attackers_mask(chess.WHITE, s) for s in
                        (chess.D4, chess.E4, chess.D5, chess.E5))
    center_ctrl_b = sum(board.attackers_mask(chess.BLACK, s) for s in
                        (chess.D4, chess.E4, chess.D5, chess.E5))
    f["center_w"] = min(_pop(center_ctrl_w), 8) / 8.0
    f["center_b"] = min(_pop(center_ctrl_b), 8) / 8.0
    mw = mb = 0
    for s in chess.SQUARES:
        pc = board.piece_at(s)
        if pc is None: continue
        n = _pop(board.attacks_mask(s) & ~occupied)
        if pc.color == chess.WHITE: mw += n
        else: mb += n
    f["mob_w"] = min(mw, 120) / 120.0
    f["mob_b"] = min(mb, 120) / 120.0
    dv = hist_dv or [0.0, 0.0, 0.0, 0.0]
    for k in range(4):
        f[f"dvh_{k}"] = dv[k]
    return f


def feat_vec(board, hist_dv=None):
    global FEATURE_KEYS
    d = extract(board, hist_dv)
    if FEATURE_KEYS is None:
        FEATURE_KEYS = sorted(d.keys())
    return np.array([d.get(k, 0.0) for k in FEATURE_KEYS], dtype=np.float32)


def move_feats(board, mv):
    g = np.zeros(8, dtype=np.float32)
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
    return g


if __name__ == "__main__":
    import fly_v3_full as v3, random, time, sys
    rng = random.Random(1)
    worst = 0.0
    for trial in range(40):
        b = chess.Board()
        for _ in range(rng.randrange(4, 70)):
            mvs = list(b.legal_moves)
            if not mvs: break
            b.push(rng.choice(mvs))
        d1 = v3.extract(b)
        d2 = extract(b)
        assert sorted(d1) == sorted(d2), "KEY MISMATCH"
        for k in d1:
            e = abs(d1[k] - d2[k])
            if e > 1e-6:
                print(f"MISMATCH {k}: ref={d1[k]} fast={d2[k]}")
                worst = max(worst, e)
        v1 = v3.feat_vec(b)
        v2 = feat_vec(b)
        assert np.allclose(v1, v2, atol=1e-6), "VEC MISMATCH"
    print(f"equality verified on 40 positions, worst diff {worst:.2e}")
    b = chess.Board()
    for _ in range(40):
        mvs = list(b.legal_moves)
        if not mvs: break
        b.push(rng.choice(mvs))
    t0 = time.time()
    for _ in range(100): feat_vec(b)
    print(f"fast feat_vec: {(time.time()-t0)*10:.2f} ms")
    t0 = time.time()
    for _ in range(20): v3.feat_vec(b)
    print(f"ref  feat_vec: {(time.time()-t0)*50:.2f} ms")
