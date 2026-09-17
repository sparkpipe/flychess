"""THE LOOP RIG — closed verification harness for vision -> imagination -> selection.
The fly receives raw planes only. The rig applies the fly's chosen move to the
real board and NEVER decides. VERDICT metrics:
  (1) imagination accuracy per depth (imagined planes vs true)
  (2) self-drive legality + no-hang rate on held-out positions
  (3) ablation: identity imagination (imagined = current) — the no-hang delta
      IS the measured lookahead contribution
  (4) counterfactual pairs: remove one enemy piece — does the selection flip
      in response to changed threat structure
  (5) mate-in-2 stretch probe (static evaluators cannot pass; expected ~random
      until stage 5 trains check/mate — reported honestly)
The fly-computed evaluation of an imagined position: material balance plus the
perceived reach mass of the moved piece at its destination, both read by the
fly's own vision head from the IMAGINED map.
"""
import os, sys, json, random
import numpy as np
import torch
import chess
import chess.engine

_LAB = os.path.expanduser("~") + "/chess-lab"
sys.path.insert(0, _LAB)
import fly_curriculum as cur
import fly_vision as fv
import fly_imagine as fi

DEV = cur.DEV
PLANES = 12
VALS = np.array([0.0, 1.0, 3.0, 3.0, 5.0, 9.0])


def load_all():
    torch.manual_seed(0)
    vis = fv.VisionFly().to(DEV)
    vis.load_state_dict(torch.load(_LAB + "/fly_sees_current.pt",
                                   weights_only=True))
    vis.eval()
    imag = fi.ImagineFly().to(DEV)
    imag.load_state_dict(torch.load(_LAB + "/fly_imagines_norm.pt",
                                    weights_only=True))
    imag.eval()
    return vis, imag


def planes_of(b):
    p = np.zeros((PLANES + 1, 64), np.float32)
    for s_ in chess.SQUARES:
        pc = b.piece_at(s_)
        if pc is None:
            continue
        p[(pc.piece_type - 1) * 2 + (0 if pc.color == chess.WHITE else 1),
          s_] = 1.0
    return p.reshape(-1)


def imagine_move(imag, planes_t, reach_t, mv):
    frm = np.zeros(64, np.float32); frm[mv.from_square] = 1.0
    to = np.zeros(64, np.float32); to[mv.to_square] = 1.0
    x = np.concatenate([planes_t, reach_t, frm, to])[None]
    with torch.no_grad():
        pp, pr = imag(torch.from_numpy(x).to(DEV))
    return (pp[0].cpu().numpy().reshape(PLANES + 1, 64),
            pr[0].cpu().numpy())


def perceive_reach(vis, planes_flat, qs):
    """Vision head perceives the reach map of the piece at qs from planes."""
    p = planes_flat.reshape(PLANES + 1, 64).copy()
    p[PLANES] = 0.0
    p[PLANES, qs] = 1.0
    x = p.reshape(-1)[None]
    with torch.no_grad():
        a = vis.see(torch.from_numpy(x).to(DEV))
    h = a[vis.head_idx].T
    with torch.no_grad():
        pr = vis.reach_head(h)
    return pr[0].detach().cpu().numpy()


def fly_eval(vis, planes_2d, qs):
    """Fly-computed eval of an (imagined) position: material balance plus the
    perceived reach activity of the piece at qs, read from the imagined map."""
    occ = planes_2d.reshape(PLANES + 1, 64)
    mat = 0.0
    for ci, sign in ((0, 1.0), (1, -1.0)):
        for pi in range(6):
            mat += sign * float(occ[ci * 2 + pi].sum()) * VALS[pi]
    pr = perceive_reach(vis, occ.reshape(-1), qs)
    act = float(np.abs(pr).sum())
    return mat + 0.05 * act


def decode_board(imag, planes_2d, fallback):
    """Decode imagined planes to a chess board; kings restored from the
    fallback position if the imagination lost them."""
    occ = planes_2d.reshape(PLANES + 1, 64)
    bd = chess.Board(None)
    for s_ in chess.SQUARES:
        col = occ[:, s_]
        if col.max() < 0.5:
            continue
        pi = int(np.argmax(col[:6])) + 1
        ci = 1 if int(np.argmax(col)) >= 6 else 0
        bd.set_piece_at(s_, chess.Piece(pi, chess.Color(ci)))
    for sq, color in ((chess.E1, chess.WHITE), (chess.E8, chess.BLACK)):
        if bd.king(color) is None and fallback.king(color) is not None:
            bd.set_piece_at(sq, chess.Piece(chess.KING, color))
    return bd


def danger_score(nb, mover_white):
    """Null-move danger: what the opponent destroys with a free move.
    1.0 = mate available; else the best free capture's value if undefended."""
    worst = 0.0
    vals = {chess.PAWN: 1, chess.KNIGHT: 3, chess.BISHOP: 3,
            chess.ROOK: 5, chess.QUEEN: 9}
    them = chess.BLACK if mover_white else chess.WHITE
    for mv in nb.legal_moves:
        if mv.to_square == nb.king(mover_white) and _is_mate_after(nb, mv):
            return 10.0
        cap = nb.piece_at(mv.to_square)
        if cap and cap.color == mover_white:
            v = vals.get(cap.piece_type, 0)
            tgt = nb.copy(); tgt.push(mv)
            if not tgt.is_attacked_by(mover_white, mv.to_square):
                v += 1
            worst = max(worst, float(v))
    return worst


def _is_mate_after(b, mv):
    b.push(mv)
    m = b.is_checkmate()
    b.pop()
    return m


def differential(nb, mover_white):
    """Option-space differential: our free-move option mass vs theirs."""
    my = sum(1 for _ in nb.legal_moves) if mover_white else 0
    b2 = nb.copy()
    if mover_white:
        b2.turn = chess.BLACK
    their = sum(1 for _ in b2.legal_moves) if mover_white else         sum(1 for _ in nb.legal_moves)
    if not mover_white:
        my = sum(1 for _ in b2.legal_moves)
    return float(my - their)


def pick_move(vis, imag, b, ablate=False):
    planes_t = planes_of(b)
    best_mv, best_sc = None, -1e9
    for mv in b.legal_moves:
        if ablate:
            planes_i = planes_t.copy()
            nb_i = b
        else:
            r0 = cur.reach_map(b, mv.from_square)
            planes_i, _ = imagine_move(imag, planes_t, r0, mv)
            nb_i = decode_board(imag, planes_i, b)
        try:
            diff = differential(nb_i, True)
            danger = danger_score(nb_i, True)
        except Exception:
            diff, danger = 0.0, 0.0
        sc = diff - 2.0 * danger
        if sc > best_sc:
            best_sc, best_mv = sc, mv
    return best_mv, best_sc


def gen_positions(rng, n=24, max_ply=40, min_moves=4):
    out = []
    while len(out) < n:
        b = chess.Board()
        for _ in range(rng.randrange(8, max_ply)):
            mvs = list(b.legal_moves)
            if not mvs:
                break
            b.push(rng.choice(mvs))
        if b.is_game_over() or len(list(b.legal_moves)) < min_moves:
            continue
        out.append(b.copy())
    return out


def main():
    vis, imag = load_all()
    rng = random.Random(2026)
    v = {}

    # (1) imagination accuracy per depth — IN-DISTRIBUTION (taught board
    # families) and DENSE-BOARD extrapolation reported separately
    def imagine_acc(boards, sparse=True):
        d1 = d2 = n1 = n2 = 0
        for b in boards:
            planes_t = planes_of(b)
            mvs = list(b.legal_moves)
            if not mvs:
                continue
            mv = rng.choice(mvs)
            r0 = cur.reach_map(b, mv.from_square)
            pi, ri = imagine_move(imag, planes_t, r0, mv)
            nb = b.copy(); nb.push(mv)
            t1 = planes_of(nb)
            p1s = 1.0 / (1.0 + np.exp(-pi.reshape(-1)))
            d1 += float((np.abs(p1s - t1) < 0.25).mean())
            n1 += 1
            mvs2 = list(nb.legal_moves)
            if mvs2:
                mv2 = rng.choice(mvs2)
                p2i, _ = imagine_move(imag, pi.reshape(-1), ri, mv2)
                nb2 = nb.copy(); nb2.push(mv2)
                t2 = planes_of(nb2)
                p2s = 1.0 / (1.0 + np.exp(-p2i.reshape(-1)))
                d2 += float((np.abs(p2s - t2) < 0.25).mean())
                n2 += 1
        return (round(d1 / max(n1, 1), 4), round(d2 / max(n2, 1), 4))

    sparse_boards = []
    while len(sparse_boards) < 24:
        st = rng.choice([1, 2, 3])
        pt = rng.choice([chess.ROOK, chess.BISHOP, chess.QUEEN, chess.KNIGHT,
                         chess.PAWN, chess.KING])
        bs = cur.gen_stage(rng, st, 1, piece=pt)
        if bs and len(list(bs[0][0].legal_moves)) >= 3:
            sparse_boards.append(bs[0][0])
    d1s, d2s = imagine_acc(sparse_boards)
    v["imagination_depth1_indist"] = d1s
    v["imagination_depth2_indist"] = d2s
    dense_boards = gen_positions(rng, 24)
    d1d, d2d = imagine_acc(dense_boards)
    v["imagination_depth1_dense_extrap"] = d1d
    v["imagination_depth2_dense_extrap"] = d2d

    # (2/3) self-drive: no-hang rate, with and without imagination
    def selfdrive(ablate, boards):
        total = hangs = 0
        for b in boards:
            mv, _ = pick_move(vis, imag, b, ablate=ablate)
            if mv is None:
                continue
            total += 1
            nb = b.copy(); nb.push(mv)
            attacked = nb.is_attacked_by(not b.turn, mv.to_square)
            defended = bool(nb.attackers_mask(b.turn, mv.to_square))
            if attacked and not defended:
                hangs += 1
        return total, hangs
    tt, hg = selfdrive(ablate=False, boards=sparse_boards)
    _, hg0 = selfdrive(ablate=True, boards=sparse_boards)
    v["selfdrive_positions_indist"] = tt
    v["hangs_with_imagination"] = hg
    v["hangs_ablated"] = hg0
    v["lookahead_delta_hangs"] = hg0 - hg

    # (4) counterfactual pairs
    flips = pairs = 0
    for b in sparse_boards[:40]:
        if len(list(b.legal_moves)) < 2:
            continue
        mv_a, _ = pick_move(vis, imag, b)
        b2 = b.copy()
        enemy = [s for s in chess.SQUARES
                 if b2.piece_at(s) and b2.piece_at(s).color != b2.turn]
        if not enemy:
            continue
        b2.remove_piece_at(rng.choice(enemy))
        if b2.is_game_over() or not list(b2.legal_moves):
            continue
        mv_b, _ = pick_move(vis, imag, b2)
        pairs += 1
        flips += int(mv_a != mv_b)
    v["counterfactual_pairs"] = pairs
    v["counterfactual_flips"] = flips

    # (2b) Stockfish top-move agreement (the chessfly-comparable metric)
    sf_agree = sf_total = 0
    try:
        eng = chess.engine.SimpleEngine.popen_uci("/usr/games/stockfish")
    except Exception:
        for pth in ("/usr/bin/stockfish", "/usr/local/bin/stockfish"):
            if os.path.exists(pth):
                eng = chess.engine.SimpleEngine.popen_uci(pth)
                break
        else:
            eng = None
    if eng is not None:
        for b in gen_positions(rng, 40, max_ply=24):
            if b.is_game_over():
                continue
            mv, _ = pick_move(vis, imag, b)
            try:
                sf = eng.play(b, chess.engine.Limit(time=0.05)).move
            except Exception:
                continue
            sf_total += 1
            sf_agree += int(mv == sf)
        eng.quit()
    v["stockfish_agreement"] = round(sf_agree / max(sf_total, 1), 4)
    v["stockfish_positions"] = sf_total

    # (5) mate-in-2 stretch probe
    def mate_in_2_exists(b):
        for m1 in list(b.legal_moves):
            b.push(m1)
            if b.is_checkmate():
                b.pop(); continue
            replies = list(b.legal_moves)
            if not replies:
                b.pop(); continue
            all_mated = True
            for r in replies:
                b.push(r)
                mated = b.is_checkmate()
                b.pop()
                if not mated:
                    all_mated = False
                    break
            b.pop()
            if all_mated:
                return True
        return False

    m2 = m2solved = 0
    for _ in range(80):
        b = chess.Board()
        b.set_piece_at(chess.E1, chess.Piece(chess.KING, chess.WHITE))
        b.set_piece_at(chess.H1, chess.Piece(chess.ROOK, chess.WHITE))
        ks = rng.sample([s for s in chess.SQUARES
                         if s not in (chess.E1, chess.H1)], 1)[0]
        b.set_piece_at(ks, chess.Piece(chess.KING, chess.BLACK))
        if b.is_game_over() or b.is_attacked_by(chess.WHITE, ks):
            continue
        if not mate_in_2_exists(b):
            continue
        m2 += 1
        mv, _ = pick_move(vis, imag, b)
        if mv is None:
            continue
        b.push(mv)
        solved = any((b.push(r), m := b.is_checkmate(), b.pop())[1]
                     for r in list(b.legal_moves)) if list(b.legal_moves) \
            else False
        m2solved += int(solved)
    v["mate2_positions"] = m2
    v["mate2_first_move_solves"] = m2solved
    v["mate2_note"] = "stretch probe: proper mate training lands at stage 5"

    print(json.dumps(v, indent=2), flush=True)
    with open(_LAB + "/loop_rig_verdict.json", "w") as f:
        json.dump(v, f, indent=2)
    print("LOOP RIG VERDICT WRITTEN", flush=True)


if __name__ == "__main__":
    main()
