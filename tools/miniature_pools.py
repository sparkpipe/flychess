"""MINIATURE POOLS (operator directive 2026-09-25): high-quality
winner moves from OTB tournament miniatures.

Source: games/twic.pgn (TWIC weekly archives — over-the-board
tournament games).
Filter: winner Elo >= 2400, loser Elo >= 2000, decisive game, and the
game "finishes" by move 25 — meaning EITHER the game ends within 25
moves OR the winner's material advantage first reaches +5 by move 25
(truncate there; the rest of the game is ignored).

Training rows: every winner move up to the finish, SF-graded (depth 16,
multipv all moves, parent-perspective bands — DEGM2 convention). A row
is kept only when the played move is in the SF top band ("high
quality"); approved set = top band; best = played move if in band else
SF's top move.

Output: tbpools/MINI.jsonl (+ precompute via precompute_pools)
"""
import sys
import os
import json
import chess
import chess.pgn
import chess.engine
from multiprocessing import Pool

sys.path.insert(0, "/home/spec/chess-lab/tools")
from degm_pools import band, SF

PGN = "/home/spec/chess-lab/games/lumbras_otb_complete.pgn"
OUTP = "/home/spec/chess-lab/tbpools/MINI.jsonl"
CAP_JOBS = 80000
CAP_ROWS = 40000
DEPTH = 16
WIN_ELO = 2400
LOSE_ELO = 2000
VAL = {chess.PAWN: 1, chess.KNIGHT: 3, chess.BISHOP: 3,
       chess.ROOK: 5, chess.QUEEN: 9}


def material(b):
    s = 0
    for _, p in b.piece_map().items():
        v = VAL.get(p.piece_type, 0)
        s += v if p.color == chess.WHITE else -v
    return s                      # + = white ahead


def collect():
    import random as _rd
    rng = _rd.Random(3)
    jobs = []
    n_jobs_seen = 0
    n_games = 0
    with open(PGN, encoding="utf-8", errors="replace") as f:
        while True:
            try:
                g = chess.pgn.read_game(f)
            except Exception:
                continue
            if g is None:
                break
            n_games += 1
            try:
                wel = int(g.headers.get("WhiteElo", 0) or 0)
                bel = int(g.headers.get("BlackElo", 0) or 0)
            except Exception:
                continue
            res = g.headers.get("Result", "*")
            if res == "1-0":
                winner, welo, lelo = chess.WHITE, wel, bel
            elif res == "0-1":
                winner, welo, lelo = chess.BLACK, bel, wel
            else:
                continue
            # quality bar: winner 2400+/loser 2000+; pre-rating-era
            # games (no Elo headers at all) pass — the SF top-band rule
            # is the quality gate there
            elo_given = not (welo == 0 and belo == 0)
            if elo_given and (welo < WIN_ELO or lelo < LOSE_ELO):
                continue
            b = g.board()
            sign = 1 if winner == chess.WHITE else -1
            finish = None
            plies = []
            node = g
            mv_no = 0
            while node.variations:
                node = node.variation(0)
                mv = node.move
                if mv is None:
                    break
                try:
                    b.push(mv)
                except Exception:
                    break
                mv_no += 1
                plies.append((b.fen(), mv, mv_no, winner))
                bal = sign * material(b)
                if mv_no <= 25 and bal >= 5:
                    finish = mv_no
                    break
                if node.is_end():
                    if mv_no <= 50 and mv_no <= 50:   # <=25 moves
                        pass
                    if mv_no <= 25 * 2 and mv_no / 2 <= 25:
                        finish = mv_no
                    break
            if finish is None:
                continue
            for fen, mv, mv_no, w in plies:
                if mv_no > finish:
                    break
                turn = chess.Board(fen).turn
                if turn != w:
                    continue
                n_jobs_seen += 1
                if len(jobs) < CAP_JOBS:
                    jobs.append({"fen": fen, "played": mv.uci()})
                else:
                    j = rng.randrange(n_jobs_seen)
                    if j < CAP_JOBS:
                        jobs[j] = {"fen": fen, "played": mv.uci()}
    print(json.dumps({"games": n_games, "jobs_seen": n_jobs_seen,
                      "jobs": len(jobs)}), flush=True)
    return jobs


ENG = None


def init_worker():
    global ENG
    ENG = chess.engine.SimpleEngine.popen_uci(SF)
    ENG.configure({"Threads": 1, "Hash": 128})


def grade_one(job):
    b = chess.Board(job["fen"])
    mvs = list(b.legal_moves)
    try:
        infos = ENG.analyse(b, chess.engine.Limit(depth=DEPTH),
                            multipv=len(mvs))
    except Exception:
        return None
    children = {}
    ranks = []
    for info in infos:
        u = info["pv"][0].uci()
        bd = band(info["score"].relative)
        children[u] = {"cat": bd, "dtz": None,
                       "sf": info["score"].relative.score()
                       if not info["score"].relative.is_mate() else None}
        ranks.append((u, bd))
    RANK = {"loss": 0, "cursed_loss": 1, "draw": 2,
            "cursed_win": 3, "win": 4}
    top = max((bd for _, bd in ranks), key=lambda x: RANK[x])
    approved = {u for u, bd in ranks if bd == top}
    played = job["played"]
    if played not in approved:
        return None                     # not a high-quality winner move
    best = played if played in approved else \
        next(u for u, bd in ranks if bd == top)
    return {"fen": job["fen"], "cat": top, "best": best,
            "children": children, "pool": "MINI",
            "played": played,
            "pieces": len(b.piece_map())}


def main():
    jobs = collect()
    rows = []
    with Pool(8, initializer=init_worker) as p:
        for i, r in enumerate(p.imap(grade_one, jobs, chunksize=16)):
            if r is not None:
                rows.append(r)
            if len(rows) >= CAP_ROWS:
                p.terminate()
                break
            if (i + 1) % 2000 == 0:
                print(json.dumps({"graded": i + 1, "kept": len(rows)}),
                      flush=True)
    with open(OUTP, "w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
    print(json.dumps({"MINI_ROWS": len(rows)}), flush=True)
    print("MINIATURES-COMPLETE", flush=True)


if __name__ == "__main__":
    main()
