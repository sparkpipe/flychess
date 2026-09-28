"""Gambit expert data: extract the INITIATIVE ERA following each mined gambit
drop — from the drop moment until the sacrificed material is regained (or a
20-ply cap). Positions from the existing mining scan (fens.txt, gid-indexed);
played moves reconstructed from consecutive FENs of the same game.

Input:  GAMBIT.jsonl  (mined drops: fen, best, ply, gid, ...)
        fens.txt      (lines: "gid ply x fen" for the full filtered pool)
Output: eval-fleet format lines  fen|move|gambit|ply|0|0|gid
Usage:  extract_gambit_era.py <GAMBIT.jsonl> <fens.txt> <out.txt>
"""
import sys
import json
import chess

pool_path, fens_path, out_path = sys.argv[1], sys.argv[2], sys.argv[3]

# gid -> list of drop plies (dedupe multiple drops per game)
drops = {}
for line in open(pool_path):
    r = json.loads(line)
    gid = r.get("gid")
    if gid and r.get("best"):
        drops.setdefault(gid, []).append(int(r.get("ply", 0)))
print("pool: %d games, %d drops" % (len(drops), sum(len(v) for v in drops.values())))


def pawn_diff(board):
    wm = board.pawns & board.occupied_co[chess.WHITE]
    bm = board.pawns & board.occupied_co[chess.BLACK]
    return chess.popcount(wm) - chess.popcount(bm)


n_out = n_games = 0
cur_gid, game = None, []

def flush(gid, positions):
    global n_out, n_games
    if gid not in drops or not positions:
        return
    n_games += 1
    by_ply = {p: f for p, f in positions}
    for drop_ply in drops[gid]:
        try:
            b0 = chess.Board(by_ply[drop_ply])
        except Exception:
            continue
        ref_diff = pawn_diff(b0)
        for p in range(drop_ply, min(drop_ply + 20, max(by_ply))):
            if p not in by_ply or (p + 1) not in by_ply:
                break
            b = chess.Board(by_ply[p])
            if p > drop_ply and pawn_diff(b) >= ref_diff + 1 \
                    and b.turn == b0.turn:
                break  # material regained: initiative era over
            # played move: the legal move leading to the next dumped fen
            nxt = chess.Board(by_ply[p + 1])
            for mv in b.legal_moves:
                b.push(mv)
                if b.board_fen() == nxt.board_fen() and b.turn == nxt.turn:
                    b.pop()
                    break
                b.pop()
            else:
                continue
            with open(out_path, "a") as w:
                w.write("%s|%s|gambit|%d|0|0|%s\n" % (by_ply[p], mv.uci(), p, gid))
            n_out += 1

for line in open(fens_path):
    parts = line.split()
    if len(parts) < 4:
        continue
    gid = parts[0]
    if gid != cur_gid:
        flush(cur_gid, game)
        cur_gid, game = gid, []
        if n_games % 10000 == 0 and n_games:
            print("games %d, positions %d" % (n_games, n_out), flush=True)
    game.append((int(parts[1]), parts[3]))
flush(cur_gid, game)

print("GAMBIT ERA: %d games -> %d positions" % (n_games, n_out))
print("GAMBIT-ERA-COMPLETE")
