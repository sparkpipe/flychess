"""Resolve winner color for pending matched games by scanning filtered.pgn:
replay each game; if any position's board matches one of the pending trough
fens, record the winner color from the result.
Output: gambit_aug_colors.txt  (lines: sfx gid color)
"""
import sys
import json
import chess
import chess.pgn

BASE = "/mnt/cold-raid6/rtx5090-archive/chess-lab/gambit-intermediates"
PGN = "/home/spec/chess-lab/gambit/filtered.pgn"
PEND = "/mnt/cold-raid6/chess-audit/gambit_aug_pending.txt"
OUT = "/mnt/cold-raid6/chess-audit/gambit_aug_colors.txt"

matched = {}
for sfx in ("", "_hist"):
    for gm in json.load(open(f"{BASE}/games{sfx}.json")):
        matched[(sfx, gm["gid"])] = gm["trough_ply"]

pending = set()
for line in open(PEND):
    if line.startswith(" "):          # elo scan: empty sfx written as " gid"
        pending.add(("", line.strip()))
    elif line.startswith("_hist "):
        pending.add(("_hist", line.split()[1]))

trough_fens = {}
for sfx in ("", "_hist"):
    with open(f"{BASE}/fens{sfx}.txt") as ff:
        for line in ff:
            p = line.split(" ", 3)
            key = (sfx, p[0])
            if len(p) < 4 or key not in pending:
                continue
            ply, tr = int(p[1]), matched[key]
            if tr <= ply <= tr + 4:
                trough_fens.setdefault(p[3].split(" ")[0], key)
print("pending games:", len(pending), "trough fens:", len(trough_fens),
      file=sys.stderr, flush=True)

resolved = {}
n = 0
with open(PGN, encoding="utf-8", errors="replace") as f, open(OUT, "w") as w:
    while True:
        g = chess.pgn.read_game(f)
        if g is None:
            break
        n += 1
        if n % 200000 == 0:
            print("scanned %d games, resolved %d" % (n, len(resolved)),
                  file=sys.stderr, flush=True)
        board = g.board()
        res = g.headers.get("Result", "*")
        if res not in ("1-0", "0-1"):
            continue
        for node in g.mainline():
            fen0 = board.board_fen()
            key = trough_fens.get(fen0)
            if key and key not in resolved:
                c = "w" if res == "1-0" else "b"
                resolved[key] = c
                w.write("%s %s %s\n" % (key[0] or "e", key[1], c))
            try:
                board.push(node.move)
            except Exception:
                break
            if len(resolved) == len(pending):
                break
        if len(resolved) == len(pending):
            break
print("resolved %d of %d pending" % (len(resolved), len(pending)))
print("COLORS-COMPLETE")
