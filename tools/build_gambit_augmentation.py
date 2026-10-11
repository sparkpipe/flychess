"""Gambit AUGMENTATION builder (ruling 2026-09-28).

All 94,395 matched gambit games: winner positions from the winner's FIRST
MOVE to the ply where winner wp reaches 70%. Classified later by the usual
cascade; duplicates with the main set are deliberate (gambit-style weight).

Phase 1 (this script): emit winner-arc positions with played moves for all
games whose winner color is known (trough-fen lookup in the both-sides main
dump). Unknown-color gids go to a pending file for PGN resolution.

Output: gambit_aug_positions.txt (fen|move|gaug|ply|0|0|gid)
        gambit_aug_pending.txt  (gids needing color resolution)
"""
import sys
import json
import chess

BASE = "/mnt/cold-raid6/rtx5090-archive/chess-lab/gambit-intermediates"
BOTH = "/mnt/cold-raid6/chess-audit/otb_complete_dump_both.txt"
OUT = "/mnt/cold-raid6/chess-audit/gambit_aug_positions.txt"
PEND = "/mnt/cold-raid6/chess-audit/gambit_aug_pending.txt"

matched = {}
for sfx in ("", "_hist"):
    for gm in json.load(open(f"{BASE}/games{sfx}.json")):
        matched[(sfx, gm["gid"])] = gm["trough_ply"]
print("matched:", len(matched), file=sys.stderr, flush=True)

# winner color via trough-position lookup in the main dump
need = {}
for sfx in ("", "_hist"):
    with open(f"{BASE}/fens{sfx}.txt") as ff:
        for line in ff:
            p = line.split(" ", 3)
            if len(p) < 4 or (sfx, p[0]) not in matched:
                continue
            ply = int(p[1])
            tr = matched[(sfx, p[0])]
            if tr <= ply <= tr + 4:
                need.setdefault((sfx, p[0]), {}).setdefault(ply, p[3].split(" ")[0])
fen2key = {fen: k for k, d in need.items() for fen in d.values()}
color = {}
with open(BOTH) as f:
    for line in f:
        k = fen2key.get(line.split("|", 1)[0].split(" ")[0])
        if k and k not in color:
            color[k] = line.split("|")[2]
print("colors known:", len(color), file=sys.stderr, flush=True)
white_to_move = {"w": chess.WHITE, "b": chess.BLACK}

n_out = 0
pending = []
with open(OUT, "w") as w, open(PEND, "w") as pw:
    for sfx in ("", "_hist"):
        cur = None
        plies = {}
        arc = None

        def flush(key, plies):
            global n_out
            if key not in matched or not plies:
                return
            c = color.get(key)
            if c is None:
                pending.append(key)
                return
            wtm = white_to_move[c]
            # arc end: first winner position at/after trough with W >= 0.70
            end = None
            for p in sorted(plies):
                if p >= matched[key] and plies[p][1] >= 0.70:
                    end = p
                    break
            if end is None:
                end = max(plies)
            for p in sorted(plies):
                if p > end:
                    break
                fen, W = plies[p]
                b = chess.Board(fen)
                if b.turn != wtm:
                    continue
                nxt = plies.get(p + 1)
                if not nxt:
                    continue
                nb = chess.Board(nxt[0])
                for mv in b.legal_moves:
                    b.push(mv)
                    ok = b.board_fen() == nb.board_fen() and b.turn == nb.turn
                    b.pop()
                    if ok:
                        w.write("%s|%s|gaug|%d|0|0|%s%s\n"
                                % (fen, mv.uci(), p, sfx or "e", key[1]))
                        n_out += 1
                        break

        with open(f"{BASE}/fens{sfx}.txt") as ff, \
                open(f"{BASE}/evals{sfx}.txt") as fe:
            for fline, eline in zip(ff, fe):
                fp = fline.rstrip("\n").split(" ", 3)
                ep = eline.split()
                if len(fp) < 4 or len(ep) < 3:
                    continue
                key = (sfx, fp[0])
                if key != cur:
                    flush(cur, plies)
                    cur, plies = key, {}
                if key in matched:
                    plies[int(fp[1])] = (fp[3], float(ep[2]))
        flush(cur, plies)
    for key in pending:
        pw.write("%s %s\n" % key)

print("AUG positions: %d; pending color: %d" % (n_out, len(pending)))
print("AUG-PHASE1-COMPLETE")
