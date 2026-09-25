"""STAGE C — mine the successful-gambit trajectory shape.

Input: gambit/fens.txt (gid ply winner fen) + gambit/evals.txt
       (gid ply W) — same order.
Shape (winner win-probability W over plies 1..50):
  TROUGH: min W in plies 6..15 <= TROUGH_MAX (soft: modern evals see
          gambits as ~neutral)
  CLIMB:  W at trough -> W at ply 40..50 rises >= CLIMB_MIN with no
          collapse in between (running min after trough >= trough - 0.1)
  JUMP:   max W in plies 30..50 >= JUMP_MIN
Rows for training: every WINNER move position from the trough onward,
played move = best (game-authority), in fens of those games.

Output: gambit/games.json (matched games + stats),
        gambit/gambit_fens.txt (winner positions, trough..finish).
"""
import sys
import os
import json
from collections import defaultdict

G = "/home/spec/chess-lab/gambit"
TROUGH_MAX = float(os.environ.get("TROUGH_MAX", "0.60"))
CLIMB_MIN = float(os.environ.get("CLIMB_MIN", "0.30"))
JUMP_MIN = float(os.environ.get("JUMP_MIN", "0.90"))


def main():
    traj = {}
    order = []
    with open(f"{G}/evals.txt") as fe, open(f"{G}/fens.txt") as ff:
        for ev, ff_line in zip(fe, ff):
            gid, ply, w = ev.split()
            fen = ff_line.rstrip("\n").split(" ", 3)[3]
            t = traj.get(gid)
            if t is None:
                t = traj[gid] = {"plies": [], "W": [], "fens": [],
                                 "winner": None}
                order.append(gid)
            t["plies"].append(int(ply))
            t["W"].append(float(w))
            t["fens"].append(fen)
            t["winner"] = ff_line.split(" ", 3)[2]
    print(json.dumps({"games": len(traj)}), flush=True)

    matched = []
    for gid in order:
        t = traj[gid]
        W = t["W"]
        pl = t["plies"]
        if len(W) < 20:
            continue
        # trough in plies 6..15
        early = [(w, p) for w, p in zip(W, pl) if 6 <= p <= 15]
        if not early:
            continue
        wmin, pmin = min(early)
        if wmin > TROUGH_MAX:
            continue
        # climb: best W late (ply >= 30) minus trough
        late = [w for w, p in zip(W, pl) if p >= 30]
        if not late:
            continue
        wlate = max(late)
        if wlate - wmin < CLIMB_MIN or wlate < JUMP_MIN:
            continue
        # no collapse after trough
        post = [w for w, p in zip(W, pl) if p >= pmin]
        if min(post) < wmin - 0.10:
            continue
        # winner-move rows from the trough onward
        rows = []
        for p, fen in zip(pl, t["fens"]):
            if p >= pmin:
                rows.append(f"{gid} {p} {fen}")
        matched.append({"gid": gid, "trough": round(wmin, 3),
                        "trough_ply": pmin, "peak": round(wlate, 3),
                        "n_rows": len(rows)})
        with open(f"{G}/gambit_fens.txt", "a") as f:
            for r in rows:
                f.write(r + "\n")
        if len(matched) % 500 == 0:
            print(json.dumps({"matched": len(matched)}), flush=True)
    peaks = sorted((m["peak"] for m in matched), reverse=True)
    print(json.dumps({"MATCHED_GAMES": len(matched),
                      "total_rows": sum(m["n_rows"]
                                        for m in matched),
                      "peak_top5": peaks[:5]}), flush=True)
    json.dump(matched, open(f"{G}/games.json", "w"))
    print("STAGE-C-COMPLETE", flush=True)


if __name__ == "__main__":
    main()
