#!/usr/bin/env python3
"""SF17 eval sweep for the live UI: fills evals_d{12,20,25} + top-5 pv_d{...}
per ply, white-pov cp. Pass-based (all plies d12, then d20, then d25), incremental.
"""
import fcntl
import json
import os
import subprocess
import sys
import time

SF17 = "/srv/workspace/flychess/src/sf17/src/stockfish"
OUT = "/srv/workspace/chess-active/matches/games.json"
DEPTHS = [12, 20, 25]
LOCK = OUT + ".lock"


class Engine:
    def __init__(self, path):
        self.p = subprocess.Popen(
            [path], stdin=subprocess.PIPE, stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL, text=True, bufsize=1)
        self.cmd("uci\nsetoption name MultiPV value 5\nisready\n")
        while "readyok" not in self.p.stdout.readline():
            pass

    def cmd(self, s):
        self.p.stdin.write(s)
        self.p.stdin.flush()

    def top5(self, fen, depth):
        self.cmd(f"position fen {fen}\ngo depth {depth}\n")
        lines = {}
        bare = None  # score from a pv-less info line (checkmate/stalemate end plies)
        while True:
            l = self.p.stdout.readline()
            if not l:
                return []
            if l.startswith("info ") and " score " in l:
                toks = l.split()
                cp = mate = None
                for i, t in enumerate(toks):
                    if t in ("cp", "mate") and toks[i - 1] == "score":
                        if t == "cp":
                            cp = int(toks[i + 1])
                        else:
                            mate = int(toks[i + 1])
                        break
                if " pv " in l:
                    mp = pv = None
                    for i, t in enumerate(toks):
                        if t == "multipv":
                            mp = int(toks[i + 1])
                        elif t == "pv":
                            pv = " ".join(toks[i + 1:])
                            break
                    if mp is not None:
                        if mate is not None:
                            # mate encoding: 10000 + mate-in (cap 900); see DATA-CONVENTIONS
                            v = (10000 + min(abs(mate), 900)) * (1 if mate > 0 else -1)
                        else:
                            v = cp
                        lines[mp] = (v, pv)
                elif cp is not None or mate is not None:
                    if mate is not None:
                        # mate 0 = side to move is mated (stm-pov -(10000+N))
                        bare = (10000 + min(abs(mate), 900)) * (1 if mate > 0 else -1)
                    else:
                        bare = cp
            elif l.startswith("bestmove"):
                break
        stm = 1 if fen.split()[1] == "w" else -1
        if lines:
            return [(v * stm, pv) for _, (v, pv) in sorted(lines.items())][:5]
        if bare is not None:
            # end ply (checkmate/stalemate): no pv lines exist at all
            return [(bare * stm, "#")]
        return []


def load_save(update):
    """Run update(doc) under a file lock; return True if it changed anything."""
    with open(LOCK, "w") as lf:
        fcntl.flock(lf, fcntl.LOCK_EX)
        try:
            doc = json.load(open(OUT))
        except Exception:
            return False
        changed = update(doc)
        if changed:
            tmp = OUT + ".tmp"
            with open(tmp, "w") as f:
                json.dump(doc, f)
            os.replace(tmp, OUT)
        return changed


def main():
    mod = int(sys.argv[1]) if len(sys.argv) > 1 else 0
    every = int(sys.argv[2]) if len(sys.argv) > 2 else 1
    eng = Engine(SF17)

    def my_games(doc):
        out = []
        for m in doc:
            for gi, g in enumerate(m["games"]):
                if gi % every == mod:
                    out.append(g)
        return out

    while True:
        try:
            games = my_games(json.load(open(OUT)))
        except Exception:
            time.sleep(5)
            continue
        changed = False
        for d in DEPTHS:  # operator: all games at d12, then d20, then d25
          for g in games:
            n = len(g["fens"])
            if True:
                ek = f"evals_d{d}"
                cur = g.get(ek) or []
                # find the next ply to fill: first gap OR first suspect zero
                # (a 0 in the middle of a game is almost always a failed eval
                # that got written as a placeholder — re-check it)
                i = len(cur)
                for j, v in enumerate(cur):
                    if j > 2 and v == 0:
                        i = j
                        break
                if i >= n:
                    continue
                res = eng.top5(g["fens"][i], d)
                if not res:
                    continue  # never write a fake 0 — retry this ply next pass
                ev = res[0][0]
                pv = " ;; ".join(f"{v}|{p}" for v, p in res)

                def update_by_fens(doc, _g=g, _ek=ek, _i=i, _ev=ev, _pv=pv):
                    did = False
                    for m in doc:
                        for gg in m["games"]:
                            if gg.get("fens") == _g.get("fens"):
                                arr = gg.setdefault(_ek, [])
                                while len(arr) <= _i:
                                    arr.append(0)
                                pvs = gg.setdefault(
                                    _ek.replace("evals_", "pv_"), [])
                                while len(pvs) <= _i:
                                    pvs.append("")
                                if arr[_i] != _ev:
                                    arr[_i] = _ev
                                    pvs[_i] = _pv
                                    did = True
                    return did

                if load_save(update_by_fens):
                    changed = True
        if not changed:
            time.sleep(8)


if __name__ == "__main__":
    main()
