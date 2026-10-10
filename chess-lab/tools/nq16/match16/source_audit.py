#!/usr/bin/env python3
"""Source label convention audit: sign-match vs SF17 stm-pov and white-pov per source."""
import glob
import random
import subprocess

SF17 = "/srv/workspace/flychess/src/sf17/src/stockfish"
random.seed(11)


class SF:
    def __init__(self):
        self.p = subprocess.Popen([SF17], stdin=subprocess.PIPE,
                                  stdout=subprocess.PIPE,
                                  stderr=subprocess.DEVNULL, text=True, bufsize=1)
        self.p.stdin.write("uci\nisready\n")
        self.p.stdin.flush()
        while "readyok" not in self.p.stdout.readline():
            pass

    def cp(self, fen, depth=12):
        self.p.stdin.write(f"position fen {fen}\ngo depth {depth}\n")
        self.p.stdin.flush()
        last = None
        while True:
            l = self.p.stdout.readline()
            if not l:
                return None
            if l.startswith("info ") and "score cp " in l and " pv " in l:
                t = l.split()
                for i, x in enumerate(t):
                    if x == "cp":
                        last = int(t[i + 1])
            elif l.startswith("bestmove"):
                return last


def audit(name, pairs, sf):
    stm_ok = white_ok = n = 0
    for fen, lab in pairs:
        t = sf.cp(fen)
        if t is None:
            continue
        stm = "w" in fen.split()[1]
        tw = t if stm else -t
        n += 1
        if lab * t > 0:
            stm_ok += 1
        if lab * tw > 0:
            white_ok += 1
    print(f"{name:<24} n={n:<4} stm-sign {stm_ok}/{n}   white-sign {white_ok}/{n}"
          f"   -> {'STM-pov' if stm_ok > white_ok else 'WHITE-pov' if white_ok > stm_ok else 'mixed/unclear'}")


sf = SF()

# puzzle shards (sample 2 shards x 40)
rows = []
for sh in sorted(glob.glob("/home/spec/backlog_shards/puzzle_eval/w*.tsv"))[:2]:
    lines = open(sh, errors="replace").readlines()
    for l in random.sample(lines, min(40, len(lines))):
        p = l.rstrip("\n").split("\t")
        if len(p) >= 5:
            rows.append((p[0], int(p[4])))
audit("puzzle shards", rows, sf)

# segments (sample 40)
rows = []
segs = sorted(glob.glob("/mnt/cold-raid6/chess-audit/wp_fit/segments/*.tsv"))
if segs:
    f = segs[len(segs) // 2]
    lines = open(f, errors="replace").readlines()
    hdr = 1 if lines and lines[0].startswith("fen") else 0
    for l in random.sample(lines[hdr:], min(40, len(lines) - hdr)):
        p = l.rstrip("\n").split("\t")
        # tolerate fen, gid, ply, cp / or fen, cp layouts
        try:
            rows.append((p[0], int(p[3]) if len(p) > 3 and p[3].lstrip("-").isdigit() else int(p[1])))
        except Exception:
            pass
audit(f"segments ({segs[len(segs)//2].split('/')[-1][:20] if segs else '-'})", rows, sf)

# miniature labels (sample 40)
rows = []
ml = "/srv/workspace/chess-active/miniature_labels.tsv"
lines = open(ml, errors="replace").readlines()
for l in random.sample(lines, min(40, len(lines))):
    fen, cp = l.rstrip("\n").split("\t")
    rows.append((fen, int(cp)))
audit("miniature labels", rows, sf)

# degm
rows = []
for l in open("/mnt/cold-raid6/chess-audit/wp_fit/trainsets/sources/degm.tsv", errors="replace").readlines()[:2000]:
    p = l.rstrip("\n").split("|")
    rows.append((p[0], int(p[2]) / 100.0))
audit("degm book", random.sample(rows, 40), sf)

sf.p.stdin.write("quit\n")
