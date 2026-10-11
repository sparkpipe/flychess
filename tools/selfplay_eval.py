"""Evaluate self-play positions with the fork itself (same nets).
Uses the fork binary directly with the full 13-slot loadout.
Output: fen|move|winner|ply|welo|belo|result|cp||best|12 (same as eval fleet)."""
import sys
import os
import subprocess

IN, OUT = sys.argv[1], sys.argv[2]
OPTS = sys.argv[3]  # full option string

SF = "/home/spec/Stockfish/src/stockfish"

with open(IN) as f:
    lines = f.readlines()

p = subprocess.Popen([SF], stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                     stderr=subprocess.DEVNULL, text=True, bufsize=1)
p.stdin.write("uci\n")
while not p.stdout.readline().startswith("uciok"):
    pass
for opt in OPTS.split():
    if opt.startswith("option."):
        name = opt[7:].split("=", 1)[0]
        val = opt.split("=", 1)[1] if "=" in opt else ""
        p.stdin.write("setoption name %s value %s\n" % (name, val))
p.stdin.write("setoption name Threads value 1\nsetoption name Hash value 128\nisready\n")
while not p.stdout.readline().startswith("readyok"):
    pass

n = 0
with open(OUT, "w") as w:
    for line in lines:
        parts = line.rstrip("\n").split("|")
        if len(parts) < 7:
            continue
        fen = parts[0]
        try:
            p.stdin.write("position fen %s\ngo depth 8\n" % fen)
            p.stdin.flush()
            cp = None
            while True:
                r = p.stdout.readline()
                if not r:
                    raise RuntimeError("died")
                if r.startswith("info ") and " score cp " in r:
                    t = r.split()
                    cp = int(t[t.index("cp") + 1])
                elif r.startswith("bestmove"):
                    break
            if cp is None:
                cp = 0
            # cp is stm-perspective from SF; flip to white perspective
            stm_black = fen.split(" ")[1] == "b"
            wcp = -cp if stm_black else cp
            w.write("%s|%d\n" % (line.rstrip("\n"), wcp))
            n += 1
        except Exception:
            try:
                p.kill()
            except Exception:
                pass
            p = subprocess.Popen([SF], stdin=subprocess.PIPE,
                                 stdout=subprocess.PIPE,
                                 stderr=subprocess.DEVNULL, text=True, bufsize=1)
            p.stdin.write("uci\n")
            while not p.stdout.readline().startswith("uciok"):
                pass
            for opt in OPTS.split():
                if opt.startswith("option."):
                    name = opt[7:].split("=", 1)[0]
                    val = opt.split("=", 1)[1] if "=" in opt else ""
                    p.stdin.write("setoption name %s value %s\n" % (name, val))
            p.stdin.write("setoption name Threads value 1\nisready\n")
            while not p.stdout.readline().startswith("readyok"):
                pass
try:
    p.stdin.write("quit\n")
except Exception:
    pass
print("SELFPLAY EVAL: %d positions" % n)
