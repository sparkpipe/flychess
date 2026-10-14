#!/usr/bin/env python3
"""NPS: n16-baseline vs nQ13, fixed 300k-node searches on real corpus positions."""
import json
import subprocess
import urllib.request

N16 = "/srv/workspace/flychess/src/Stockfish/src/stockfish"
NQ = "/mnt/cold-raid6/chess-audit/nq13_binary_backup"
NQN = "/mnt/cold-raid6/chess-audit/nets"
E = "/srv/workspace/chess-active/engine16-baseline"

req = urllib.request.Request(
    "http://127.0.0.1:8123/allocate",
    data=json.dumps({"node": "npsbench", "engine": "sf17", "depth": 12,
                     "n": 12}).encode(),
    headers={"Content-Type": "application/json"})
r = json.load(urllib.request.urlopen(req, timeout=30))
fens = r["fens"][:12]

NQ_MAP = ["balanced_l0", "balanced_l1", "balanced_l2", "balanced_l3", "nvb",
          "nvr", "bvr", "rv2m", "qvmat", "oppb", "dvoretsky", "exchanges",
          "tactics"]
N16_MAP = ["tb", "mvr", "rv2m", "qvmat", "nvb", "piece_down", "oppb", "dv_Q",
           "dv_R", "dv_rest", "op_pawnimb", "op_even_l0", "op_even_l1",
           "op_even_l2p", "mg_unsafe", "mg_safe"]


def opts_for(netdir, names):
    out = []
    for i, n in enumerate(names):
        opt = "EvalFile" if i == 0 else f"EvalFile{i+1}"
        out.append(f"setoption name {opt} value {netdir}/{n}.nnue")
    return out


def bench(binary, opts):
    p = subprocess.Popen([binary], stdin=subprocess.PIPE,
                         stdout=subprocess.PIPE, stderr=subprocess.DEVNULL,
                         text=True, bufsize=1)
    p.stdin.write("uci\n")
    p.stdin.flush()
    while "uciok" not in p.stdout.readline():
        pass
    for o in opts:
        p.stdin.write(o + "\n")
    p.stdin.write("isready\n")
    p.stdin.flush()
    while "readyok" not in p.stdout.readline():
        pass
    total_n = total_ms = 0
    for fen in fens:
        p.stdin.write(f"position fen {fen}\ngo nodes 300000\n")
        p.stdin.flush()
        n = s = 0
        while True:
            l = p.stdout.readline()
            if not l:
                break
            if l.startswith("info ") and " nodes " in l and " nps " in l:
                t = l.split()
                n = int(t[t.index("nodes") + 1])
                s = int(t[t.index("nps") + 1])
            elif l.startswith("bestmove"):
                if s > 0:
                    total_n += n
                    total_ms += n * 1000.0 / s
                break
    p.stdin.write("quit\n")
    return total_n / (total_ms / 1000) if total_ms else 0


q = bench(NQ, opts_for(NQN, NQ_MAP))
s = bench(N16, opts_for(E, N16_MAP))
print(f"nQ13:      {q:,.0f} nps")
print(f"n16-base:  {s:,.0f} nps")
print(f"ratio:     {q/s:.3f}  (n16 needs {q/s:.2f}s to match nQ 1s)")
