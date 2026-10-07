import sys, os, subprocess, re, random, math
import chess
sys.path.insert(0, "/tmp")
from routerB import routeB

RUN = "/extnvme/active/train16_test_oppb"
ACT = "/srv/workspace/flychess/src/Stockfish-act/src/stockfish"
E16 = "/extnvme/active/engine23"
SEG = "/extnvme/segments"

rows = []
for fn in sorted(os.listdir(SEG)):
    if not (fn.startswith("pos_") and (fn.endswith(".clean.tsv") or fn.endswith(".draws.tsv"))):
        continue
    for line in open(f"{SEG}/{fn}", errors="ignore"):
        p = line.rstrip("\n").split("|")
        if len(p) < 8 or p[3] != "pos":
            continue
        rows.append((p[0], int(p[1]), p[11]))
        if len(rows) > 400000: break
    if len(rows) > 400000: break
rows = [r for r in rows if routeB(r[0]) == "oppb"]
random.seed(7); random.shuffle(rows); rows = rows[:600]

opts = [f"EvalFile={E16}/tb.nnue"]
names = "mvr rv2m qvmat n2v2 pd_down pd_up oppb dv_rend dv_QRend dv_qend dv_core nvb op_gambiteer op_acceptor op_even_l0 op_even_l1 op_even_l2p mg_unsafe_king mg_safe_both_same mg_safe_my_castled mg_safe_uncastled mg_safe_other_castled".split()
opts += [f"EvalFile{i+2}={E16}/{n}.nnue" for i, n in enumerate(names)]

def engine_evals(netfile, fens):
    o = list(opts)
    o[6] = f"EvalFile7={netfile}"
    p = subprocess.Popen([ACT], stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                         stderr=subprocess.DEVNULL, text=True, bufsize=1,
                         env={**os.environ, "PHASE_MOE": "2"})
    def send(x): p.stdin.write(x+"\n"); p.stdin.flush()
    send("uci")
    while "uciok" not in p.stdout.readline(): pass
    send("setoption name Threads value 1")
    for x in o:
        k, v = x.split("=", 1); send(f"setoption name {k} value {v}")
    send("isready")
    while "readyok" not in p.stdout.readline(): pass
    out = []
    for fen in fens:
        send("position fen "+fen); send("eval")
        v = None
        while True:
            l = p.stdout.readline()
            if not l: break
            m = re.search(r"NNUE evaluation\s+(-?[\d.]+)", l)
            if m:
                v = float(m.group(1))
                if "side to move" in l and chess.Board(fen).turn == chess.BLACK:
                    v = -v
                break
            if l.startswith("bestmove"): break
        out.append(v)
    send("quit")
    return out

fens = [r[0] for r in rows]
cps   = [r[1] for r in rows]
ress  = [r[2] for r in rows]
# climber==winner in positive-slope segments; convert climber-pov cp -> white-pov
cp_white = [c if r.startswith("1") else -c for c, r in zip(cps, ress)]

def stats(xs, ys):
    n = len(xs)
    mx, my = sum(xs)/n, sum(ys)/n
    cov = sum((a-mx)*(b-my) for a, b in zip(xs, ys))/n
    vx = sum((a-mx)**2 for a in xs)/n
    vy = sum((b-my)**2 for b in ys)/n
    corr = cov/math.sqrt(max(vx*vy, 1e-12))
    errs = sorted(abs(a-b) for a, b in zip(xs, ys))
    return corr, sum(errs)/n, errs[n//2]

print(f"{'net':14s} {'corr(raw)':>10s} {'corr(flip)':>10s} {'MAE':>7s} {'MED':>7s}")
for ep in (0, 9, 30, 75):
    nf = f"{RUN}/nets/oppb_e{ep}.nnue"
    if not os.path.exists(nf):
        continue
    ys = engine_evals(nf, fens)
    ok = [(c, cw, y) for c, cw, y in zip(cps, cp_white, ys) if y is not None]
    c1, m1, d1 = stats([x[0] for x in ok], [x[2] for x in ok])
    c2, m2, d2 = stats([x[1] for x in ok], [x[2] for x in ok])
    print(f"epoch {ep:<7d} {c1:+10.4f} {c2:+10.4f} {m2:7.1f} {d2:7.1f}")
