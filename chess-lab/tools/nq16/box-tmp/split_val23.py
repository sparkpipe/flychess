import sys, os, io, random, subprocess, re, math
sys.path.insert(0, "/srv/workspace/flychess/src/chess-lab/tools")
import router12 as R
import chess

SEG = "/extnvme/segments"
TARGET = "op_even_l1"
N_SAMPLE = 3000
random.seed(23)

# 1) collect TARGET-routed positions from game segments (clean + draws)
rows = []
for fn in sorted(os.listdir(SEG)):
    if not (fn.startswith("pos_") and (fn.endswith(".clean.tsv") or fn.endswith(".draws.tsv"))):
        continue
    for line in open(f"{SEG}/{fn}", errors="ignore"):
        p = line.rstrip("\n").split("|")
        if len(p) < 13:
            continue
        fen, cp, res = p[0], int(p[1]), p[11]
        e = R.route23(fen)
        if e == TARGET:
            rows.append((fen, cp, res))
print(f"game-only {TARGET} routed positions: {len(rows)}")
random.shuffle(rows)
rows = rows[:N_SAMPLE]

# 2) engine evals: pinned to TARGET net via slot 6 (PHASE_MOE=2)
E = "/mnt/cold-raid6/chess-audit/engine23"
ACT = "/srv/workspace/flychess/src/Stockfish-act/src/stockfish"
opts = [f"EvalFile={E}/tb.nnue"]
names = "mvr rv2m qvmat n2v2 pd_down pd_up oppb dv_rend dv_QRend dv_qend dv_core nvb op_gambiteer op_acceptor op_even_l0 op_even_l1 op_even_l2p mg_unsafe_king mg_safe_both_same mg_safe_my_castled mg_safe_uncastled mg_safe_other_castled".split()
opts += [f"EvalFile{i+2}={E}/{n}.nnue" for i, n in enumerate(names)]
opts[6] = f"EvalFile7={E}/{TARGET}.nnue"  # slot 6 (EX_DV_CORE) carries TARGET net
p = subprocess.Popen([ACT], stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                     stderr=subprocess.DEVNULL, text=True, bufsize=1, env={**os.environ, "PHASE_MOE": "2"})
def send(x):
    p.stdin.write(x + "\n"); p.stdin.flush()
send("uci")
while "uciok" not in p.stdout.readline():
    pass
send("setoption name Threads value 1")
for o in opts:
    k, v = o.split("=", 1)
    send(f"setoption name {k} value {v}")
send("isready")
while "readyok" not in p.stdout.readline():
    pass

def engine_cp(fen):
    send("position fen " + fen)
    send("eval")
    while True:
        l = p.stdout.readline()
        if not l:
            return None
        m = re.search(r"NNUE evaluation\s+(-?[\d.]+) \(white side\)", l)
        if m:
            return float(m.group(1))
        m = re.search(r"NNUE evaluation\s+(-?\d+) \(side to move", l)
        if m:
            stm = float(m.group(1))
            return stm if chess.Board(fen).turn == chess.WHITE else -stm

pairs = []
for fen, cp, res in rows:
    v = engine_cp(fen)
    if v is not None:
        pairs.append((cp, v, res))
send("quit")
print(f"evaluated: {len(pairs)}")

def stats(label, tf):
    xs = [tf(cp, res) for cp, v, res in pairs]
    ys = [v for cp, v, res in pairs]
    n = len(xs)
    mx, my = sum(xs) / n, sum(ys) / n
    cov = sum((x - mx) * (y - my) for x, y in zip(xs, ys)) / n
    vx = sum((x - mx) ** 2 for x in xs) / n
    vy = sum((y - my) ** 2 for y in ys) / n
    corr = cov / math.sqrt(max(vx * vy, 1e-12))
    mae = sum(abs(x - y) for x, y in zip(xs, ys)) / n
    print(f"{label:28s} MAE={mae:8.2f}cp corr={corr:+.4f}")

stats("as-written (identity)", lambda cp, res: cp)
stats("white-pov (flip on 0-1)", lambda cp, res: cp if res.startswith("1") else -cp if res.startswith("0") else cp)
