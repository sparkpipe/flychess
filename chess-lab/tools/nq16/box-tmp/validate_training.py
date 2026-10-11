"""Training-path validation for the oppb test run.
1. checkpoint.N files load (torch) and contain resumable state
2. exported .nnue loads in the engine and evaluates
3. val metrics from the final net via engine evals: MEAN AE, MEDIAN AE, corr
"""
import sys, os, subprocess, re, json, random
import chess

sys.path.insert(0, "/tmp")
from routerB import routeB

RUN = "/extnvme/active/train16_test_oppb"
ACT = "/srv/workspace/flychess/src/Stockfish-act/src/stockfish"
E16 = "/extnvme/active/engine23"
SEG = "/extnvme/segments"

print("=== 1. checkpoint validation")
import torch
for f in sorted(os.listdir(RUN)):
    if f.startswith("checkpoint."):
        ck = torch.load(f"{RUN}/{f}", map_location="cpu", weights_only=False)
        has_sd = "state_dict" in ck
        has_opt = "optimizer_states" in ck
        print(f"  {f}: state_dict={has_sd} optimizer={has_opt} epoch={ck.get('epoch')}")

print("=== 2. .nnue engine validation")
nets = sorted([f for f in os.listdir(f"{RUN}/nets")],
              key=lambda x: int(re.search(r"_e(\d+)", x).group(1)))
first, last = nets[0], nets[-1]
opts = [f"EvalFile={E16}/tb.nnue"]
names = "mvr rv2m qvmat n2v2 pd_down pd_up oppb dv_rend dv_QRend dv_qend dv_core nvb op_gambiteer op_acceptor op_even_l0 op_even_l1 op_even_l2p mg_unsafe_king mg_safe_both_same mg_safe_my_castled mg_safe_uncastled mg_safe_other_castled".split()
opts += [f"EvalFile{i+2}={E16}/{n}.nnue" for i, n in enumerate(names)]
for nf in (first, last):
    o = list(opts)
    o[6] = f"EvalFile7={RUN}/nets/{nf}"   # slot 6 carries the test net
    p = subprocess.Popen([ACT], stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                         stderr=subprocess.DEVNULL, text=True, bufsize=1,
                         env={**os.environ, "PHASE_MOE": "2"})
    def send(x):
        p.stdin.write(x + "\n"); p.stdin.flush()
    send("uci")
    while "uciok" not in p.stdout.readline():
        pass
    send("setoption name Threads value 1")
    for x in o:
        k, v = x.split("=", 1)
        send(f"setoption name {k} value {v}")
    send("isready")
    while "readyok" not in p.stdout.readline():
        pass
    send("position fen r1bqk2r/pppp1ppp/2n2n2/2b1p3/2B1P3/2N2N2/PPPP1PPP/R1BQK2R w KQkq - 6 5")
    send("eval")
    val = None
    while True:
        l = p.stdout.readline()
        m = re.search(r"NNUE evaluation\s+(-?[\d.]+)", l)
        if m:
            val = m.group(1); break
        if l.startswith("bestmove") or not l:
            break
    send("quit")
    print(f"  {nf}: engine eval = {val} {'OK' if val is not None else 'FAIL'}")

print("=== 3. val metrics (mean/median AE, corr) from final net via engine")
# rebuild the oppb val sample: segments routed to oppb, val-split (hash(seg)%20==0)
rows = []
for fn in sorted(os.listdir(SEG)):
    if not (fn.startswith("pos_") and (fn.endswith(".clean.tsv") or fn.endswith(".draws.tsv"))):
        continue
    for line in open(f"{SEG}/{fn}", errors="ignore"):
        p = line.rstrip("\n").split("|")
        if len(p) < 8 or p[3] != "pos":
            continue
        seg_id = p[5]
        if hash(seg_id) % 20 != 0:
            continue
        rows.append((p[0], int(p[1])))
        if len(rows) > 400000:
            break
    if len(rows) > 400000:
        break
rows = [r for r in rows if routeB(r[0]) == "oppb"]
random.seed(7)
random.shuffle(rows)
rows = rows[:2000]
print(f"  val sample: {len(rows)} oppb positions")

o = list(opts)
o[6] = f"EvalFile7={RUN}/nets/{last}"
p = subprocess.Popen([ACT], stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                     stderr=subprocess.DEVNULL, text=True, bufsize=1,
                     env={**os.environ, "PHASE_MOE": "2"})
def send(x):
    p.stdin.write(x + "\n"); p.stdin.flush()
send("uci")
while "uciok" not in p.stdout.readline():
    pass
send("setoption name Threads value 1")
for x in o:
    k, v = x.split("=", 1)
    send(f"setoption name {k} value {v}")
send("isready")
while "readyok" not in p.stdout.readline():
    pass
pairs = []
for fen, cp in rows:
    send("position fen " + fen)
    send("eval")
    while True:
        l = p.stdout.readline()
        if not l:
            break
        m = re.search(r"NNUE evaluation\s+(-?[\d.]+)", l)
        if m:
            v = float(m.group(1))
            if "side to move" in l:
                v = v if chess.Board(fen).turn == chess.WHITE else -v
            else:
                pass  # white side already
            pairs.append((cp, v))
            break
        if l.startswith("bestmove"):
            break
send("quit")
print(f"  evaluated: {len(pairs)}")
import math
xs = [a for a, b in pairs]; ys = [b for a, b in pairs]
n = len(xs)
mx, my = sum(xs)/n, sum(ys)/n
cov = sum((a-mx)*(b-my) for a, b in pairs)/n
vx = sum((a-mx)**2 for a in xs)/n
vy = sum((b-my)**2 for b in ys)/n
corr = cov / math.sqrt(max(vx*vy, 1e-12))
mae = sum(abs(a-b) for a, b in pairs)/n
med = sorted(abs(a-b) for a, b in pairs)[n//2]
print(f"  MEAN AE   = {mae:.2f} cp")
print(f"  MEDIAN AE = {med:.2f} cp")
print(f"  corr      = {corr:+.4f}")
print("VALIDATION COMPLETE")
