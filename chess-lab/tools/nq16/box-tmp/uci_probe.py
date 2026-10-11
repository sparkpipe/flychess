import subprocess, sys, time, re
E="/mnt/cold-raid6/chess-audit/engine23"
N="/mnt/cold-raid6/chess-audit/nets"
ENGINES = {
 "nQ23": ("/srv/workspace/flychess/src/Stockfish-act/src/stockfish",
   {"EvalFile":f"{E}/tb.nnue","EvalFile2":f"{E}/mvr.nnue","EvalFile3":f"{E}/rv2m.nnue","EvalFile4":f"{E}/qvmat.nnue",
    "EvalFile5":f"{E}/n2v2.nnue","EvalFile6":f"{E}/pd_down.nnue","EvalFile7":f"{E}/pd_up.nnue","EvalFile8":f"{E}/oppb.nnue",
    "EvalFile9":f"{E}/dv_rend.nnue","EvalFile10":f"{E}/dv_QRend.nnue","EvalFile11":f"{E}/dv_qend.nnue","EvalFile12":f"{E}/dv_core.nnue",
    "EvalFile13":f"{E}/nvb.nnue","EvalFile14":f"{E}/op_gambiteer.nnue","EvalFile15":f"{E}/op_acceptor.nnue","EvalFile16":f"{E}/op_even_l0.nnue",
    "EvalFile17":f"{E}/op_even_l1.nnue","EvalFile18":f"{E}/op_even_l2p.nnue","EvalFile19":f"{E}/mg_unsafe_king.nnue","EvalFile20":f"{E}/mg_safe_both_same.nnue",
    "EvalFile21":f"{E}/mg_safe_my_castled.nnue","EvalFile22":f"{E}/mg_safe_uncastled.nnue","EvalFile23":f"{E}/mg_safe_other_castled.nnue"}),
 "nQ": ("/srv/workspace/flychess/src/Stockfish/src/stockfish",
   {"EvalFile":f"{N}/balanced_l0.nnue","EvalFile2":f"{N}/balanced_l1.nnue","EvalFile3":f"{N}/balanced_l2.nnue","EvalFile4":f"{N}/balanced_l3.nnue",
    "EvalFile5":f"{N}/nvb.nnue","EvalFile6":f"{N}/nvr.nnue","EvalFile7":f"{N}/bvr.nnue","EvalFile8":f"{N}/rv2m.nnue","EvalFile9":f"{N}/qvmat.nnue",
    "EvalFile10":f"{N}/oppb.nnue","EvalFile11":f"{N}/dvoretsky.nnue","EvalFile12":f"{N}/exchanges.nnue","EvalFile13":f"{N}/tactics.nnue"}),
 "SF8": ("/usr/games/stockfish", {}),
}
name = sys.argv[1]; mt = sys.argv[2] if len(sys.argv)>2 else "2000"
cmd, opts = ENGINES[name]
p = subprocess.Popen([cmd], stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                      stderr=None, text=True, bufsize=1)
def send(s): p.stdin.write(s+"\n"); p.stdin.flush()
send("uci")
while "uciok" not in (p.stdout.readline() or ""): pass
send("setoption name Threads value 1")
for k,v in opts.items(): send(f"setoption name {k} value {v}")
send("isready")
while "readyok" not in (p.stdout.readline() or ""): pass
send("position startpos moves e2e4 e7e5")
send(f"go movetime {mt}")
last=""
while True:
    line = p.stdout.readline()
    if not line: break
    if line.startswith("info") and " nps " in line: last=line.strip()
    if line.startswith("bestmove"):
        m=re.search(r"nodes (\d+) nps (\d+)", last)
        print(f"{name}: {m.group(0) if m else last} | {line.strip()}")
        break
send("quit")
