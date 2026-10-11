import subprocess, re
E="/srv/workspace/chess-active/engine16"
ACT="/srv/workspace/flychess/src/Stockfish/src/stockfish"
names = "tb mvr rv2m qvmat nvb piece_down oppb dv_Q dv_R dv_rest op_pawnimb op_even_l0 op_even_l1 op_even_l2p mg_unsafe mg_safe".split()
opts = [f"EvalFile={E}/tb.nnue"]
opts += [f"EvalFile{i+2}={E}/{n}.nnue" for i, n in enumerate(names[1:])]
p = subprocess.Popen([ACT], stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                     stderr=subprocess.DEVNULL, text=True, bufsize=1)
def send(x): p.stdin.write(x+"\n"); p.stdin.flush()
send("uci")
while "uciok" not in p.stdout.readline(): pass
send("setoption name Threads value 1")
for o in opts:
    k, v = o.split("=", 1); send(f"setoption name {k} value {v}")
send("isready")
while "readyok" not in p.stdout.readline(): pass
tests = [("rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1","op_even_l0"),
         ("8/8/3k4/8/8/2R5/8/K7 w - - 0 1","tb"),
         ("8/2p5/3p4/KP5r/1R3p1k/8/4P1P1/8 w - - 0 1","dv_R"),
         ("8/8/4k1p1/2KpP2p/5PP1/8/8/8 w - - 0 53","dv_R"),
         ("r2q1rk1/pp2bppp/2n1pn2/3p4/3P4/2N1PN2/PP2BPPP/R1BQ1RK1 w - - 0 9","piece_down")]
for fen, want in tests:
    send("position fen " + fen); send("route")
    l = p.stdout.readline()
    print(f"{want:>12s}: {l.strip()}")
send("position startpos"); send("go movetime 1500")
last = ""
while True:
    l = p.stdout.readline()
    if not l: break
    if l.startswith("info") and " nps " in l: last = l
    if l.startswith("bestmove"):
        m = re.search(r"nodes (\d+) nps (\d+)", last)
        print("search:", l.split()[1], m.group(0) if m else "")
        break
send("quit")
