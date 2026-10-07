import subprocess
opts=[f"EvalFile{i+1}=/mnt/cold-raid6/chess-audit/engine23/{n}.nnue" for i,n in enumerate("tb mvr rv2m qvmat n2v2 pd_down pd_up oppb dv_rend dv_QRend dv_qend dv_core nvb op_gambiteer op_acceptor op_even_l0 op_even_l1 op_even_l2p mg_unsafe_king mg_safe_both_same mg_safe_my_castled mg_safe_uncastled mg_safe_other_castled".split())]
opts[0]="EvalFile=/mnt/cold-raid6/chess-audit/engine23/tb.nnue"
p=subprocess.Popen(["/srv/workspace/flychess/src/Stockfish-act/src/stockfish"],stdin=subprocess.PIPE,stdout=subprocess.PIPE,stderr=subprocess.DEVNULL,text=True,bufsize=1)
def send(x): p.stdin.write(x+"\n"); p.stdin.flush()
send("uci")
while "uciok" not in p.stdout.readline(): pass
for o in opts: send(f"setoption name {o.replace(chr(61),chr(32)+chr(118)+chr(97)+chr(108)+chr(117)+chr(101)+chr(32),1)}")
send("isready")
while "readyok" not in p.stdout.readline(): pass
fens=["rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1","rnbqkbnr/ppp1pppp/8/3p4/4P3/8/PPPP1PPP/RNBQKBNR w KQkq - 0 2","8/2p5/3p4/KP5r/1R3p1k/8/4P1P1/8 w - - 0 1","r1bqkb1r/pppp1ppp/2n2n2/4p3/2B1P3/5N2/PPPP1PPP/RNBQK2R w KQkq - 4 4","r2q1rk1/pp2bppp/2n1pn2/3p4/3P4/2N1PN2/PP2BPPP/R1BQ1RK1 w - - 0 9"]
for f in fens:
    send("position fen "+f); send("eval")
    while True:
        l=p.stdout.readline()
        if "NNUE evaluation" in l: print(f.split()[0][:20], "->", l.strip()[:70]); break
        if l.startswith("bestmove"): break
send("quit")
