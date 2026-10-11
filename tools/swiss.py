"""SWISS tournament between training checkpoints — 9 rounds at st=1.

Players: ep{N}.pt checkpoints exported to .evh v4 (export_dense_v4 per ckpt).
Pairing: standard Swiss — sort by score, pair top-down, no rematches, random colors.
Each round: N/2 single games via fastchess (parallel batches).
Standings + PGNs accumulate in the swiss dir (tourney_live can serve it).
"""
import os, sys, glob, json, random, subprocess, re, time

R = "/mnt/cold-raid6/chess-audit"
SP = f"{R}/selfplay_rl"
SW = f"{R}/swiss_v4"
FC = "/srv/workspace/flychess/src/arena/fastchess-linux-x86-64/fastchess"
ACT = "/srv/workspace/flychess/src/Stockfish-act/src/stockfish"
N = f"{R}/nets"
OPTS = " ".join([
    f"option.EvalFile={N}/balanced_l0.nnue",
    f"option.EvalFile2={N}/balanced_l1.nnue", f"option.EvalFile3={N}/balanced_l2.nnue",
    f"option.EvalFile4={N}/balanced_l3.nnue", f"option.EvalFile5={N}/nvb.nnue",
    f"option.EvalFile6={N}/nvr.nnue", f"option.EvalFile7={N}/bvr.nnue",
    f"option.EvalFile8={N}/rv2m.nnue", f"option.EvalFile9={N}/qvmat.nnue",
    f"option.EvalFile10={N}/oppb.nnue", f"option.EvalFile11={N}/dvoretsky.nnue",
    f"option.EvalFile12={N}/exchanges.nnue", f"option.EvalFile13={N}/tactics.nnue",
])
ROUNDS = 9
ST = 1

def pair_round(players, scores, played, rng):
    order = sorted(players, key=lambda p: (-scores[p], rng.random()))
    remaining = list(order)
    pairs = []
    while len(remaining) > 1:
        a = remaining.pop(0)
        opp = None
        for j, b in enumerate(remaining):
            if (a, b) not in played and (b, a) not in played:
                opp = j; break
        if opp is None:
            opp = 0
        b = remaining.pop(opp)
        pairs.append((a, b))
    return pairs

def game(a, b, round_no, game_no, white_a):
    pa, pb = (a, b) if white_a else (b, a)
    pgn = f"{SW}/r{round_no:02d}g{game_no:02d}_{a}_vs_{b}.pgn"
    cmd = [FC,
           "-engine", f"name={pa}", f"cmd={ACT}", OPTS, f"option.StackHead={SW}/heads/{a}.evh",
           "-engine", f"name={pb}", f"cmd={ACT}", OPTS, f"option.StackHead={SW}/heads/{pb}.evh",
           "-each", "proto=uci", f"st={ST}", "timemargin=200",
           "-rounds", "1",
           "-pgnout", f"file={pgn}"]
    subprocess.run(cmd, capture_output=True, text=True, timeout=1800)
    try:
        txt = open(pgn).read()
        res = re.search(r"\[Result .([^\"]+).", txt).group(1)
    except Exception:
        res = "*"
    sa = 1.0 if res == "1-0" else 0.0 if res == "0-1" else 0.5 if res == "1/2-1/2" else 0.0
    if not white_a:
        sa = 1.0 - sa
    return sa, res

def main():
    os.makedirs(f"{SW}/heads", exist_ok=True)
    players = [os.path.basename(p)[:-3] for p in sorted(glob.glob(f"{SP}/dense_v4_ckpts2/ep*.pt"))]
    players.append("nQ")
    if len(players) < 10:
        print(f"only {len(players)} checkpoints — need >=10 for a 9-round Swiss"); return
    print(f"{len(players)} players, {ROUNDS} rounds, st={ST}s", flush=True)
    scores = {p: 0.0 for p in players}
    played = set()
    rng = random.Random(42)
    standings_f = f"{SW}/standings.json"
    for rd in range(1, ROUNDS + 1):
        pairs = pair_round(players, scores, played, rng)
        procs = []
        for i, (a, b) in enumerate(pairs):
            played.add((a, b)); played.add((b, a))
            procs.append((a, b, i))
        running = []
        idx = 0
        CONC = 6
        results = {}
        while idx < len(procs) or running:
            while idx < len(procs) and len(running) < CONC:
                a, b, i = procs[idx]
                white_a = rng.random() < 0.5
                def spec(p):
                    if p == "nQ":
                        return f"name=nQ cmd={ACT} {OPTS}"
                    return f"name={p} cmd={ACT} {OPTS} option.StackHead={SW}/heads/{p}.evh"
                pr = subprocess.Popen(
                    ["bash", "-c",
                     " ".join([FC, "-engine", spec(a if white_a else b),
                               "-engine", spec(b if white_a else a),
                               "-each", "proto=uci", f"st={ST}", "timemargin=200",
                               "-rounds", "1",
                               "-pgnout", f"file={SW}/r{rd:02d}g{i:02d}_{a}_vs_{b}.pgn",
                               ">", f"{SW}/r{rd:02d}g{i:02d}.log", "2>&1"])])
                running.append((pr, a, b, white_a, i))
                idx += 1
            done = [r for r in running if r[0].poll() is not None]
            for pr, a, b, white_a, i in done:
                running.remove((pr, a, b, white_a, i))
                pgn = f"{SW}/r{rd:02d}g{i:02d}_{a}_vs_{b}.pgn"
                try:
                    res = re.search(r"\[Result .([^\"]+).", open(pgn).read()).group(1)
                except Exception:
                    res = "0-1"
                sa = 1.0 if res == "1-0" else 0.0 if res == "0-1" else 0.5
                if not white_a:
                    sa = 1.0 - sa
                scores[a] += sa
                scores[b] += 1.0 - sa
            time.sleep(3)
        print(f"round {rd} done: " + ", ".join(f"{p}={scores[p]:.1f}" for p in sorted(players, key=lambda q: -scores[q])[:4]), flush=True)
        json.dump({"scores": scores, "round": rd}, open(standings_f, "w"), indent=1)
    final = sorted(players, key=lambda p: -scores[p])
    print("FINAL:", [(p, scores[p]) for p in final])
    json.dump({"scores": scores, "final": final}, open(standings_f, "w"), indent=1)

if __name__ == "__main__":
    main()
