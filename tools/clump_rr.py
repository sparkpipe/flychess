"""Top-clump 10s round robin — runs after the Swiss.

Clump: all players within 1.5 points of the Swiss leader (min 3, max 8 players;
ties at the boundary broken by Swiss score). nQ included if it qualifies.
Format: every pair plays 2 games (colors reversed) at st=10, 3open book.
"""
import os, sys, json, glob, subprocess, re, time

R = "/mnt/cold-raid6/chess-audit"
SW = f"{R}/swiss_v4"
RR = f"{SW}/rr"
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
ST = 10

def spec(p):
    if p == "nQ":
        return f"name=nQ cmd={ACT} {OPTS}"
    return f"name={p} cmd={ACT} {OPTS} option.StackHead={SW}/heads/{p}.evh"

def score_pgn(path, first):
    txt = open(path).read()
    pts = 0.0; n = 0
    for g in txt.split("[Event")[1:]:
        res = re.search(r"\[Result .([^\"]+).", g)
        if not res or res.group(1) == "*":
            continue
        wh = re.search(r"\[White .([^\".]+).", g).group(1)
        sc = (1.0 if res.group(1) == "1-0" else 0.0 if res.group(1) == "0-1" else 0.5) \
            if wh == first else (0.0 if res.group(1) == "1-0" else 1.0 if res.group(1) == "0-1" else 0.5)
        pts += sc; n += 1
    return pts, n

def main():
    standings = json.load(open(f"{SW}/standings.json"))
    scores = standings["scores"]
    ranked = sorted(scores, key=lambda p: -scores[p])
    top = scores[ranked[0]]
    clump = [p for p in ranked if scores[p] >= top - 1.5][:8]
    if len(clump) < 3:
        clump = ranked[:3]
    print(f"clump (leader {top:.1f}): {[(p, scores[p]) for p in clump]}", flush=True)
    os.makedirs(RR, exist_ok=True)
    json.dump(clump, open(f"{RR}/clump.json", "w"))

    pairs = [(clump[i], clump[j]) for i in range(len(clump)) for j in range(i + 1, len(clump))]
    running = []
    idx = 0
    CONC = 6
    while idx < len(pairs) or running:
        while idx < len(pairs) and len(running) < CONC:
            a, b = pairs[idx]
            pgn = f"{RR}/{a}_vs_{b}.pgn"
            if not (os.path.exists(pgn) and open(pgn).read().count("[Result") >= 2):
                pr = subprocess.Popen(["bash", "-c",
                    " ".join([FC, "-engine", spec(a), "-engine", spec(b),
                              "-each", "proto=uci", f"st={ST}", "timemargin=200",
                              "-rounds", "1", "-repeat",
                              "-openings", f"file={R}/3open.epd", "format=epd",
                              "-pgnout", f"file={pgn}",
                              ">", pgn.replace(".pgn", ".log"), "2>&1"])])
                running.append((pr, a, b))
            idx += 1
        done = [r for r in running if r[0].poll() is not None]
        for r in done:
            running.remove(r)
        time.sleep(5)

    table = {}
    for a in clump:
        table[a] = {"pts": 0.0, "games": 0}
    for a, b in pairs:
        pgn = f"{RR}/{a}_vs_{b}.pgn"
        if os.path.exists(pgn):
            pa, n = score_pgn(pgn, a)
            table[a]["pts"] += pa; table[a]["games"] += n
            table[b]["pts"] += n - pa; table[b]["games"] += n
    final = sorted(table, key=lambda p: -table[p]["pts"])
    print("RR FINAL @10s:", [(p, table[p]["pts"], table[p]["games"]) for p in final], flush=True)
    json.dump({"clump": clump, "table": table, "final": final},
              open(f"{RR}/rr_standings.json", "w"), indent=1)

if __name__ == "__main__":
    main()
