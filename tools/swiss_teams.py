"""FULL-TEAM CHECKPOINT SWISS — complete 13-expert engines at each training epoch.

Players (each = ALL 13 slots from one round+epoch, cumulative as operator intends):
  nQ        = nets/   (quarter-trained, the incumbent)
  r2_ep19   = all 13 from runs2 at epoch 19
  r2_ep39   = all 13 from runs2 at epoch 39
  r2_ep59   = all 13 from runs2 at epoch 59
  r2_last   = all 13 from runs2 final (= nR2 as played in the 8-tournament)
  r3_ep19   = best available (4/13 have it; rest from r3 last — mixed, noted)
  r3_last   = best available (= nR3 approximation)

Slot-13 handling: rounds 2/3 trained tb_training (nets2 has it), quarter uses
tactics. For r2 teams we use tb_training_r2_last (the era-correct 13th expert).
Missing experts filled from the team's last checkpoint.

9 rounds, st=1, startpos, Swiss pairing, 6 concurrent.
"""
import os, sys, glob, json, random, subprocess, re, time

R = "/mnt/cold-raid6/chess-audit"
CK = f"{R}/nets_ckpts"
SW = f"{R}/swiss_teams"
FC = "/srv/workspace/flychess/src/arena/fastchess-linux-x86-64/fastchess"
FORK = "/srv/workspace/flychess/src/Stockfish/src/stockfish"
N = f"{R}/nets"
N2 = f"{R}/nets2"
N3 = f"{R}/nets3"
ROUNDS = 9
ST = 1

BASE13 = ["balanced_l0","balanced_l1","balanced_l2","balanced_l3",
          "nvb","nvr","bvr","rv2m","qvmat","oppb",
          "dvoretsky","exchanges","tactics"]

def team_nets(player):
    """returns list of 13 .nnue paths for this team"""
    if player == "nQ":
        return [f"{N}/{e}.nnue" for e in BASE13]
    if player.startswith("r2_"):
        ep = player[3:]
        files = []
        for e in BASE13[:12]:
            f = f"{CK}/{e}_r2_{ep}.nnue"
            if not os.path.exists(f):
                f = f"{CK}/{e}_r2_last.nnue"
            files.append(f)
        # 13th slot: tb_training for the r2 era
        files.append(f"{CK}/tb_training_r2_last.nnue")
        return files
    if player.startswith("r3_"):
        ep = player[3:]
        files = []
        for e in BASE13[:12]:
            f = f"{CK}/{e}_r3_{ep}.nnue"
            if not os.path.exists(f):
                f = f"{CK}/{e}_r3_last.nnue"
            if not os.path.exists(f):
                f = f"{N3}/{e}.nnue"
            files.append(f)
        files.append(f"{N2}/tb_training.nnue")
        return files
    raise ValueError(player)

def spec_of(player):
    nets = team_nets(player)
    o = [f"option.EvalFile={nets[0]}"]
    o += [f"option.EvalFile{i+2}={nets[i]}" for i in range(1, 13)]
    return f"name={player} cmd={FORK} " + " ".join(o)

def pair_round(players, scores, played, rng):
    order = sorted(players, key=lambda p: (-scores[p], rng.random()))
    rem = list(order)
    pairs = []
    while len(rem) > 1:
        a = rem.pop(0)
        opp = None
        for j, b in enumerate(rem):
            if (a, b) not in played and (b, a) not in played:
                opp = j; break
        if opp is None:
            opp = 0
        b = rem.pop(opp)
        pairs.append((a, b))
    return pairs

def main():
    os.makedirs(SW, exist_ok=True)
    players = ["nQ", "r2_ep19", "r2_ep39", "r2_ep59", "r2_last",
               "r3_ep19", "r3_last"]
    # verify all teams loadable
    for p in players:
        nets = team_nets(p)
        missing = [n for n in nets if not os.path.exists(n)]
        if missing:
            print(f"WARN {p}: missing {missing}", flush=True)
    print(f"{len(players)} teams, {ROUNDS} rounds, st={ST}s — full round-robin "
          f"({len(players)*(len(players)-1)//2} pairings x 2 games)", flush=True)

    scores = {p: 0.0 for p in players}
    played = set()
    rng = random.Random(99)
    # small field -> full round-robin with 2 games per pairing (color-reversed)
    pairs = []
    for i in range(len(players)):
        for j in range(i + 1, len(players)):
            pairs.append((players[i], players[j]))
    running = []
    idx = 0
    CONC = 6
    while idx < len(pairs) or running:
        while idx < len(pairs) and len(running) < CONC:
            a, b = pairs[idx]
            pgn = f"{SW}/{a}_vs_{b}.pgn"
            if not (os.path.exists(pgn) and open(pgn).read().count("[Result") >= 2):
                pr = subprocess.Popen(["bash", "-c",
                    " ".join([FC, "-engine", spec_of(a), "-engine", spec_of(b),
                              "-each", "proto=uci", f"st={ST}", "timemargin=200",
                              "-rounds", "1", "-repeat",
                              "-pgnout", f"file={pgn}",
                              ">", pgn.replace(".pgn", ".log"), "2>&1"])])
                running.append((pr, a, b, pgn))
            idx += 1
        done = [r for r in running if r[0].poll() is not None]
        for pr, a, b, pgn in done:
            running.remove((pr, a, b, pgn))
            try:
                txt = open(pgn).read()
                pts = 0.0; n = 0
                for g in txt.split("[Event")[1:]:
                    res = re.search(r"\[Result .([^\"]+).", g)
                    if not res or res.group(1) == "*": continue
                    wh = re.search(r"\[White .([^\"]+).", g).group(1)
                    sc = (1.0 if res.group(1)=="1-0" else 0.0 if res.group(1)=="0-1" else 0.5) if wh==a else (0.0 if res.group(1)=="1-0" else 1.0 if res.group(1)=="0-1" else 0.5)
                    pts += sc; n += 1
                scores[a] += pts
                scores[b] += n - pts
            except Exception as e:
                print(f"  ERROR {a} vs {b}: {e}", flush=True)
        time.sleep(5)
        if not running and idx >= len(pairs):
            break
    final = sorted(players, key=lambda p: -scores[p])
    print("FINAL:", [(p, scores[p]) for p in final], flush=True)
    json.dump({"scores": scores, "final": final}, open(f"{SW}/standings.json", "w"), indent=1)

if __name__ == "__main__":
    main()
