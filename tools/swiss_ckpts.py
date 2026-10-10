"""CHECKPOINT SWISS — round-2/round-3 training-curve engines + nQ, 1s/move.

A checkpoint-engine = same 13-slot loadout but with ONE slot's net swapped
to a checkpoint version (e.g., balanced_l0_r2_ep19 in place of balanced_l0).
Players = {nQ} + {each expert's checkpoint variants}. Same routing.
This measures: does swapping one expert to an earlier/later checkpoint
change engine strength? -> per-expert training curves, measured by combat.

Pairing: standard Swiss, no rematches, startpos, st=1.
"""
import os, sys, glob, json, random, subprocess, re, time

R = "/mnt/cold-raid6/chess-audit"
CK = f"{R}/nets_ckpts"
SW = f"{R}/swiss_ckpt"
FC = "/srv/workspace/flychess/src/arena/fastchess-linux-x86-64/fastchess"
FORK = "/srv/workspace/flychess/src/Stockfish/src/stockfish"
N = f"{R}/nets"
ROUNDS = 9
ST = 1

BASE = ["balanced_l0","balanced_l1","balanced_l2","balanced_l3",
        "nvb","nvr","bvr","rv2m","qvmat","oppb",
        "dvoretsky","exchanges","tactics"]

def opts_for(nets_dir, swap=None):
    """13-slot opts from nets_dir; swap = (expert, ckpt_name) replaces that slot."""
    o = []
    for i, e in enumerate(BASE):
        f = e
        if swap and e == swap[0]:
            f = swap[1]
        opt = "EvalFile" if i == 0 else f"EvalFile{i+1}"
        o.append(f"option.{opt}={nets_dir}/{f}.nnue")
    return " ".join(o)

def loadout_for(player):
    """returns the nets_dir + optional swap for a player name"""
    if player == "nQ":
        return (N, None)
    # player names like: bal0_r2_ep19  (balanced_l0 swapped to r2 checkpoint)
    parts = player.split("_", 1)
    expert_map = {"bal0":"balanced_l0","bal1":"balanced_l1","bal2":"balanced_l2",
                  "bal3":"balanced_l3"}
    expert = expert_map.get(parts[0], parts[0])
    ck_name = parts[1]  # e.g. "r2_ep19" or "r3_last"
    return (N, (expert, ck_name))

def spec_of(player):
    d, swap = loadout_for(player)
    return f"name={player} cmd={FORK} {opts_for(d, swap)}"

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
    # players: nQ + one checkpoint per (expert, round, epoch) — but keep the
    # count manageable: for each expert, take the earliest + last ckpt per round
    players = ["nQ"]
    for ckf in sorted(glob.glob(f"{CK}/*.nnue")):
        name = os.path.basename(ckf)[:-5]  # e.g. balanced_l0_r2_ep19
        parts = name.rsplit("_", 2)  # [expert, round, epoch]
        expert, rnd, ep = parts[0], parts[1], parts[2]
        emap = {"balanced_l0":"bal0","balanced_l1":"bal1","balanced_l2":"bal2",
                "balanced_l3":"bal3"}
        short = emap.get(expert, expert)
        # only take ep19 and last for each (expert, round) to limit field size
        if ep.startswith("ep") and ep != "ep19":
            continue
        players.append(f"{short}_{rnd}_{ep}")
    # dedup
    players = list(dict.fromkeys(players))
    if len(players) < 8:
        print(f"only {len(players)} players"); return
    print(f"{len(players)} players, {ROUNDS} rounds, st={ST}s", flush=True)
    for p in players:
        print(f"  {p}", flush=True)

    scores = {p: 0.0 for p in players}
    played = set()
    rng = random.Random(123)
    for rd in range(1, ROUNDS + 1):
        pairs = pair_round(players, scores, played, rng)
        running = []
        idx = 0
        CONC = 6
        while idx < len(pairs) or running:
            while idx < len(pairs) and len(running) < CONC:
                a, b = pairs[idx]
                played.add((a, b)); played.add((b, a))
                pgn = f"{SW}/r{rd:02d}_{a}_vs_{b}.pgn"
                white_a = rng.random() < 0.5
                pa = spec_of(a if white_a else b)
                pb = spec_of(b if white_a else a)
                pr = subprocess.Popen(["bash", "-c",
                    " ".join([FC, "-engine", pa, "-engine", pb,
                              "-each", "proto=uci", f"st={ST}", "timemargin=200",
                              "-rounds", "1",
                              "-pgnout", f"file={pgn}",
                              ">", pgn.replace(".pgn", ".log"), "2>&1"])])
                running.append((pr, a, b, white_a, pgn))
                idx += 1
            done = [r for r in running if r[0].poll() is not None]
            for pr, a, b, white_a, pgn in done:
                running.remove((pr, a, b, white_a, pgn))
                try:
                    res = re.search(r"\[Result .([^\"]+).", open(pgn).read()).group(1)
                except Exception:
                    res = "0-1"
                # first engine in the file is the white player
                wh = re.search(r"\[White .([^\"]+).", open(pgn).read()).group(1)
                # sa = a's score
                if wh == a:
                    sa = 1.0 if res == "1-0" else 0.0 if res == "0-1" else 0.5
                else:
                    sa = 0.0 if res == "1-0" else 1.0 if res == "0-1" else 0.5
                scores[a] += sa
                scores[b] += 1.0 - sa
            time.sleep(3)
        print(f"round {rd}: " + ", ".join(f"{p}={scores[p]:.1f}"
              for p in sorted(players, key=lambda q: -scores[q])[:5]), flush=True)
        json.dump({"scores": scores, "round": rd}, open(f"{SW}/standings.json", "w"), indent=1)
    final = sorted(players, key=lambda p: -scores[p])
    print("FINAL:", [(p, scores[p]) for p in final[:10]], flush=True)
    json.dump({"scores": scores, "final": final}, open(f"{SW}/standings.json", "w"), indent=1)

if __name__ == "__main__":
    main()
