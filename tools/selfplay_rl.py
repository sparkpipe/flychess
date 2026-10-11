"""SELF-PLAY RL LOOP with Chess960 openings and curriculum TC advancement.

Architecture:
  1. Generate self-play games at current TC using Chess960 openings (960 positions, both colors = 1920 games)
  2. Extract expert features (evals + routed activations) for all game positions
  3. Train head on (features → game outcome) — RL signal, not eval matching
  4. Measure weight change L2 norm → advance TC when converged
  5. At each advancement: Elo checkpoint at 1s and 10s

Curriculum: 1s → 2s → 3s → 5s → 8s → 10s → 15s → 30s → 60s
Advancement: weight L2 change < threshold for 3 consecutive iterations
Validation: Elo spot test at each checkpoint
"""
import subprocess, os, sys, json, time, math, random, glob
import numpy as np
import torch
import chess
import chess.pgn

R = "/mnt/cold-raid6/chess-audit"
SP = R + "/selfplay_rl"
ARENA = "/srv/workspace/flychess/src/arena"
FORK = "/srv/workspace/flychess/src/Stockfish/src/stockfish"
HEAD_SCRIPT = "/srv/workspace/flychess/src/chess-lab/tools/train_head_v3.py"

TCS = [1, 2, 3, 5, 8, 10, 15, 30, 60]
CONVERGENCE_THRESHOLD = 1e-5  # L2 norm of weight change
CONVERGENCE_PATIENCE = 3      # consecutive iterations before advancing

EXPERTS = ["balanced_l0","balanced_l1","balanced_l2","balanced_l3",
           "nvb","nvr","bvr","rv2m","qvmat","oppb",
           "dvoretsky","exchanges","tactics"]

OPTS = " ".join([
    f"option.EvalFile={R}/nets/{EXPERTS[0]}.nnue",
    *[f"option.EvalFile{i+2}={R}/nets/{e}.nnue" for i, e in enumerate(EXPERTS[1:])],
])

def current_tc():
    """Read current TC from state file, or default to first."""
    state_f = f"{SP}/state.json"
    if os.path.exists(state_f):
        return json.load(open(state_f))
    return {"tc_index": 0, "iteration": 0, "convergence_streak": 0, "checkpoint": 0}

def save_state(state):
    os.makedirs(SP, exist_ok=True)
    json.dump(state, open(f"{SP}/state.json", "w"))

def generate_games(tc, n_positions=960):
    """Generate 1920 self-play games (960 Chess960 positions × both colors)."""
    ts = time.strftime("%Y%m%d_%H%M%S")
    game_dir = f"{SP}/games_{ts}"
    os.makedirs(game_dir, exist_ok=True)
    book = f"{R}/chess960_book.epd"

    n_rounds = n_positions // 2  # each round = 2 games (color-reversed)
    cmd = [
        f"{ARENA}/fastchess-linux-x86-64/fastchess",
        "-engine", "name=sp_a", f"cmd={ARENA}/fork_default.sh", OPTS,
        "-engine", "name=sp_b", f"cmd={ARENA}/fork_default.sh", OPTS,
        "-each", "proto=uci", f"st={tc}",
        "-rounds", str(n_rounds), "-repeat",
        "-openings", f"file={book}", "format=epd", "order=random",
        "-pgnout", f"file={game_dir}/games.pgn",
        "-concurrency", "4",
    ]
    print(f"  generating {n_rounds * 2} games at {tc}s/move...", flush=True)
    result = subprocess.run(cmd, capture_output=True, text=True, timeout=7200)
    if result.returncode != 0:
        print(f"  WARNING: fastchess returned {result.returncode}")
    return game_dir

def extract_game_positions(game_dir):
    """Extract (fen, move, result) from self-play PGNs."""
    positions = []
    for pgn_f in glob.glob(f"{game_dir}/*.pgn"):
        with open(pgn_f) as f:
            while True:
                game = chess.pgn.read_game(f)
                if game is None:
                    break
                result = game.headers.get("Result", "*")
                outcome = 1.0 if result == "1-0" else 0.0 if result == "0-1" else 0.5
                board = game.board()
                ply = 0
                for node in game.mainline():
                    positions.append({
                        "fen": board.fen(),
                        "move": node.move.uci(),
                        "outcome": outcome,
                        "ply": ply,
                    })
                    board.push(node.move)
                    ply += 1
    print(f"  extracted {len(positions)} positions from games", flush=True)
    return positions

def measure_weight_change(old_sd, new_sd):
    """L2 norm of parameter changes."""
    total = 0.0
    for key in old_sd:
        if key in new_sd:
            diff = (old_sd[key].float() - new_sd[key].float())
            total += diff.pow(2).sum().item()
    return math.sqrt(total)

def elo_checkpoint(tc_1s=True, tc_10s=True):
    """Run calibrated Elo tests at 1s and 10s to verify no regression."""
    results = {}
    for label, st in [("1s", 1), ("10s", 10)]:
        if (label == "1s" and not tc_1s) or (label == "10s" and not tc_10s):
            continue
        scores = []
        for elo in [2000, 2500, 2850]:
            cmd = [
                f"{ARENA}/fastchess-linux-x86-64/fastchess",
                "-engine", "name=checkpoint", f"cmd={ARENA}/fork_default.sh", OPTS,
                "-engine", "name=opponent", "cmd=/usr/games/stockfish",
                "option.UCI_LimitStrength=true", f"option.UCI_Elo={elo}",
                "-each", "proto=uci", f"st={st}", "-rounds", "3", "-repeat",
                "-openings", f"file={R}/3open.epd", "format=epd",
            ]
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=3600)
            # parse score from output
            for line in result.stdout.split("\n"):
                if "Points" in line:
                    scores.append(line.strip())
                    break
        results[label] = scores
    return results

def main():
    state = current_tc()
    tc = TCS[state["tc_index"]]

    print(f"=== RL LOOP: TC={tc}s, iteration={state['iteration']}, "
          f"streak={state['convergence_streak']} ===", flush=True)

    # Save current head weights for comparison
    head_path = f"{R}/head_v3.pt"
    if os.path.exists(head_path):
        old_sd = torch.load(head_path, map_location="cpu", weights_only=False)
    else:
        old_sd = None

    # 1. Generate self-play games
    game_dir = generate_games(tc)

    # 2. Extract positions with outcomes
    positions = extract_game_positions(game_dir)

    # 3. Prepare training data: expert features + outcome labels
    #    (reuse the existing eval extraction pipeline)
    print("  extracting expert features...", flush=True)
    # Write positions to a temp file for the extraction pipeline
    os.makedirs(f"{SP}/features", exist_ok=True)
    with open(f"{SP}/features/positions.txt", "w") as f:
        for p in positions:
            f.write(f"{p['fen']}|{p['move']}|{p['outcome']}|{p['ply']}|0|0|rl\n")

    # Extract expert evals for these positions
    extract_cmd = [
        "python3", "/srv/workspace/flychess/src/chess-lab/tools/extract_evals_rl.py",
        f"{SP}/features/positions.txt",
        f"{SP}/features/evals.npy",
    ]
    subprocess.run(extract_cmd, capture_output=True, text=True, timeout=7200)

    # 4. Train head on game outcomes
    print("  training head on RL outcomes...", flush=True)
    train_cmd = ["python3", HEAD_SCRIPT,
                 f"{SP}/features",  # features directory
                 "head_v3_rl",      # output name
                 "--rl-mode",       # use outcome labels, not eval labels
                 "--iterations", "50",  # train for 50 epochs per batch
    ]
    subprocess.run(train_cmd, capture_output=True, text=True, timeout=3600)

    # 5. Measure weight change
    if os.path.exists(head_path) and old_sd is not None:
        new_sd = torch.load(head_path, map_location="cpu", weights_only=False)
        weight_change = measure_weight_change(old_sd, new_sd)
        print(f"  weight change L2: {weight_change:.6f}", flush=True)

        if weight_change < CONVERGENCE_THRESHOLD:
            state["convergence_streak"] += 1
            if state["convergence_streak"] >= CONVERGENCE_PATIENCE:
                # Advance to next TC
                if state["tc_index"] < len(TCS) - 1:
                    state["tc_index"] += 1
                    state["convergence_streak"] = 0
                    state["checkpoint"] += 1
                    print(f"  *** ADVANCING to TC={TCS[state['tc_index']]}s ***", flush=True)

                    # Elo checkpoint
                    print("  running Elo checkpoint...", flush=True)
                    results = elo_checkpoint()
                    json.dump(results, open(f"{SP}/checkpoint_{state['checkpoint']}.json", "w"))
        else:
            state["convergence_streak"] = 0

    state["iteration"] += 1
    save_state(state)
    print(f"  iteration {state['iteration']} complete", flush=True)


if __name__ == "__main__":
    main()
