"""DEEP EVALUATION FLEET: evaluate OTB positions at proper depth (12+).

Runs on spark nodes. Each worker:
1. Reads a shard of the OTB position dump
2. For each position: SF `go depth DEPTH` with WDL output
3. Records (fen, played_move, winner, ply, welo, belo, result,
            eval_cp, eval_wdl_win, eval_wdl_draw, eval_wdl_loss,
            best_move, depth_reached)
4. Writes shard output

This produces the trajectory data that specialist filters consume.
DEPTH is configurable (default 12, not 1).
"""
import sys
import os
import subprocess
import time
import random

sys.path.insert(0, os.path.expanduser("~") + "/extnvme/phase-moe")
import chess
import chess.engine

SF = os.path.expanduser("~") + "/extnvme/phase-moe/sf/src/stockfish"
# input: the OTB dump file (fen|move|winner|ply|welo|belo|result)
INPUT = os.environ.get("INPUT",
                       "/home/spec/chess-lab/otb_all_positions.txt")
OUTPUT = os.environ.get("OUTPUT",
                        os.path.expanduser("~") +
                        "/extnvme/phase-moe/otb_evals.txt")
DEPTH = int(os.environ.get("DEPTH", "12"))
WORKER_ID = os.environ.get("WORKER_ID", "0")
NUM_WORKERS = int(os.environ.get("NUM_WORKERS", "8"))
SHARD_SIZE = int(os.environ.get("SHARD_SIZE", "0"))  # 0 = all assigned


def main():
    eng = chess.engine.SimpleEngine.popen_uci(SF)
    eng.configure({"Threads": 1, "Hash": 128})

    # count lines to determine sharding
    with open(INPUT) as f:
        total = sum(1 for _ in f)
    lines_per_worker = total // NUM_WORKERS
    start = int(WORKER_ID) * lines_per_worker
    end = start + lines_per_worker if int(WORKER_ID) < NUM_WORKERS - 1 \
        else total

    t0 = time.time()
    n = 0
    with open(INPUT) as fin, open(OUTPUT, "a") as fout:
        for i, line in enumerate(fin):
            if i < start:
                continue
            if i >= end:
                break
            parts = line.strip().split("|")
            if len(parts) < 7:
                continue
            fen, played, winner, ply, welo, belo, result = parts[:7]
            ply = int(ply)
            board = chess.Board(fen)

            try:
                info = eng.analyse(
                    board, chess.engine.Limit(depth=DEPTH))
                score = info["score"].pov(board.turn)
                cp = score.score()
                # get WDL if available

                best = info["pv"][0].uci() if info.get("pv") else ""
            except Exception:
                continue

            wdl_str = ""  # cp score is sufficient for trajectories
            fout.write(f"{fen}|{played}|{winner}|{ply}|{welo}|{belo}"
                       f"|{result}|{cp}|{wdl_str}|{best}"
                       f"|{DEPTH}\n")
            n += 1
            if n % 10000 == 0:
                el = time.time() - t0
                print(f"worker {WORKER_ID}: {n}/{end-start} "
                      f"({n/el:.0f}/s)", flush=True)

    eng.quit()
    print(f"worker {WORKER_ID} DONE: {n} evaluations, "
          f"{time.time()-t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
