"""Lichess puzzle DB -> stage-7 tactic pools (mateIn1, mateIn2 first).

CSV: PuzzleId,FEN,Moves,Rating,RatingDeviation,Popularity,NbPlays,
     Themes,GameUrl,OpeningTags,DailyDate  (moves in UCI, space-sep;
     the FIRST move is the opponent's setup, the SECOND is ours)

Pool rows use the stage-6 schema the trainer already consumes:
{fen, cat, best, children, pool}. For mate puzzles: cat=win,
best = our first solution move, children = {best: {cat: "loss", dtz: 1}}
(opponent-perspective convention of the TB pools).
"""
import csv
import json
import os
import sys
import random
import collections

CSV = "/home/spec/chess-lab/puzzles/lichess_db_puzzle.csv"
OUT = "/home/spec/chess-lab/tbpools"
CAPS = {"mateIn1": 60000, "mateIn2": 60000}


def main():
    random.seed(20260919)
    kept = {k: [] for k in CAPS}
    with open(CSV, newline="", encoding="utf-8") as f:
        rd = csv.DictReader(f)
        for row in rd:
            themes = row["Themes"]
            for k in CAPS:
                if k in themes and len(kept[k]) < CAPS[k]:
                    kept[k].append(row)
                    break
    for k, rows in kept.items():
        out = []
        for row in rows:
            mv = row["Moves"].split()
            if len(mv) < 2:
                continue
            best = mv[1]
            out.append({"fen": row["FEN"], "cat": "win", "best": best,
                        "children": {best: {"cat": "loss", "dtz": 1}},
                        "pool": k, "rating": int(row["Rating"])})
        with open(os.path.join(OUT, f"{k}.jsonl"), "w") as f:
            for r in out:
                f.write(json.dumps(r) + "\n")
        print(f"POOL {k}: {len(out)} positions", flush=True)
    print("PUZZLE-POOLS-COMPLETE", flush=True)


if __name__ == "__main__":
    main()
