"""Build ranked-endgame evaluation sets from the Lichess puzzle DB
(operator directive: "endgame training puzzles of ranked difficulty").

Filters Themes for endgame-tagged puzzles, buckets by Rating band,
samples per band, and emits question rows: fen AFTER the opponent's
setup move, solution = the first solution move (Moves[1]).

Output: tbpools/lichess_endgame_eval.jsonl
  {id, fen, solution, rating, themes, band}
"""
import csv
import json
import random

CSV = "/home/spec/chess-lab/puzzles/lichess_db_puzzle.csv"
OUT = "/home/spec/chess-lab/tbpools/lichess_endgame_eval.jsonl"
PER_BAND = 250
BANDS = [(0, 1200), (1200, 1600), (1600, 2000), (2000, 4000)]

rng = random.Random(5)
seen = {}
with open(CSV, newline="", encoding="utf-8") as f:
    rd = csv.DictReader(f)
    for row in rd:
        themes = row["Themes"]
        if "endgame" not in themes.split():
            continue
        r = int(row["Rating"])
        band = None
        for lo, hi in BANDS:
            if lo <= r < hi:
                band = f"{lo}-{hi}"
                break
        if band is None:
            continue
        ctr = seen.setdefault(band, [])
        if len(ctr) < PER_BAND * 3:
            ctr.append(row)

rows_out = []
import chess
for band, rows in seen.items():
    rng.shuffle(rows)
    used = 0
    for row in rows:
        if used >= PER_BAND:
            break
        try:
            b = chess.Board(row["FEN"])
            mvs = row["Moves"].split()
            if len(mvs) < 2:
                continue
            b.push(chess.Move.from_uci(mvs[0]))   # opponent setup
            if not b.is_valid() or b.king(chess.WHITE) is None \
                    or b.king(chess.BLACK) is None or b.is_game_over():
                continue
            sol = chess.Move.from_uci(mvs[1])
            if sol not in b.legal_moves:
                continue
            rows_out.append({"id": row["PuzzleId"], "fen": b.fen(),
                             "solution": sol.uci(),
                             "rating": int(row["Rating"]),
                             "themes": row["Themes"], "band": band})
            used += 1
        except Exception:
            continue
with open(OUT, "w") as f:
    for r in rows_out:
        f.write(json.dumps(r) + "\n")
from collections import Counter
print(json.dumps({"total": len(rows_out),
                  "bands": dict(Counter(r["band"]
                                        for r in rows_out))}))
