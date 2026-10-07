"""CAP-50-PER-EXPERT PLAN — trim puzzle records to <=50% of each expert's
train bin. Emits a cull list (byte-range offsets per expert bin) WITHOUT
touching the bins yet — the operator approves counts first.

Puzzle records are the LAST appended block in each .train.bin (game records
were packed first by assemble23, puzzles appended after by puzzle_pack).
Cull = take the first floor(game_n) puzzle records, drop the rest.
BUT ordering isn't guaranteed per-source within the append... puzzle_pack
appended ALL its records at file end, contiguous. So the boundary =
file size before append (recorded? no). Instead: identify puzzle records
by re-deriving: the puzzle_pack wrote records for fens in puzzle evals with
solution moves. We re-scan the tail block against the puzzle-eval fen set
to find the exact boundary (first record from the tail whose position is in
the puzzle set = start of puzzle block, since game block never contains them
... not strictly true — a puzzle position can equal a game position. So
boundary = max start such that >=99% of records after it are puzzle-set
members. Simple robust scan from the end.
"""
import os, glob, struct, sys
sys.path.insert(0, "/srv/workspace/flychess/src/chess-lab/tools")
from audit_packer import unpack_sfen

EVAL_DIR = os.path.expanduser("~/backlog_shards/puzzle_eval")
BINS = "/mnt/cold-raid6/chess-audit/train23/main"

puzzle_keys = set()
import subprocess
for shard in glob.glob(f"{EVAL_DIR}/w*.tsv"):
    for line in open(shard, errors="replace"):
        puzzle_keys.add(line.split("\t")[0].split(" ")[0])
print(f"puzzle eval keys: {len(puzzle_keys):,}")

plan = {}
for b in sorted(glob.glob(f"{BINS}/*.train.bin")):
    expert = os.path.basename(b)[:-10]
    raw = open(b, "rb").read()
    n = len(raw) // 40
    # scan from end: count how many trailing records are puzzle-set
    i = n - 1
    while i >= 0:
        try:
            board = unpack_sfen(raw[i*40:i*40+32])[0]
            k = board.fen().split(" ")[0]
        except Exception:
            i -= 1
            continue
        if k in puzzle_keys:
            i -= 1
        else:
            break
    puzzle_n = n - 1 - i
    game_n = i + 1
    keep = min(puzzle_n, game_n) if game_n > 0 else 0
    drop = puzzle_n - keep
    plan[expert] = (game_n, puzzle_n, keep, drop, n)
    print("%-22s game=%9s puzzle=%9s keep=%9s drop=%9s (total %s)" % (
        expert, format(game_n, ","), format(puzzle_n, ","),
        format(keep, ","), format(drop, ","), format(n, ",")))

import json
json.dump(plan, open("/mnt/cold-raid6/chess-audit/train23/cap50_plan.json", "w"), indent=1)
print("plan -> cap50_plan.json (NO bins modified)")
