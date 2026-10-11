"""PUZZLE SHARE per expert — how much of each train bin came from puzzles.

Source discrimination: puzzle positions' played move = the puzzle line's next
move; game positions' = the game's played move. Cheaper: we recorded the
puzzle append counts in puzzle_pack.log. Read that + bin sizes.
"""
import os, re, glob

log = open("/mnt/cold-raid6/chess-audit/wp_fit/puzzle_pack.log", errors="replace").read()
appended = {}
for m in re.finditer(r"^\s+(\S+)\s+(train|val): \+([\d,]+)", log, re.M):
    expert, tag, n = m.group(1), m.group(2), int(m.group(3).replace(",", ""))
    if tag == "train":
        appended[expert] = n

print("%-24s %12s %12s %7s" % ("expert", "puzzles", "total_train", "pct"))
for b in sorted(glob.glob("/mnt/cold-raid6/chess-audit/train23/main/*.train.bin")):
    expert = os.path.basename(b)[:-10]
    total = os.path.getsize(b) // 40
    pz = appended.get(expert, 0)
    print("%-24s %12s %12s %6.1f%%" % (expert, format(pz, ","),
          format(total, ","), 100 * pz / max(total, 1)))
