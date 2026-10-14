#!/usr/bin/env python3
"""Insert OTB-segment and puzzle fens into the positions table so the
d20 cursor service serves them. Skips fens already present (miniatures)."""
import sqlite3
import time

con = sqlite3.connect("/srv/workspace/chess-active/store/games.db", timeout=300)
t0 = time.time()

n = con.execute(
    "insert or ignore into positions (fen, kind, ref)"
    " select fen, 'otb_seg', 'd20sweep' from seg_positions"
).rowcount
print(f"OTB fens inserted: {n:,} ({time.time()-t0:.0f}s)", flush=True)

n2 = con.execute(
    "insert or ignore into positions (fen, kind, ref)"
    " select fen, 'puzzle', 'd20sweep' from nongame"
    " where label glob '[-0-9]*'"
).rowcount
print(f"puzzle fens inserted: {n2:,} ({time.time()-time.time()+time.time()-t0:.0f}s)", flush=True)

con.commit()
total = con.execute("select count(*) from positions").fetchone()[0]
d20 = con.execute(
    "select count(*) from labels where engine='sf17' and depth=20"
).fetchone()[0]
print(f"positions table now: {total:,}")
print(f"d20-labeled: {d20:,}")
print(f"needing d20: {total-d20:,}")
