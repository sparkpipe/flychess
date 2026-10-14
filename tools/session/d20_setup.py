#!/usr/bin/env python3
"""Count fens needing insertion into positions for the d20 sweep,
estimate disk impact."""
import sqlite3

con = sqlite3.connect("file:/srv/workspace/chess-active/store/games.db?mode=ro",
                      uri=True)
seg = con.execute(
    "select count(*) from seg_positions s"
    " where not exists (select 1 from positions p where p.fen=s.fen)"
).fetchone()[0]
pz = con.execute(
    "select count(*) from nongame n"
    " where n.label glob '[-0-9]*'"
    " and not exists (select 1 from positions p where p.fen=n.fen)"
).fetchone()[0]
print(f"new OTB fens: {seg:,}")
print(f"new puzzle fens: {pz:,}")
print(f"total to insert: {seg+pz:,}")
print(f"estimated positions growth: ~{(seg+pz)*150/1e9:.1f} GB")
print(f"estimated labels growth (20 depths): ~{(seg+pz)*20*80/1e9:.1f} GB")
print(f"total estimated: ~{(seg+pz)*150/1e9 + (seg+pz)*20*80/1e9:.1f} GB")
