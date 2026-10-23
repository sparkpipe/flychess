#!/usr/bin/env python3
"""Export (read-only, hub) the labels needed for the COMBINED training set:
old general3 sources (otb_seg, puzzle, gambit) + new aj_sr, plus the plies
of the clean-short-win games — each with cp at depth 12 and depth 20.

Usage: export_combined_labels.py --db games.db --clean-wins JSON --out-prefix P
  writes P_positions.tsv  (kind<TAB>fen<TAB>cp12<TAB>cp20)
         P_plies.tsv      (gid<TAB>ply<TAB>fen<TAB>cp12<TAB>cp20)
"""
import argparse
import json
import sqlite3

ap = argparse.ArgumentParser()
ap.add_argument("--db", required=True)
ap.add_argument("--clean-wins", required=True)
ap.add_argument("--out-prefix", required=True)
a = ap.parse_args()

con = sqlite3.connect(f"file:{a.db}?mode=ro", uri=True, timeout=600)

out = open(f"{a.out_prefix}_positions.tsv", "w")
n = 0
for fen, kind, c12, c20 in con.execute(
        "select p.fen, p.kind, l12.cp, l20.cp from positions p"
        " join labels l12 on l12.fen = p.fen and l12.engine='sf17' and l12.depth=12"
        " join labels l20 on l20.fen = p.fen and l20.engine='sf17' and l20.depth=20"
        " where p.kind in ('otb_seg','puzzle','gambit','aj_sr')"):
    out.write(f"{kind}\t{fen}\t{c12}\t{c20}\n")
    n += 1
    if n % 4000000 == 0:
        print(f"  {n:,}", flush=True)
out.close()
print(f"positions exported: {n:,}", flush=True)

clean = json.load(open(a.clean_wins))
gids = sorted(set(int(g["gid"]) for g in clean))
q = ",".join("?" * len(gids))
out = open(f"{a.out_prefix}_plies.tsv", "w")
n = 0
for gid, ply, fen, c12, c20 in con.execute(
        f"select pl.gid, pl.ply, pl.fen, l12.cp, l20.cp from plies pl"
        f" join labels l12 on l12.fen = pl.fen and l12.engine='sf17' and l12.depth=12"
        f" join labels l20 on l20.fen = pl.fen and l20.engine='sf17' and l20.depth=20"
        f" where pl.gid in ({q})", gids):
    out.write(f"{gid}\t{ply}\t{fen}\t{c12}\t{c20}\n")
    n += 1
out.close()
print(f"clean-win plies exported: {n:,}", flush=True)
