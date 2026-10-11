#!/bin/bash
# every-ply experiment telemetry: progress, rate, ingest backlog, spark CPU
E=/srv/workspace/chess-active/pd-depthexp
python3 - << "PYE"
import sqlite3, time, os, glob
con = sqlite3.connect("file:/srv/workspace/chess-active/store/games.db?mode=ro", uri=True)
q = lambda s: con.execute(s).fetchone()[0]
tot = 2735804
d1 = q("select count(*) from labels where engine=\"sf17\" and depth=1")
d11 = q("select count(*) from labels where engine=\"sf17\" and depth=11")
nodes = con.execute("select by_node, count(*) from labels where by_node like \"pddepth-%\" and depth=1 group by by_node order by 2 desc").fetchall()
backlog = sum(os.path.getsize(f) for f in glob.glob("/srv/workspace/chess-active/shards/pddepth*.jsonl")) / 1e6
print(f"every-ply: {d1:,}/{tot:,} positions ({100*d1/tot:.1f}%)  d11={d11:,}  ingest backlog={backlog:.0f}MB")
print("top nodes:", ", ".join(f"{n}={c//1000}k" for n, c in nodes[:8]))
PYE
echo "spark SF procs/busy%:"; for s in spark0 spark1 spark2 spark3 spark4 spark5 spark6 spark7 spark8 spark9 sparka sparkb sparkc sparkd sparke sparkf; do R=$(ssh -o BatchMode=yes -o ConnectTimeout=5 $s "ps aux | grep [s]f17_arm | awk \"{print \\\$3}\" | sort -rn | head -1; pgrep -f sf17_arm | wc -l" 2>/dev/null | tr "\n" "/"); echo -n "$s:${R:-x} "; done; echo
