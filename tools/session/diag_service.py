import json
import sqlite3
import time
import urllib.request

t0 = time.time()
req = urllib.request.Request(
    "http://127.0.0.1:8123/allocate",
    data=json.dumps({"node": "rattest2", "engine": "sf17",
                     "depth": 20, "n": 100}).encode(),
    headers={"Content-Type": "application/json"})
try:
    r = json.load(urllib.request.urlopen(req, timeout=180))
    print(f"allocate: {r.get('allocated', 0)} in {time.time()-t0:.1f}s")
except Exception as e:
    print(f"allocate FAILED after {time.time()-t0:.1f}s: {e}")

# check allocations table size — the expired-lease reclaim may be the culprit
c = sqlite3.connect("file:/srv/workspace/chess-active/store/games.db?mode=ro",
                    uri=True)
n = c.execute("select count(*) from allocations").fetchone()[0]
o = c.execute(
    "select count(*) from allocations where state='open'").fetchone()[0]
print(f"allocations: {n:,} total, {o:,} open")
