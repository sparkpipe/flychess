"""D20 COVERAGE VERIFICATION — do training positions have d20 evals?

For each engine's per-expert train bins: sample N records, decode fen,
look up in (a) depth-DB shards (d20), (b) store plies (d12 source).
Reports: % training positions with d20 available, % in store.
"""
import glob, os, struct, sys, random, sqlite3
sys.path.insert(0, "/srv/workspace/flychess/src/chess-lab/tools")
from audit_packer import unpack_sfen

random.seed(20261005)
BINS = "/mnt/cold-raid6/chess-audit/train23"
DB = "/mnt/cold-raid6/chess-audit/wp_fit/store/games.db"
N_PER = 4000

# d20 lookup: all depth-db shards on the box + spark-local? box has old shards
d20 = set()
for f in glob.glob("/mnt/cold-raid6/chess-audit/depth_db/sparks/*/*.tsv.*") + \
         glob.glob("/mnt/cold-raid6/chess-audit/depth_db/shards/w*.tsv"):
    try:
        for line in open(f, errors="replace"):
            d20.add(line.split("\t")[0].split(" ")[0])
    except Exception:
        pass
print(f"d20 shard keys on box: {len(d20):,}")

con = sqlite3.connect(f"file:{DB}?mode=ro", uri=True)
store = con.execute("SELECT COUNT(DISTINCT fen_key) FROM (SELECT substr(fen,1,instr(fen||' ',' ')-1) AS fen_key FROM plies LIMIT 1)").fetchone()

for eng in ("main", "anti"):
    tot_n = tot_d20 = tot_store = 0
    for b in sorted(glob.glob(f"{BINS}/{eng}/*.train.bin")):
        expert = os.path.basename(b).replace(".train.bin", "")
        raw = open(b, "rb").read()
        nrec = len(raw) // 40
        idxs = random.sample(range(nrec), min(N_PER, nrec))
        n = d20hit = sthit = 0
        for i in idxs:
            rec = raw[i * 40:(i + 1) * 40]
            try:
                board = unpack_sfen(rec[:32])[0]
            except Exception:
                continue
            key = board.fen().split(" ")[0]
            n += 1
            if key in d20:
                d20hit += 1
            r = con.execute(
                "SELECT cp FROM plies WHERE fen LIKE ? || '%'", (key,)).fetchone()
            if r is not None:
                sthit += 1
        tot_n += n; tot_d20 += d20hit; tot_store += sthit
        if n:
            print(f"{eng}/{expert}: n={n:,} d20={100*d20hit/n:.1f}% store={100*sthit/n:.1f}%")
    print(f"[{eng}] TOTAL: n={tot_n:,}  d20-coverage={100*tot_d20/max(tot_n,1):.1f}%  in-store={100*tot_store/max(tot_n,1):.1f}%\n")
