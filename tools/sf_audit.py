"""SF sanity audit of every DEGM position (operator order): re-grade
every legal move of every row with Stockfish at fixed depth and check
the row's cat / best / children against it.

Writes tbpools/sf_audit.jsonl (per-row verdicts) + sf_audit_summary.json.
Workers each run their own SF (1 thread, 64MB hash) — polite to the
training sharing this box.
"""
import os
import sys
import json
import glob
import chess
import chess.engine
from multiprocessing import Pool

sys.path.insert(0, "/home/spec/chess-lab/tools")
from degm_pools import band, SF, DEPTH

POOLS = "/home/spec/chess-lab/tbpools"
OUT = os.path.join(POOLS, "sf_audit.jsonl")
SUMMARY = "/home/spec/chess-lab/sf_audit_summary.json"
WORKERS = 8
RANK = {"loss": 0, "cursed_loss": 1, "draw": 2,
        "cursed_win": 3, "win": 4}


def rows():
    for f in sorted(glob.glob(os.path.join(POOLS, "DEGM_Ch[0-9]*.jsonl"))):
        if "_s" in os.path.basename(f):
            continue
        for line in open(f):
            e = json.loads(line)
            e["_src"] = os.path.basename(f).replace(
                "DEGM_", "").replace(".jsonl", "")
            yield e


ENG = None


def init_worker():
    global ENG
    ENG = chess.engine.SimpleEngine.popen_uci(SF)
    ENG.configure({"Threads": 1, "Hash": 64})


def audit_one(e):
    b = chess.Board(e["fen"])
    if not b.is_valid() or b.king(chess.WHITE) is None \
            or b.king(chess.BLACK) is None:
        return {"fen": e["fen"], "pool": e.get("pool"),
                "verdict": "illegal_fen"}
    eng = ENG
    try:
        mvs = list(b.legal_moves)
        infos = eng.analyse(b, chess.engine.Limit(depth=DEPTH),
                            multipv=len(mvs))
        sf = {}
        for info in infos:
            u = info["pv"][0].uci()
            sf[u] = {"band": band(info["score"].relative),
                     "cp": info["score"].relative.score()
                     if not info["score"].relative.is_mate()
                     else None,
                     "mate": info["score"].relative.mate()}
        sf_class = max((v["band"] for v in sf.values()),
                       key=lambda x: RANK[x])
        top = {u for u, v in sf.items() if v["band"] == sf_class}
        old = e.get("children", {})
        agree = 0
        n = 0
        for u, c in old.items():
            if u in sf:
                n += 1
                agree += int(c.get("cat") == sf[u]["band"])
        rec = {"fen": e["fen"], "pool": e.get("pool"),
               "pieces": e.get("pieces"),
               "cat": e["cat"], "sf_class": sf_class,
               "best": e.get("best"),
               "sf_top": sorted(top),
               "best_in_top": e.get("best") in top,
               "children_agree": round(agree / n, 3) if n else None,
               "verdict": "sane" if (
                   sf_class == e["cat"] and e.get("best") in top)
               else "insane"}
        return rec
    except Exception as ex:
        return {"fen": e["fen"], "pool": e.get("pool"),
                "verdict": f"error:{type(ex).__name__}"}


def main():
    all_rows = list(rows())
    print(f"auditing {len(all_rows)} positions "
          f"with {WORKERS} SF workers", flush=True)
    with Pool(WORKERS, initializer=init_worker) as p, \
            open(OUT, "w") as f:
        for i, rec in enumerate(p.imap(audit_one, all_rows, chunksize=4)):
            f.write(json.dumps(rec) + "\n")
            if i % 100 == 0:
                print(f"[{i}]", flush=True)
    from collections import Counter
    recs = [json.loads(l) for l in open(OUT)]
    verd = Counter(r["verdict"] for r in recs)
    per_pool = {}
    for r in recs:
        k = r.get("pool", "?")
        d = per_pool.setdefault(k, Counter())
        d[r["verdict"]] += 1
        if r["verdict"] == "insane":
            if r["cat"] != r["sf_class"]:
                d["cat_mismatch"] += 1
            if not r["best_in_top"]:
                d["best_not_top"] += 1
    summary = {"total": len(recs), "verdicts": dict(verd),
               "per_pool": {k: dict(v) for k, v in per_pool.items()}}
    json.dump(summary, open(SUMMARY, "w"), indent=1)
    print(json.dumps(summary["verdicts"]), flush=True)
    print("SF-AUDIT-COMPLETE", flush=True)


if __name__ == "__main__":
    main()
