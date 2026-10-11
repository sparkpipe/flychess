#!/usr/bin/env python3
"""d1-12 eval-depth labeler for the piece_down experiment.
One SF17 search per position (go depth 12) — the info stream carries the
depth 1..12 verdicts. All 12 rows go to the labeling service via HTTP
(operator shard architecture: HTTP -> local append file -> serial ingest).
Stores cp verbatim stm-pov (the label LAW). Resumable via .done file."""
import json, os, subprocess, sys, time, urllib.request

SF = os.environ.get("SF", "/srv/workspace/flychess/src/sf17/src/stockfish")
SVC = os.environ.get("SVC", "http://127.0.0.1:8123")
NODE = os.environ.get("NODE", "pddepth-" + str(os.getpid() % 1000))
shard = sys.argv[1]
donef = shard + ".done"

done = set()
if os.path.exists(donef):
    done = {l.strip() for l in open(donef) if l.strip()}

p = subprocess.Popen([SF], stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                     stderr=subprocess.DEVNULL, text=True, bufsize=1)
p.stdin.write("uci\nsetoption name Threads value 1\nisready\n"); p.stdin.flush()
while "readyok" not in p.stdout.readline(): pass

def post(rows):
    req = urllib.request.Request(SVC + "/submit",
        data=json.dumps({"node": NODE, "results": rows}).encode(),
        headers={"Content-Type": "application/json"})
    return json.loads(urllib.request.urlopen(req, timeout=300).read())

buf, t0, cnt, postn = [], time.time(), 0, 0
with open(shard) as f, open(donef, "a") as donefh:
    for line in f:
        fen = line.rstrip("\n").rsplit("\t", 1)[0].strip()
        if not fen or fen in done: continue
        try:
            p.stdin.write(f"ucinewgame\nposition fen {fen}\ngo depth 12\n"); p.stdin.flush()
        except BrokenPipeError:
            sys.exit(f"{NODE}: sf died")
        scores = {}
        while True:
            ln = p.stdout.readline()
            if not ln: sys.exit(f"{NODE}: sf died")
            if ln.startswith("info") and not ln.startswith("info string") and " score " in ln:
                t = ln.split()
                try: d = int(t[t.index("depth") + 1])
                except (ValueError, IndexError): continue
                for i, x in enumerate(t):
                    if i and t[i-1] == "score" and x in ("cp", "mate"):
                        try: v = int(t[i+1])
                        except (ValueError, IndexError): break
                        scores[d] = ((10000 + min(abs(v), 900)) * (1 if v > 0 else -1)
                                     if x == "mate" else v)
                        break
            elif ln.startswith("bestmove"):
                break
        for d in range(1, 13):
            if d in scores:
                buf.append({"fen": fen, "cp": scores[d], "depth": d})
        donefh.write(fen + "\n"); donefh.flush()
        cnt += 1
        if len(buf) >= 600:
            post(buf); postn += len(buf); buf = []
            el = time.time() - t0
            print(f"{NODE} {cnt:,} done, {postn:,} rows posted ({cnt/el:.1f} pos/s)", flush=True)
if buf: post(buf); postn += len(buf)
print(f"{NODE} SHARD COMPLETE {cnt:,} positions, {postn:,} rows", flush=True)
