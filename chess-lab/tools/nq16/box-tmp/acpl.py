import json, sys, glob, re, urllib.request

def analyze(pair, g):
    try:
        return json.load(urllib.request.urlopen(
            f"http://localhost:8077/api/analyze?p={pair}&g={g}", timeout=300))
    except Exception:
        return {"error": "?"}

tot = {}
for pgn in sorted(glob.glob("/mnt/cold-raid6/chess-audit/tourney8/*.pgn")):
    pair = pgn.split("/")[-1][:-4]
    txt = open(pgn).read()
    for gi, gtxt in enumerate(txt.split("[Event")[1:]):
        res = re.search(r"\[Result .([^\"]+).", gtxt)
        if not res or res.group(1) == "*":
            continue
        d = analyze(pair, gi)
        if "error" in d or not d.get("plies"):
            continue
        evs = d["evals"]
        for k in range(d["plies"]):
            b, a = evs[k]["cp"], evs[k+1]["cp"]
            if b is None or a is None:
                continue
            mover = 1 if k % 2 == 0 else -1
            who = "White" if k % 2 == 0 else "Black"
            wh = re.search(r"\[White .(\w+).", gtxt).group(1)
            bl = re.search(r"\[Black .(\w+).", gtxt).group(1)
            player = wh if k % 2 == 0 else bl
            loss = max(0, b*mover - a*mover)
            tot.setdefault(player, []).append(loss)
for p, losses in sorted(tot.items(), key=lambda kv: sum(kv[1])/len(kv[1])):
    print(f"{p:6s} avg self-harm {sum(losses)/len(losses)/100:5.2f} pawns/move  (n={len(losses)})")
