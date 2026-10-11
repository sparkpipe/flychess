import re

p = "/extnvme/active/matches/eval_worker.py"
s = open(p).read()

old_loop_start = s.index("while True:")
old_loop = s[old_loop_start:]
new_loop = '''def load_focus():
    try:
        return json.load(open(D + "/focus.json"))
    except Exception:
        return None

def work_game(m, gi, g, ev):
    eg = ev["games"][gi]
    sig = len(g.get("fens", []))
    live = g.get("open", False) or g.get("result") == "*"
    if live:
        have = eg.get("evals_d12") or []
        fens = g["fens"][:-1] if sig > 1 else g["fens"]
        if len(have) < len(fens):
            newf = fens[len(have):]
            vals = [None] * len(newf)
            chunk = max(1, (len(newf) + NPROC - 1) // NPROC)
            threads = []
            for i in range(0, len(newf), chunk):
                part = newf[i:i+chunk]
                t = threading.Thread(target=eval_batch, args=(part, 12, vals, i, None, []))
                t.start(); threads.append(t)
            for t in threads:
                t.join()
            eg["evals_d12"] = have + vals
            eg["nplies_live"] = sig
            save_evals(m["tag"], ev)
            print(f"LIVE {m['tag']} game{gi} d12 +{len(vals)} plies", flush=True)
        return
    if eg.get("nplies") == sig and all(eg.get(f"evals_d{d}") is not None for d in DEPTHS):
        return
    for d in DEPTHS:
        need = (eg.get("nplies") != sig) or (eg.get(f"evals_d{d}") is None)
        if not need:
            continue
        t0 = time.time()
        vals = evaluate_game(g, d)
        eg[f"evals_d{d}"] = vals
        eg["nplies"] = sig
        save_evals(m["tag"], ev)
        print(f"{m['tag']} game{gi} d{d}: {len(vals)} plies in {time.time()-t0:.0f}s", flush=True)

while True:
    games = load_games()
    focus = load_focus()
    stores = {}
    order = []
    for mi, m in enumerate(games):
        for gi in range(len(m["games"])):
            if focus and m["tag"] == focus.get("tag") and gi == focus.get("gi"):
                order.insert(0, (mi, gi))
            else:
                order.append((mi, gi))
    for mi, gi in order:
        m = games[mi]
        tag = m["tag"]
        if tag not in stores:
            stores[tag] = load_evals(tag)
            while len(stores[tag]["games"]) < len(m["games"]):
                stores[tag]["games"].append({})
        work_game(m, gi, m["games"][gi], stores[tag])
    time.sleep(15)
'''
s = s[:old_loop_start] + new_loop
open(p, "w").write(s)
print("worker patched: focus-first + live incremental d12")
