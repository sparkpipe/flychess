
s = open("/extnvme/active/matches/eval_worker.py").read()

# per-depth schema keys + depth-major passes
a = """    if (eg.get("schema") == 2 and eg.get("nplies") == sig
            and all(eg.get(f"evals_d{d}") is not None for d in DEPTHS)):
        return"""
b = """    if all(eg.get(f"schema_d{d}") == 2 and eg.get("nplies") == sig for d in DEPTHS):
        return"""
assert s.count(a) == 1, "done-check"
s = s.replace(a, b)

a = """    for d in DEPTHS:
        need = (eg.get("schema") != 2) or (eg.get("nplies") != sig) or (eg.get(f"evals_d{d}") is None)
        if not need:
            continue
        t0 = time.time()
        vals, pvs = eval_fens_parallel(fens, d)
        eg[f"evals_d{d}"] = vals
        eg[f"pv_d{d}"] = pvs
        eg["nplies"] = sig
        eg["schema"] = 2
        save_evals(m["tag"], ev)
        print(f"{m['tag']} game{gi} d{d}: {len(vals)} plies in {time.time()-t0:.0f}s", flush=True)"""
b = """    for d in DEPTHS:
        need = (eg.get(f"schema_d{d}") != 2) or (eg.get("nplies") != sig)
        if not need:
            continue
        t0 = time.time()
        vals, pvs = eval_fens_parallel(fens, d)
        eg[f"evals_d{d}"] = vals
        eg[f"pv_d{d}"] = pvs
        eg["nplies"] = sig
        eg[f"schema_d{d}"] = 2
        save_evals(m["tag"], ev)
        print(f"{m['tag']} game{gi} d{d}: {len(vals)} plies in {time.time()-t0:.0f}s", flush=True)"""
assert s.count(a) == 1, "eval-loop"
s = s.replace(a, b)

# depth-major order: iterate depth outer, games inner
a = """        for mi, m in enumerate(games):
            for gi in range(len(m["games"])):
                if focus and m["tag"] == focus.get("tag") and gi == focus.get("gi"):
                    order.insert(0, (mi, gi))
                else:
                    order.append((mi, gi))"""
b = """        for d in DEPTHS:
            for mi, m in enumerate(games):
                for gi in range(len(m["games"])):
                    g = m["games"][gi]
                    eg_pre = None
                    prio = 0 if (focus and m["tag"] == focus.get("tag") and gi == focus.get("gi")) else 1
                    order.append((prio, d, mi, gi))
            order.sort()"""
assert s.count(a) == 1, "order"
s = s.replace(a, b)

a = """        for mi, gi in order:
            m = games[mi]
            tag = m["tag"]
            if tag not in stores:
                stores[tag] = load_evals(tag)
                while len(stores[tag]["games"]) < len(m["games"]):
                    stores[tag]["games"].append({})
            work_game(m, gi, m["games"][gi], stores[tag])"""
b = """        for prio, d, mi, gi in order:
            m = games[mi]
            tag = m["tag"]
            if tag not in stores:
                stores[tag] = load_evals(tag)
                while len(stores[tag]["games"]) < len(m["games"]):
                    stores[tag]["games"].append({})
            work_game(m, gi, m["games"][gi], stores[tag], only_depth=d)"""
assert s.count(a) == 1, "dispatch"
s = s.replace(a, b)

a = """def work_game(m, gi, g, ev):"""
b = """def work_game(m, gi, g, ev, only_depth=None):"""
assert s.count(a) == 1, "signature"
s = s.replace(a, b)

a = """    for d in DEPTHS:
        need = (eg.get(f"schema_d{d}") != 2) or (eg.get("nplies") != sig)"""
b = """    for d in DEPTHS:
        if only_depth is not None and d != only_depth:
            continue
        need = (eg.get(f"schema_d{d}") != 2) or (eg.get("nplies") != sig)"""
assert s.count(a) == 1, "only-depth"
s = s.replace(a, b)

open("/extnvme/active/matches/eval_worker.py", "w").write(s)
import py_compile
py_compile.compile("/extnvme/active/matches/eval_worker.py", doraise=True)
print("depth-major worker patched, compiles")
