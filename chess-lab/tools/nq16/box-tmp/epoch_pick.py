"""EPOCH-PICK — best checkpoint per main expert, from collected metrics.
Sources: (a) box metrics.csv (3 experts with 16 levels); (b) named_all2.csv
corpus for the rest (corr at epoch). Pick rule: highest val_corr; ties ->
later epoch. Output: picks.tsv (expert, chosen_epoch, corr) + copy plan for
the engine dir. Experts with only -last: pick last.
"""
import csv, glob, os, shutil, re

NETS = "/mnt/cold-raid6/chess-audit/runs23/main/nets"
OUT = "/mnt/cold-raid6/chess-audit/engine23"
os.makedirs(OUT, exist_ok=True)

EXPERTS = ["tb", "mvr", "rv2m", "qvmat", "n2v2", "pd_down", "pd_up", "oppb",
           "dv_rend", "dv_QRend", "dv_qend", "dv_core", "nvb",
           "op_gambiteer", "op_acceptor", "op_even_l0", "op_even_l1", "op_even_l2+",
           "mg_unsafe_king", "mg_safe_both_same", "mg_safe_my_castled",
           "mg_safe_uncastled", "mg_safe_other_castled"]

# corr-at-epoch from named metrics corpus
corr_by = {}
cur = None
lines = open("/tmp/named_all2.csv", errors="replace").readlines()
i = 0
runs = []
while i < len(lines):
    if lines[i].startswith("RUN "):
        if cur:
            runs.append((cur, rows_))
        cur = lines[i][4:].strip()
        rows_ = []
        i += 1
        if i < len(lines) and lines[i].startswith("epoch"):
            i += 1
        continue
    rows_.append(lines[i])
    i += 1
if cur:
    runs.append((cur, rows_))
for name, rows_ in runs:
    if not name.startswith("main/"):
        continue
    exp = name.split("/")[1]
    per = {}
    for r in rows_:
        p = r.rstrip("\n").split(",")
        if len(p) < 8 or not p[0] or not p[7]:
            continue
        try:
            per[int(float(p[0]))] = float(p[7])
        except ValueError:
            continue
    if per:
        best_ep = max(per, key=lambda e: (per[e], e))
        prev = corr_by.get(exp)
        if prev is None or per[best_ep] > prev[1]:
            corr_by[exp] = (best_ep, per[best_ep])

# box metrics for the 3 rich experts
for d in glob.glob("/mnt/cold-raid6/chess-audit/runs23/main/*/lightning_logs/version_*/metrics.csv"):
    exp = d.split("runs23/main/")[1].split("/")[0]
    per = {}
    try:
        for r in csv.DictReader(open(d)):
            if r.get("val_corr"):
                try:
                    per[int(float(r["epoch"]))] = float(r["val_corr"])
                except (ValueError, KeyError):
                    pass
    except Exception:
        continue
    if per:
        be = max(per, key=lambda e: (per[e], e))
        if exp not in corr_by or per[be] > corr_by[exp][1]:
            corr_by[exp] = (be, per[be])

print("%-22s %8s %6s  file" % ("expert", "epoch", "corr"))
picked = {}
for e in EXPERTS:
    if e not in corr_by:
        # only -last exists
        src = f"{NETS}/{e}-last.nnue"
        if os.path.exists(src):
            picked[e] = ("last", None, src)
            print("%-22s %8s %6s  %s" % (e, "last", "-", e + "-last.nnue"))
        else:
            print(f"{e}: NO NET FOUND")
        continue
    ep, corr = corr_by[e]
    src = f"{NETS}/{e}-e{ep}.nnue"
    if not os.path.exists(src):
        alt = sorted(glob.glob(f"{NETS}/{e}-e*.nnue"), key=os.path.getmtime)
        src = alt[-1] if alt else f"{NETS}/{e}-last.nnue"
    picked[e] = (str(ep), corr, src)
    print("%-22s %8d %6.3f  %s" % (e, ep, corr, os.path.basename(src)))

# assemble engine dir: slot-name = expert
for e, (ep, corr, src) in picked.items():
    if src and os.path.exists(src):
        shutil.copy(src, f"{OUT}/{e}.nnue")
print(f"\nengine dir: {OUT} ({len(glob.glob(OUT + '/*.nnue'))} nets)")
