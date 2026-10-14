#!/bin/bash
# Assemble baseline engine: best-of-all picker across runs, evalsign gate, match.
set -u
R=/srv/workspace/chess-active
E=$R/engine16-baseline
mkdir -p $E
# best per expert: fleet1h best.nnue vs earlier rounds' final.nnue (by val_corr)
python3 - <<'EOF'
import csv, glob, os, shutil
R = "/srv/workspace/chess-active"
E = f"{R}/engine16-baseline"
EXPERTS = ["tb", "mvr", "rv2m", "qvmat", "nvb", "piece_down", "oppb", "dv_Q",
           "dv_R", "dv_rest", "op_pawnimb", "op_even_l0", "op_even_l1",
           "op_even_l2p", "mg_unsafe", "mg_safe"]
RUNS = [f"{R}/runs-fleet1h", f"{R}/runs-continue", f"{R}/runs-extra10",
        f"{R}/runs-10min"]
for e in EXPERTS:
    best = (None, -1)
    for R_ in RUNS:
        net = f"{R_}/{e}/best.nnue" if "fleet1h" in R_ else f"{R_}/{e}/final.nnue"
        fs = glob.glob(f"{R_}/{e}/lightning_logs/version_*/metrics.csv")
        corr = -1
        for m in fs:
            try:
                for r in csv.DictReader(open(m)):
                    c = r.get("val_corr")
                    if c not in (None, ""):
                        try:
                            corr = max(corr, float(c))
                        except ValueError:
                            pass
            except Exception:
                pass
        if os.path.exists(net) and corr > best[1]:
            best = (net, corr)
    if best[0]:
        shutil.copy(best[0], f"{E}/{e}.nnue")
        src = best[0].split("/")[-2]
        print(f"{e:<12} corr={best[1]:.4f}  <- {src}")
    else:
        print(f"{e:<12} MISSING")
EOF
echo "=== evalsign gate"
python3 /tmp/pilot_verdict.py 2>/dev/null | head -1 || true
