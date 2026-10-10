"""Pick best-val epoch per expert, pull the picked .nnue to the box."""
import subprocess, os, re

RAID = "/mnt/cold-raid6/chess-audit/train16_backup"
OUT = "/srv/workspace/chess-active/engine16"
os.makedirs(OUT, exist_ok=True)

EXPERTS = "tb mvr rv2m qvmat nvb piece_down oppb dv_Q dv_R dv_rest op_pawnimb op_even_l0 op_even_l2p mg_unsafe mg_safe".split()
SPARKS = "spark0 spark1 spark2 spark3 spark4 spark5 spark6 spark7 spark8 spark9 sparka sparkb sparkd sparke sparkf".split()

results = []
for e, s in zip(EXPERTS, SPARKS):
    # best epoch from metrics on the spark
    r = subprocess.run(["ssh", "-o", "ConnectTimeout=10", "-o", "BatchMode=yes", s,
                        f"f=$(ls ~/run16_{e}/lightning_logs/version_*/metrics.csv | tail -1); "
                        f"awk -F, 'NF>8 && $8!=\"\" {{print $1\" \"$8}}' $f | sort -k2 -gr | head -1"],
                       capture_output=True, text=True)
    line = r.stdout.strip()
    if not line:
        results.append((e, s, None, None, "NO_METRICS"))
        continue
    ep, corr = line.split()
    net = f"{e}_e{ep}.nnue"
    # net on spark (prune keeps best) or in RAID backup
    chk = subprocess.run(["ssh", "-o", "ConnectTimeout=10", "-o", "BatchMode=yes", s,
                          f"ls ~/run16_{e}/nets/{net} 2>/dev/null"],
                        capture_output=True, text=True)
    src = f"{s}:run16_{e}/nets/{net}" if chk.stdout.strip() else None
    if not src and os.path.exists(f"{RAID}/{e}/nets/{net}"):
        src = f"{RAID}/{e}/nets/{net}"
    if not src:
        # fallback: newest net available
        chk2 = subprocess.run(["ssh", "-o", "ConnectTimeout=10", "-o", "BatchMode=yes", s,
                               f"ls -t ~/run16_{e}/nets/*.nnue 2>/dev/null | head -1"],
                              capture_output=True, text=True)
        if chk2.stdout.strip():
            net = os.path.basename(chk2.stdout.strip())
            ep = re.search(r"_e(\d+)", net).group(1)
            src = f"{s}:run16_{e}/nets/{net}"
        elif os.path.exists(f"{RAID}/{e}/nets"):
            cands = sorted(os.listdir(f"{RAID}/{e}/nets"))
            if cands:
                net = cands[-1]
                ep = re.search(r"_e(\d+)", net).group(1)
                src = f"{RAID}/{e}/nets/{net}"
    if src:
        dst = f"{OUT}/{e}.nnue"
        if ":" in src and not src.startswith("/"):
            subprocess.run(["scp", "-q", "-o", "ConnectTimeout=15", src, dst], check=False)
        else:
            subprocess.run(["cp", src, dst], check=False)
        ok = os.path.exists(dst) and os.path.getsize(dst) > 80000000
        results.append((e, s, ep, corr, "OK" if ok else "COPY_FAILED"))
    else:
        results.append((e, s, ep, corr, "NET_MISSING"))

print(f"{'expert':14s} {'spark':8s} {'ep':>5s} {'corr':>7s} status")
for e, s, ep, corr, st in results:
    print(f"{e:14s} {s:8s} {str(ep):>5s} {str(corr)[:7]:>7s} {st}")
