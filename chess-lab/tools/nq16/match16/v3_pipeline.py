#!/usr/bin/env python3
"""v3 pipeline orchestrator (box): wait for SF17 relabel -> assemble v3 bins ->
audit gate -> sequential expert training on the 5090 with plateau stop.
Auto-runs overnight; logs to /srv/workspace/chess-active/v3_pipeline.log"""
import json
import os
import subprocess
import sys
import time

CA = "/srv/workspace/chess-active"
LOG = f"{CA}/v3_pipeline.log"
RELABEL_LOG = f"{CA}/relabel17c.log"
LABELS = f"{CA}/miniature_labels17.tsv.part"
ASM = "/tmp/assemble16v3.py"
NNUE = "/srv/workspace/flychess/src/nnue-pytorch"

# training order: play-critical miniature-heavy experts first
ORDER = ["op_even_l0", "op_even_l1", "mg_safe", "op_pawnimb", "mg_unsafe",
         "piece_down", "nvb", "dv_R", "dv_Q", "mvr", "oppb", "op_even_l2p",
         "rv2m", "qvmat", "dv_rest", "tb"]
# plateau stop: (patience epochs without val_corr improvement, wallclock cap h)
BIG = ("op_even_l0", "nvb", "piece_down", "op_even_l1", "mg_safe", "dv_R", "dv_Q")


def log(msg):
    print(time.strftime("%H:%M:%S "), msg, flush=True)


def wait_relabel():
    total = sum(1 for _ in open(f"{CA}/miniature_rows.tsv"))
    while True:
        if "COMPLETE" in open(RELABEL_LOG).read()[-400:]:
            return
        try:
            n = sum(1 for _ in open(LABELS))
        except FileNotFoundError:
            n = 0
        log(f"waiting for relabel: {n:,}/{total:,}")
        time.sleep(600)


def assemble_v3():
    src = open("/tmp/assemble16r_v3src.py").read()
    src = src.replace('OUT = "/srv/workspace/chess-active/train16"',
                      f'OUT = "{CA}/train16v3"')
    src = src.replace('MINILBL = "/srv/workspace/chess-active/miniature_labels.tsv"',
                      f'MINILBL = "{CA}/miniature_labels17.tsv"')
    src = src.replace('if emit(board, cp, mv, ("mini", gid), pov="white"):',
                      'if emit(board, cp, mv, ("mini", gid)):')  # labels17 = stm-pov
    open(ASM, "w").write(src)
    r = subprocess.run([sys.executable, ASM], capture_output=True, text=True)
    open(f"{CA}/train16v3_assemble.log", "w").write(r.stdout + r.stderr)
    log("v3 assembly done" if r.returncode == 0 else f"ASSEMBLY FAILED rc={r.returncode}")
    return r.returncode == 0


def audit_gate():
    r = subprocess.run([sys.executable, "/tmp/audit_bin_labels_v3.py",
                        "op_even_l0", "piece_down", "tb"],
                       capture_output=True, text=True)
    open(f"{CA}/v3_audit.log", "w").write(r.stdout + r.stderr)
    ok = "stm-sign" in r.stdout
    log("audit gate:\n" + r.stdout)
    return ok


def train_expert(e):
    tr = f"{CA}/train16v3/{e}.train.bin"
    va = f"{CA}/train16v3/{e}.val.bin"
    recs = os.path.getsize(tr) // 40
    vr = min(30000, os.path.getsize(va) // 40)
    run = f"{CA}/run_v3_{e}"
    os.makedirs(run, exist_ok=True)
    cap_h = 8 if e in BIG else 4
    patience = 25 if e in BIG else 60
    cmd = [sys.executable, "train.py", tr,
           "--validation-datasets", va, "--validation-size", str(vr),
           "--check-val-every-n-epoch", "1", "--epoch-size", str(recs),
           "--batch-size", "4096", "--max-time", f"00:{cap_h:02d}:00:00",
           "--max-epochs", "100000", "--network-save-period", "10",
           "--random-fen-skipping", "3", "--default-root-dir", run]
    p = subprocess.Popen(cmd, cwd=NNUE, stdout=open(f"{run}/train.log", "w"),
                         stderr=subprocess.STDOUT)
    log(f"[{e}] training (pid {p.pid}, {recs:,} rec/epoch, cap {cap_h}h, patience {patience})")
    best, best_ep, last_ep = -1, -1, -1
    mfile = None
    while p.poll() is None:
        time.sleep(120)
        try:
            import glob
            fs = sorted(glob.glob(f"{run}/lightning_logs/version_*/metrics.csv"),
                        key=os.path.getmtime)
            mfile = fs[-1]
        except Exception:
            continue
        best = best_ep = -1
        for line in open(mfile):
            pass
        # cheap scan
        rows = [l.split(",") for l in open(mfile)]
        for r in rows:
            if len(r) > 7 and r[7] not in ("", "val_corr\n") and r[7].strip():
                try:
                    ep, c = int(r[0]), float(r[7])
                except ValueError:
                    continue
                last_ep = ep
                if c > best:
                    best, best_ep = c, ep
        if best_ep >= 0 and last_ep - best_ep >= patience:
            log(f"[{e}] plateau: best {best:.4f}@e{best_ep}, now e{last_ep} — stopping")
            p.terminate()
            time.sleep(5)
            p.kill()
            break
    log(f"[{e}] done rc={p.returncode()} best={best:.4f}@e{best_ep}")
    return best, best_ep


def main():
    log("v3 pipeline start")
    wait_relabel()
    if not assemble_v3():
        sys.exit(1)
    if not audit_gate():
        log("AUDIT GATE FAILED — not training")
        sys.exit(1)
    for e in ORDER:
        if not os.path.exists(f"{CA}/train16v3/{e}.train.bin"):
            log(f"[{e}] MISSING BIN — skip")
            continue
        train_expert(e)
    log("ALL EXPERTS DONE — pick nets, build n16.2, run evalsigntest + match")


if __name__ == "__main__":
    main()
