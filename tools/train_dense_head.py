"""DENSE head v3 — stage 1: assemble training data (operator design).

Joins per-position, aligned by bin order:
  13 scalar evals        (eval_cache/*.evals.npy)
  13x1024 activations    (cache/*.uint8.npy, all experts)
  36 HCE + domain        (cache/*.hce.npz)
  search-eval targets    (cache/*.targets.npy)  — dense/sharp supervision
  raw active indices     (engine rawft dumps over bin FENs — authoritative)
"""
import sys, os, glob, time, subprocess, struct
sys.path.insert(0, "/srv/workspace/flychess/src/nnue-pytorch")
sys.path.insert(0, "/srv/workspace/flychess/src/chess-lab/tools")
import numpy as np
import audit_packer

R = "/mnt/cold-raid6/chess-audit"
SP = f"{R}/selfplay_rl"
ACT_BIN = "/srv/workspace/flychess/src/Stockfish-act/src/stockfish"
SFEN_SIZE = 40
N_TARGET = 120000

def main():
    bins = sorted(glob.glob(f"{R}/expert_bins_both/*.bin"))
    evals_l, acts_l, hce_l, dom_l, tgt_l, fens = [], [], [], [], [], []
    total = 0
    for bfn in bins:
        name = os.path.basename(bfn)[:-4]
        ev_f = f"/srv/workspace/flychess/eval_cache/{name}.evals.npy"
        ac_f = f"/srv/workspace/flychess/cache/{name}.uint8.npy"
        hd_f = f"/srv/workspace/flychess/cache/{name}.hce.npz"
        tg_f = f"/srv/workspace/flychess/cache/{name}.targets.npy"
        if not all(os.path.exists(p) for p in [ev_f, ac_f, hd_f, tg_f, bfn]):
            continue
        evals = np.load(ev_f, mmap_mode="r")
        acts = np.load(ac_f, mmap_mode="r")
        tgts = np.load(tg_f, mmap_mode="r")
        hd = np.load(hd_f)
        n = min(evals.shape[0], acts.shape[0], tgts.shape[0], hd["hce"].shape[0])
        if total + n > N_TARGET:
            n = N_TARGET - total
        if n <= 0:
            break
        evals_l.append(np.array(evals[:n], dtype=np.float32))
        acts_l.append(np.array(acts[:n]))
        hce_l.append(hd["hce"][:n])
        dom_l.append(hd["domain"][:n])
        tgt_l.append(np.array(tgts[:n], dtype=np.float32))
        # bin records: 40B each = [32B packed sfen][2B cp][2B move][2B fm][2B pad]
        raw = open(bfn, "rb").read()
        for i in range(n):
            fens.append(audit_packer.unpack_sfen(raw[i * SFEN_SIZE: i * SFEN_SIZE + 32])[0].fen())
        total += n
        print(f"  {name}: {n} (total {total})", flush=True)
        if total >= N_TARGET:
            break
    evals = np.concatenate(evals_l)
    acts = np.concatenate(acts_l)
    hces = np.concatenate(hce_l)
    doms = np.concatenate(dom_l)
    tgts = np.concatenate(tgt_l)
    print(f"joined {len(fens)} positions", flush=True)

    # raw feature indices via the engine (authoritative enumeration)
    t0 = time.time()
    p = subprocess.Popen([ACT_BIN], stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                         stderr=subprocess.STDOUT, text=True, bufsize=1)
    p.stdin.write("uci\n"); p.stdin.flush(); time.sleep(1.0)
    idx_lists = []
    MAXF = 160
    ft = np.full((len(fens), MAXF), -1, dtype=np.int32)
    cnt = np.zeros(len(fens), dtype=np.int32)
    for i, fen in enumerate(fens):
        p.stdin.write(f"rawft {fen}\n"); p.stdin.flush()
        line = p.stdout.readline()
        if not line.startswith("rawft"):
            # engine may emit other lines; skip until rawft
            while True:
                line = p.stdout.readline()
                if line.startswith("rawft") or not line:
                    break
        toks = line.split()[1:]
        c = min(len(toks), MAXF)
        ft[i, :c] = [int(t) for t in toks[:c]]
        cnt[i] = c
        if i % 20000 == 0 and i:
            print(f"  rawft {i}/{len(fens)} ({time.time()-t0:.0f}s)", flush=True)
    p.stdin.write("quit\n"); p.stdin.flush()
    try: p.wait(timeout=5)
    except Exception: p.kill()
    print(f"rawft done: {len(fens)} positions, mean active {cnt.mean():.1f}, max {cnt.max()}", flush=True)

    np.savez(f"{SP}/dense_data.npz", evals=evals, acts=acts, hces=hces,
             doms=doms, tgts=tgts, ft=ft, ftcnt=cnt)
    print("STAGE1_DONE", flush=True)

if __name__ == "__main__":
    main()
