"""Witness precompute for the MoF micro-reader (operator directive:
"inputs being all the visible outputs from all the flies").

For every DEGM2 row and every closure fly: the fly's canonical score
T for every legal move of that row. Stored fp16:
  witness.npy  [138, T_total]   (memmap; per-move fly scores)
  tokens_mf.npy[T, F]     fp16  (move identity features)
  pos_x.npy    [N, F]     fp16  (position features)
  approvals.npy[T]        uint8 (approved-set mask per move)
  meta.json    fens / pool / offs / L
Rows enumerated in load_all() order (sorted DEGM2 pools).
"""
import sys
import os
import json
import glob
import time

os.environ.setdefault("ANORM", "1")
os.environ.setdefault("B", "1")
sys.path.insert(0, "/home/spec/chess-lab")
sys.path.insert(0, "/home/spec/chess-lab/tools")
import chess
import torch
import numpy as np
import ten_parallel as tp
import flyfeat_cb
import fly_curriculum as fc

CLOS = "/home/spec/chess-lab/singles/closure"
OUT = "/home/spec/chess-lab/singles/witness"
CHUNK = 2048
os.makedirs(OUT, exist_ok=True)


def load_rows():
    rows = []
    pools = sorted(os.path.basename(f)[:-6] for f in glob.glob(
        "/home/spec/chess-lab/tbpools/DEGM2_Ch*.jsonl"))
    for pool in pools:
        rows += fc.load_pools([pool])
    return rows


def pack(rows_sl):
    """pack a chunk of rows into forward() inputs (pre path)."""
    B = len(rows_sl)
    Ls = []
    for r in rows_sl:
        pres, ri = r["_pre"]
        Ls.append(int(pres["fen_off"][ri + 1] - pres["fen_off"][ri]))
    M = max(Ls)
    F = rows_sl[0]["_pre"][0]["mf"].shape[1]     # move feats = 17
    Fx = flyfeat_cb.feat_vec_by_fen(rows_sl[0]["fen"]).shape[0]
    slotb = np.zeros((B, M), np.int64)
    pcrowb = np.zeros((B, M), np.int64)
    mfb = np.zeros((B, M, F), np.float32)
    maskb = np.zeros((B, M), bool)
    psb = np.zeros((B, M), np.float32)
    thb = np.zeros((B, M), np.float32)
    ps2b = np.zeros((B, M), np.float32)
    vmb = np.zeros((B, M), bool)
    xs = np.zeros((B, Fx), np.float32)
    for b, r in enumerate(rows_sl):
        pres, ri = r["_pre"]
        s, e = int(pres["fen_off"][ri]), int(pres["fen_off"][ri + 1])
        L = e - s
        slotb[b, :L] = pres["slot"][s:e]
        pcrowb[b, :L] = pres["pcrow"][s:e]
        mfb[b, :L] = pres["mf"][s:e]
        maskb[b, :L] = True
        psb[b, :L] = pres["pseudo"][s:e]
        thb[b, :L] = pres["threat"][s:e]
        ps2b[b, :L] = pres["pseudo2"][s:e]
        vmb[b, :L] = True
        xs[b] = flyfeat_cb.feat_vec_by_fen(r["fen"])   # 2746-dim
    return (xs, slotb, pcrowb, mfb, maskb, psb, thb, ps2b, vmb, Ls)


def main():
    flyfeat_cb.feat_vec(chess.Board())
    rows = load_rows()
    N = len(rows)
    print(json.dumps({"rows": N}), flush=True)

    Ls = []
    fens = []
    pool_of = []
    approvals = []
    for r in rows:
        pres, ri = r["_pre"]
        s, e = int(pres["fen_off"][ri]), int(pres["fen_off"][ri + 1])
        Ls.append(e - s)
        fens.append(r["fen"])
        pool_of.append(r["pool"])
        approvals.append(pres["opt"][s:e].astype(np.uint8))
    Ls = np.array(Ls, dtype=np.int64)
    offs = np.concatenate([[0], np.cumsum(Ls)])
    T = int(offs[-1])
    print(json.dumps({"T_total": T}), flush=True)
    approvals = np.concatenate(approvals)
    np.save(f"{OUT}/approvals.npy", approvals)
    json.dump({"fens": fens, "pool": pool_of, "offs": offs.tolist(),
               "L": Ls.tolist()}, open(f"{OUT}/meta.json", "w"))

    F = rows[0]["_pre"][0]["mf"].shape[1]      # move feats = 17
    Fx = flyfeat_cb.feat_vec(chess.Board())[0].shape[0]   # pos = 2746
    xs = np.zeros((N, Fx), np.float16)
    mfm = np.lib.format.open_memmap(
        f"{OUT}/tokens_mf.npy", mode="w+", dtype=np.float16,
        shape=(T, F))
    for s in range(0, N, CHUNK):
        e = min(s + CHUNK, N)
        (xs_c, slotb, pcrowb, mfb, maskb, psb, thb, ps2b, vmb,
         Lc) = pack(rows[s:e])
        xs[s:e] = xs_c.astype(np.float16)
        mfm[offs[s]:offs[e]] = mfb[maskb].astype(np.float16)
    mfm.flush()
    np.save(f"{OUT}/pos_x.npy", xs)
    json.dump({"F_move": F, "F_pos": Fx},
              open(f"{OUT}/dims.json", "w"))
    print("static arrays done", flush=True)

    z = np.load(f"{CLOS}/deltas_2.npz")
    state = json.load(open(f"{CLOS}/models_2.json"))
    flies = [{"idx": z[f"m{i:05d}_idx"],
              "val": z[f"m{i:05d}_val"]} for i in range(len(state))]
    base = tp.build_model(0)
    base_state = {k: p.detach().clone()
                  for k, p in base.named_parameters()}
    base_cat = torch.cat([base_state[k].flatten()
                          for k, _ in base.named_parameters()])
    W = np.lib.format.open_memmap(
        f"{OUT}/witness.npy", mode="w+", dtype=np.float16,
        shape=(len(flies), T))
    t0 = time.time()
    for fi, f in enumerate(flies):
        with torch.no_grad():
            for k, p in base.named_parameters():
                p.copy_(base_state[k])
            flat = base_cat.clone()
            flat.index_add_(0, torch.from_numpy(f["idx"]).to(fc.DEV),
                            torch.from_numpy(f["val"]).to(fc.DEV))
            off = 0
            for _, p in base.named_parameters():
                n_ = p.numel()
                p.copy_(flat[off:off + n_].view_as(p))
                off += n_
        for s in range(0, N, CHUNK):
            e = min(s + CHUNK, N)
            (xs_c, slotb, pcrowb, mfb, maskb, psb, thb, ps2b, vmb,
             Lc) = pack(rows[s:e])
            Tt, _, _, _ = fc.forward(base, xs_c, slotb, pcrowb, mfb,
                                     maskb, psb, thb, ps2b)
            Tt = Tt.detach().cpu().numpy()
            for b in range(e - s):
                o = int(offs[s + b])
                W[fi, o:o + Lc[b]] = Tt[b, :Lc[b]].astype(np.float16)
        W.flush()
        print(json.dumps({"fly": fi, "elapsed_s":
                          round(time.time() - t0)}), flush=True)
    print("WITNESS-COMPLETE", flush=True)


if __name__ == "__main__":
    main()
