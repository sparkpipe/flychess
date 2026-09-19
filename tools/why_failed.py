"""Diagnose the un-learned questions: per-move scores at INIT vs after
the saved delta, against the graded targets. q4/q8 passed at step 0 and
training destroyed the answer — show the flip."""
import sys
import json
import random
import chess
import torch

sys.path.insert(0, "/home/spec/chess-lab")
sys.path.insert(0, "/home/spec/chess-lab/tools")
import numpy as np
import ten_parallel as tp
import flyfeat_cb
import fly_curriculum as fc

flyfeat_cb.feat_vec(chess.Board())

for qi in (int(x) for x in (sys.argv[1:] or [4, 8, 5])):
    rows = fc.load_pools([f"TENQ_{qi}"])
    e = rows[0]
    fvb, slots, tgts, cl, rich = fc.build_tb_batch(random.Random(1),
                                                   rows, 1)
    sv, vv, vmm = slots[0]
    L = len(sv)

    def scores(mm):
        F = rich[0][0].shape[1]
        M = L
        slotb = np.zeros((1, M), np.int64)
        pcrowb = np.zeros((1, M), np.int64)
        mfb = np.zeros((1, M, F), np.float32)
        maskb = np.zeros((1, M), bool)
        psb = np.zeros((1, M), np.float32)
        thb = np.zeros((1, M), np.float32)
        ps2b = np.zeros((1, M), np.float32)
        vmm1 = np.zeros((1, M), bool)
        mfs, pss, ps2s, thrs, pci = rich[0]
        slotb[0, :L] = sv
        pcrowb[0, :L] = pci
        mfb[0, :L] = mfs
        maskb[0, :L] = True
        psb[0, :L] = pss
        thb[0, :L] = thrs
        ps2b[0, :L] = ps2s
        vmm1[0, :L] = vmm
        T, _, _, _ = fc.forward(mm, fvb, slotb, pcrowb, mfb, maskb,
                                psb, thb, ps2b)
        return T[0].tolist()[:L]

    m_init = tp.build_model(0)
    s_init = scores(m_init)

    m_tr = tp.build_model(0)
    d = torch.load(f"/home/spec/chess-lab/ten_q/vec_{qi}.pt",
                   map_location=tp.DEV)
    with torch.no_grad():
        for k, p in m_tr.named_parameters():
            p.add_(d[k].to(tp.DEV))
    s_tr = scores(m_tr)

    b = chess.Board(e["fen"])
    mus = [mv.uci() for mv in b.legal_moves]
    print(f"=== q{qi}  {e['fen']}  best={e['best']} cat={e['cat']}")
    print(f"{'move':>7} {'target':>8} {'init':>8} {'trained':>9}")
    for j in range(L):
        mark = " <== BEST" if mus[j] == e["best"] else ""
        print(f"{mus[j]:>7} {vv[j]:>8.3f} {s_init[j]:>8.3f}"
              f"{s_tr[j]:>9.3f}{mark}")
    order_i = sorted(range(L), key=lambda j: -s_init[j])
    order_t = sorted(range(L), key=lambda j: -s_tr[j])
    print(f"init rank of best: "
          f"{[mus[j] for j in order_i].index(e['best']) + 1}/{L}"
          f"   trained rank: "
          f"{[mus[j] for j in order_t].index(e['best']) + 1}/{L}")
