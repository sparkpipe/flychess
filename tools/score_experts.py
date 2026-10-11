"""EXPERT SCORING MATRIX: every expert evaluates every unseen test position;
MAE in win-prob units vs depth-20 ground truth; domain-tagged; plus the
seen-val comparison for the generalization gap.

Inputs: gt_test_draws.w*, gt_test_puzzles.w* (fen-line|cp),
        runs2 ckpts (best-val selection replicated), train_val/val bins.
Output: /mnt/cold-raid6/chess-audit/expert_matrix.txt (+ .json)
"""
import os
import re
import sys
import glob
import struct
import math
import json

import numpy as np
import torch

sys.path.insert(0, "/home/spec/nnue-pytorch")
sys.path.insert(0, "/home/spec/chess-lab/tools")
import data_loader
import model as M
import chess
from audit_packer import unpack_sfen

R = "/mnt/cold-raid6/chess-audit"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
EXPERTS = ["balanced_l0", "balanced_l1", "nvb", "exchanges", "balanced_l2",
           "oppb", "bvr", "dvoretsky", "balanced_l3", "nvr", "rv2m",
           "qvmat", "tactics"]
BATCH = 512


def best_ckpt(name):
    log = f"{R}/runs2/{name}.trainlog"
    if not os.path.exists(log):
        return None
    best, bep = 1e9, 0
    for line in open(log):
        m = re.match(r"Epoch (\d+) \(Val\): \[val_loss_epoch=([0-9.]+)\]", line)
        if m:
            ep, v = int(m.group(1)), float(m.group(2))
            if v < best:
                best, bep = v, ep
    cks = []
    for c in glob.glob(f"{R}/runs2/{name}/lightning_logs/version_*/"
                       "checkpoints/epoch=*-step=*.ckpt"):
        em = re.search(r"epoch=(\d+)-", c)
        cks.append((int(em.group(1)), c))
    below = [c for e, c in sorted(cks) if e <= bep]
    if below:
        return below[-1]
    last = sorted(glob.glob(f"{R}/runs2/{name}/lightning_logs/version_*/"
                            "checkpoints/last.ckpt"))
    return last[-1] if last else None


def load_expert(name):
    ck = best_ckpt(name)
    if ck is None:
        return None, None
    ckpt = torch.load(ck, map_location="cpu", weights_only=False)
    net = M.NNUE(config=M.NNUELightningConfig())
    net.load_state_dict(ckpt["state_dict"])
    net.to(DEVICE).eval()
    return net.model, ck


def eval_fens(mdl, fens):
    """stm-perspective cp list for fens via the trainer's forward."""
    out = []
    for i in range(0, len(fens), BATCH):
        chunk = fens[i:i + BATCH]
        b = data_loader.get_sparse_batch_from_fens(
            "Full_Threats+PP_3Wide+HalfKAv2_hm", chunk,
            [0] * len(chunk), [1] * len(chunk), [0] * len(chunk))
        (us, them, wi, bi, _o, _s, pc) = b.contents.get_tensors(DEVICE)
        with torch.no_grad():
            vs = mdl.forward(us, them, wi, bi, pc)
        out += [float(v) * mdl.quantization.nnue2score for v in vs]
        data_loader.destroy_sparse_batch(b)
    return out


def wp(cp):
    return 1.0 / (1.0 + math.exp(-max(min(cp, 3000), -3000) / 361.0))


def domain(fen):
    b = chess.Board(fen)
    men = len(b.piece_map())
    if men <= 5:
        return "tb"
    cw, cb = {"Q": 0, "R": 0, "B": 0, "N": 0}, {"Q": 0, "R": 0, "B": 0, "N": 0}
    wcol, bcol = [], []
    for s, p in b.piece_map().items():
        u = p.symbol().upper()
        if u in ("K", "P"):
            continue
        (cw if p.color else cb)[u] += 1
        if u == "B":
            (wcol if p.color else bcol).append(
                (chess.square_file(s) + chess.square_rank(s)) % 2)
    resw = "".join(sorted(sum(([k] * max(0, cw[k] - cb[k]) for k in "QRBN"), [])))
    resb = "".join(sorted(sum(([k] * max(0, cb[k] - cw[k]) for k in "QRBN"), [])))
    kres = tuple(sorted((resw, resb)))
    sym = resw == "" and resb == ""
    RES = {("B", "N"): "nvb", ("N", "R"): "nvr", ("B", "R"): "bvr",
           ("NN", "R"): "rv2m", ("BN", "R"): "rv2m", ("BB", "R"): "rv2m",
           ("Q", "RR"): "qvmat", ("Q", "RN"): "qvmat", ("Q", "RB"): "qvmat"}
    if not sym and kres in RES:
        return RES[kres]
    if sym and wcol and bcol and \
            (sum(wcol) * 2 > len(wcol)) != (sum(bcol) * 2 > len(bcol)):
        return "oppb"
    if men <= 10:
        return "dvoretsky"
    lock = 0
    for f in range(8):
        wr = [chess.square_rank(s) for s in chess.SquareSet(
            b.pieces(chess.PAWN, chess.WHITE)) if chess.square_file(s) == f]
        br = [chess.square_rank(s) for s in chess.SquareSet(
            b.pieces(chess.PAWN, chess.BLACK)) if chess.square_file(s) == f]
        if wr and br and min(br) - max(wr) == 1 and 2 <= f <= 5:
            lock += 1
    return "balanced_l%d" % (lock if lock < 3 else 3)


def main():
    # test sets with ground truth
    test = {}
    for src in ("gt_test_draws", "gt_test_puzzles"):
        for wf in glob.glob(f"{R}/{src}.w*"):
            for line in open(wf):
                parts = line.rstrip("\n").split("|")
                if len(parts) < 8:
                    continue
                fen, gt = parts[0], int(parts[7])
                test.setdefault(src, []).append((fen, gt, domain(fen)))
    print("unseen: draws %d, puzzles %d"
          % (len(test.get("gt_test_draws", [])),
             len(test.get("gt_test_puzzles", []))), flush=True)

    # seen-val samples (300/bin) for the gap
    seen = {}
    for vf in sorted(glob.glob(R + "/train_val/val/*.bin")):
        name = os.path.basename(vf)[:-4]
        rows = []
        with open(vf, "rb") as f:
            n = os.path.getsize(vf) // 40
            step = max(1, n // 300)
            for i in range(0, n, step):
                f.seek(i * 40)
                r = f.read(40)
                if len(r) < 40:
                    break
                b, hm, fm = unpack_sfen(r[:32])
                cp = struct.unpack("<h", r[32:34])[0]
                rows.append((b.fen(), cp))
        seen[name] = rows[:300]

    nets = {}
    for e in EXPERTS:
        mdl, ck = load_expert(e)
        if mdl is not None:
            nets[e] = mdl
            print("loaded", e, "->", os.path.basename(ck), flush=True)
        else:
            print("MISSING", e, flush=True)

    results = {}
    for e, mdl in nets.items():
        results[e] = {}
        # unseen
        for src in test:
            fens = [x[0] for x in test[src]]
            evs = eval_fens(mdl, fens)
            doms = [x[2] for x in test[src]]
            gts = [x[1] for x in test[src]]
            per = {}
            for f_, ev, dm, g in zip(fens, evs, doms, gts):
                per.setdefault(dm, []).append(abs(wp(ev) - wp(g)))
            results[e][src] = {d: (sum(v) / len(v), len(v))
                               for d, v in per.items()}
        # seen
        per = {}
        for name, rows in seen.items():
            if not rows:
                continue
            evs = eval_fens(mdl, [x[0] for x in rows])
            for (f_, cp), ev in zip(rows, evs):
                # seen bins: cp IS the label this net was trained toward
                per.setdefault(name, []).append(abs(wp(ev) - wp(cp)))
        results[e]["seen"] = {d: (sum(v) / len(v), len(v))
                              for d, v in per.items()}
        print("scored", e, flush=True)

    with open(R + "/expert_matrix.json", "w") as j:
        json.dump(results, j, indent=1)
    with open(R + "/expert_matrix.txt", "w") as t:
        t.write("MAE in win-prob units vs depth-20 GT (lower=better)\n")
        for src in ("gt_test_draws", "gt_test_puzzles"):
            t.write("\n=== %s ===\n" % src)
            t.write("%-14s %s\n" % ("expert\\domain",
                                    " ".join("%-9s" % d for d in
                                             sorted({d for e in results
                                                     for d in results[e].get(src, {})}))))
            for e in results:
                row = results[e].get(src, {})
                t.write("%-14s %s\n" % (e, " ".join(
                    "%-9s" % ("%.4f" % row[d][0] if d in row else "-")
                    for d in sorted({d for x in results
                                     for d in results[x].get(src, {})}))))
        t.write("\n=== GENERALIZATION GAP (unseen-draws MAE - seen-own-bin MAE) ===\n")
        for e in results:
            own = results[e]["seen"].get(e, (None, 0))[0]
            t.write("%-14s seen_own=%s\n" % (e,
                    "%.4f" % own if own else "n/a"))
    print("MATRIX WRITTEN")


if __name__ == "__main__":
    main()
