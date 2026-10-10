"""STACKED HEAD — Phase A trainer (streaming, memory-safe).

Streams activation batches from the uint8 mmap cache. Never loads
the full dataset into RAM. Trains the linear head with per-domain
gates (init 1.0), routed one-hot, and HCE features.
"""
import sys, os, glob, struct, math, random
import numpy as np
import torch, torch.nn as nn

sys.path.insert(0, "/home/spec/chess-lab/tools")
from audit_packer import unpack_sfen
import chess
from score_experts import domain

R = "/mnt/cold-raid6/chess-audit"
CACHE = R + "/stack_cache"
EXPERTS = ["balanced_l0", "balanced_l1", "nvb", "exchanges", "balanced_l2",
           "oppb", "bvr", "dvoretsky", "balanced_l3", "nvr", "rv2m",
           "qvmat", "tactics", "tb_vacant"]
DOMAINS = ["balanced_l0", "balanced_l1", "balanced_l2", "balanced_l3",
           "nvb", "nvr", "bvr", "rv2m", "qvmat", "oppb", "dvoretsky",
           "exchanges", "none"]
D2I = {d: i for i, d in enumerate(DOMAINS)}
E2I = {e: i for i, e in enumerate(EXPERTS)}


def hce(board):
    f = []
    for pt, v in [(chess.PAWN,1),(chess.KNIGHT,3),(chess.BISHOP,3),(chess.ROOK,5),(chess.QUEEN,9)]:
        f.append(len(board.pieces(pt, chess.WHITE)) * v / 9.0)
        f.append(len(board.pieces(pt, chess.BLACK)) * v / 9.0)
    f.append(1.0 if board.turn == chess.WHITE else -1.0)
    f.append(len(board.piece_map()) / 32.0)
    f.append(board.fullmove_number / 100.0)
    f.append(1.0 if board.has_kingside_castling_rights(chess.WHITE) else 0.0)
    f.append(1.0 if board.has_queenside_castling_rights(chess.WHITE) else 0.0)
    f.append(1.0 if board.has_kingside_castling_rights(chess.BLACK) else 0.0)
    f.append(1.0 if board.has_queenside_castling_rights(chess.BLACK) else 0.0)
    f.append(1.0 if board.ep_square is not None else 0.0)
    for side in (chess.WHITE, chess.BLACK):
        pawns = board.pieces(chess.PAWN, side)
        files = [0] * 8
        for s in pawns:
            files[chess.square_file(s)] += 1
        f += [x / 2.0 for x in files]
        f.append(sum(1 for x in files if x >= 2) / 4.0)
    return f


def main():
    smoke = "--smoke" in sys.argv
    torch.manual_seed(42)
    rng = random.Random(42)

    calib = np.load(CACHE + "/calibration.npz")
    scale, zero, D = calib["scale"], calib["zero"], int(calib["D"])

    # collect bins
    bins = []
    for cf in sorted(glob.glob(CACHE + "/*.uint8.npy")):
        name = os.path.basename(cf)[:-10]
        feats = np.load(cf, mmap_mode="r")
        tgts = np.load(CACHE + "/" + name + ".targets.npy")
        fens, _ = [], []
        src = R + "/expert_bins_both/" + name + ".bin"
        with open(src, "rb") as f:
            for i in range(feats.shape[0]):
                r = f.read(40)
                if len(r) < 40:
                    break
                b, hm, fm = unpack_sfen(r[:32])
                fens.append(b.fen())
        n = min(feats.shape[0], len(tgts), len(fens))
        if smoke:
            n = min(n, 8000)
        bins.append({"name": name, "feats": feats, "tgts": tgts,
                     "fens": fens, "n": n})
    total = sum(b["n"] for b in bins)
    E_n = bins[0]["feats"].shape[1]
    print("dataset: %d positions, %d experts, D=%d, %d bins" %
          (total, E_n, D, len(bins)), flush=True)

    # precompute small arrays (HCE, route, targets)
    X_hce_l, X_route_l, Y_l = [], [], []
    global_idx = []  # (bin_idx, local_idx)
    for bi, b in enumerate(bins):
        for i in range(b["n"]):
            board = chess.Board(b["fens"][i])
            try:
                dom = domain(board.fen())
            except Exception:
                dom = "none"
            X_hce_l.append(hce(board))
            X_route_l.append(D2I.get(dom, D2I["none"]))
            Y_l.append(b["tgts"][i])
            global_idx.append((bi, i))
    X_hce = torch.tensor(np.array(X_hce_l, dtype=np.float32))
    X_route = torch.tensor(np.array(X_route_l, dtype=torch.long)
                           ) if False else torch.tensor(X_route_l, dtype=torch.long)
    Y = torch.tensor(np.array(Y_l, dtype=np.float32))
    del X_hce_l, X_route_l, Y_l

    # calibration
    sc = torch.zeros(1, E_n, 1)
    ze = torch.zeros(1, E_n, 1)
    for i in range(min(len(scale), E_n)):
        sc[0, i, 0] = scale[i]
        ze[0, i, 0] = zero[i]

    def fetch(gindices):
        acts = np.zeros((len(gindices), E_n, D), dtype=np.float32)
        for j, (bi, li) in enumerate(gindices):
            acts[j] = bins[bi]["feats"][li].astype(np.float32)
        t = torch.from_numpy(acts)
        t = (t / 255.0) * (sc * 255.0) + ze
        return t / (t.abs().mean() + 1e-9)

    n = total
    perm = list(range(n))
    rng.shuffle(perm)
    n_val = max(1000, n // 20)
    val_idx, tr_idx = perm[:n_val], perm[n_val:]
    HCE_D = X_hce.shape[1]
    N_DOM, N_EXP = len(DOMAINS) - 1, E_n

    class Head(nn.Module):
        def __init__(self):
            super().__init__()
            self.gates = nn.Parameter(torch.ones(N_DOM, N_EXP))
            self.lin = nn.Linear(D * N_EXP + HCE_D + N_EXP, 1)
            nn.init.zeros_(self.lin.weight)
            nn.init.zeros_(self.lin.bias)

        def forward(self, act, hce, route):
            g_sel = self.gates[route]
            gated = act * g_sel[:, :, None]
            routed_w = torch.zeros(act.shape[0], N_EXP)
            routed_w.scatter_(1, route[:, None], 1.0)
            x = torch.cat([gated.flatten(1), hce, routed_w], dim=1)
            return self.lin(x).squeeze(-1)

    head = Head()
    opt = torch.optim.AdamW(head.parameters(), lr=1e-3, weight_decay=1e-6)

    def wp(cp):
        return 1.0 / (1.0 + torch.exp(-cp / 361.0))

    def loss_fn(p, t):
        return ((wp(p) - wp(t)) ** 2).mean()

    best_val = 1e9
    EPOCHS = 3 if smoke else 30
    BS = 4096
    for ep in range(EPOCHS):
        head.train()
        rng.shuffle(tr_idx)
        total_loss = 0
        nb = 0
        for i in range(0, len(tr_idx), BS):
            idx = tr_idx[i:i + BS]
            gi = [global_idx[k] for k in idx]
            acts = fetch(gi)
            pred = head(acts, X_hce[idx], X_route[idx])
            loss = loss_fn(pred, Y[idx])
            opt.zero_grad()
            loss.backward()
            opt.step()
            total_loss += loss.item()
            nb += 1
        head.eval()
        vl, vp_l, vt_l = [], [], []
        with torch.no_grad():
            for i in range(0, len(val_idx), BS):
                idx = val_idx[i:i + BS]
                gi = [global_idx[k] for k in idx]
                acts = fetch(gi)
                vp = head(acts, X_hce[idx], X_route[idx])
                vl.append(loss_fn(vp, Y[idx]).item())
                vp_l.extend(vp.tolist())
                vt_l.extend(Y[idx].tolist())
        vloss = sum(vl) / len(vl)
        corr = np.corrcoef(vp_l, vt_l)[0, 1] if len(vp_l) > 1 else 0
        print("epoch %d: train=%.6f val=%.6f corr=%.4f [%d batches]"
              % (ep, total_loss / max(nb, 1), vloss, corr, nb), flush=True)
        if vloss < best_val:
            best_val = vloss
            torch.save(head.state_dict(), R + "/head_v1.pt")

    g = head.gates.detach().numpy()
    print("\ngates (rows=domains, cols=experts):")
    for di, dn in enumerate(DOMAINS[:-1]):
        print("  %-14s %s" % (dn, " ".join("%.2f" % x for x in g[di])))
    print("DONE best_val=%.6f" % best_val)


if __name__ == "__main__":
    main()
