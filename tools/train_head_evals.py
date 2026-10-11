"""STACKED HEAD v2 — uses expert SCALAR EVALS as features (not activations).
Architecture: 14 expert evals + 36 HCE features + 14 routing one-hot → head → eval.
This is the correct stacking ensemble: each input is an expert's FULL evaluation."""
import sys, os, glob, struct, time
import numpy as np
import torch, torch.nn as nn

sys.path.insert(0, "/srv/workspace/flychess/src/chess-lab/tools")
from audit_packer import unpack_sfen
import chess
from score_experts import domain

R = "/mnt/cold-raid6/chess-audit"
DOMAINS = ["balanced_l0","balanced_l1","balanced_l2","balanced_l3","nvb",
           "nvr","bvr","rv2m","qvmat","oppb","dvoretsky","exchanges","none"]
D2I = {d:i for i,d in enumerate(DOMAINS)}
EXPERTS = ["balanced_l0","balanced_l1","balanced_l2","balanced_l3",
           "nvb","nvr","bvr","rv2m","qvmat","oppb",
           "dvoretsky","exchanges","tactics"]
E2I = {e:i for i,e in enumerate(EXPERTS)}


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
    cache = sys.argv[1] if len(sys.argv) > 1 and not sys.argv[1].startswith("--") else "/srv/workspace/flychess/eval_cache"
    out_name = sys.argv[2] if len(sys.argv) > 2 and not sys.argv[2].startswith("--") else "head_evals_v1"
    device = "cuda" if torch.cuda.is_available() else "cpu"
    torch.manual_seed(42)

    # Load eval features + HCE + domain + targets
    X_eval_l, X_hce_l, X_dom_l, Y_l = [], [], [], []
    total = 0
    for ef in sorted(glob.glob(cache + "/*.evals.npy")):
        name = os.path.basename(ef)[:-10]
        evals = np.load(ef)
        # read fens from source bin for HCE/domain
        src = R + "/expert_bins_both/" + name + ".bin"
        tgts_f = "/srv/workspace/flychess/cache/" + name + ".targets.npy"
        hce_f = "/srv/workspace/flychess/cache/" + name + ".hce.npz"
        if not os.path.exists(hce_f):
            continue
        hd = np.load(hce_f)
        tgts = np.load(tgts_f)
        n = min(evals.shape[0], tgts.shape[0], hd["hce"].shape[0])
        if smoke: n = min(n, 5000)
        X_eval_l.append(evals[:n])
        X_hce_l.append(hd["hce"][:n])
        X_dom_l.append(hd["domain"][:n])
        Y_l.append(tgts[:n])
        total += n
        print(f"  {name}: {n}", flush=True)

    X_eval = torch.tensor(np.concatenate(X_eval_l), dtype=torch.float32)
    X_hce = torch.tensor(np.concatenate(X_hce_l), dtype=torch.float32)
    X_dom = torch.tensor(np.concatenate(X_dom_l), dtype=torch.long)
    Y = torch.tensor(np.concatenate(Y_l), dtype=torch.float32)
    del X_eval_l, X_hce_l, X_dom_l, Y_l
    E_n = X_eval.shape[1]
    HCE_D = X_hce.shape[1]
    print(f"total: {total} pos, evals={X_eval.shape}, hce={X_hce.shape}", flush=True)

    # to GPU
    X_eval = X_eval.to(device)
    X_hce = X_hce.to(device)
    X_dom = X_dom.to(device)
    Y = Y.to(device)

    # normalize evals (each expert has different scale)
    eval_mean = X_eval.mean(dim=0, keepdim=True)
    eval_std = X_eval.std(dim=0, keepdim=True).clamp(min=1.0)
    X_eval = (X_eval - eval_mean) / eval_std

    N_DOM = len(DOMAINS) - 1
    INPUT_D = E_n + HCE_D + E_n  # evals + hce + one-hot

    class Head(nn.Module):
        def __init__(self):
            super().__init__()
            # routing init: 1.0 on diagonal (hard routing), 0.0 elsewhere.
            # Training can only improve from here — any non-zero off-diagonal
            # is a learned blend that the data says helps.
            gates_init = torch.zeros(N_DOM, E_n)
            for d in range(min(N_DOM, E_n)):
                gates_init[d, d] = 1.0
            if N_DOM > E_n:  # "none" domain falls back to expert 0
                gates_init[E_n, 0] = 1.0
            self.gates = nn.Parameter(gates_init)
            self.lin = nn.Linear(INPUT_D, 1)
            nn.init.zeros_(self.lin.weight)
            nn.init.zeros_(self.lin.bias)
        def forward(self, ev, hce, dom):
            g = self.gates[dom]
            gated = ev * g
            oh = torch.zeros(ev.shape[0], E_n, device=ev.device)
            oh.scatter_(1, dom[:, None], 1.0)
            x = torch.cat([gated, hce, oh], dim=1)
            return self.lin(x).squeeze(-1)

    head = Head().to(device)
    opt = torch.optim.AdamW(head.parameters(), lr=3e-3, weight_decay=1e-5)

    # resume
    ckpt_path = f"/srv/workspace/flychess/logs/{out_name}_ckpt.pt"
    start_ep, best_val = 0, 1e9
    if os.path.exists(ckpt_path):
        ck = torch.load(ckpt_path, map_location=device, weights_only=False)
        head.load_state_dict(ck["model"]); opt.load_state_dict(ck["optimizer"])
        best_val = ck["best_val"]; start_ep = ck["epoch"] + 1
        print(f"RESUMED epoch {start_ep}", flush=True)

    def wp(cp): return 1.0 / (1.0 + torch.exp(-cp / 361.0))
    def loss_fn(p, t): return ((wp(p) - wp(t)) ** 2).mean()

    perm = np.random.RandomState(42).permutation(total)
    n_val = max(1000, total // 20)
    val_idx = torch.from_numpy(perm[:n_val]).long().to(device)
    tr_idx = torch.from_numpy(perm[n_val:]).long().to(device)

    EPOCHS = 5 if smoke else 500
    BS = 16384
    for ep in range(start_ep, EPOCHS):
        head.train()
        shuffle = tr_idx[torch.randperm(len(tr_idx), device=device)]
        tl, nb = 0.0, 0
        for i in range(0, len(shuffle), BS):
            idx = shuffle[i:i+BS]
            pred = head(X_eval[idx], X_hce[idx], X_dom[idx])
            loss = loss_fn(pred, Y[idx])
            opt.zero_grad(); loss.backward(); opt.step()
            tl += loss.item(); nb += 1
        head.eval()
        with torch.no_grad():
            vl, vps, vts = [], [], []
            for i in range(0, len(val_idx), BS):
                idx = val_idx[i:i+BS]
                vp = head(X_eval[idx], X_hce[idx], X_dom[idx])
                vl.append(loss_fn(vp, Y[idx]).item())
                vps.extend(vp.cpu().tolist()); vts.extend(Y[idx].cpu().tolist())
        vloss = sum(vl) / len(vl)
        corr = np.corrcoef(vps, vts)[0, 1] if len(vps) > 1 else 0
        print(f"epoch {ep}: train={tl/max(nb,1):.6f} val={vloss:.6f} corr={corr:.4f}", flush=True)
        if vloss < best_val:
            best_val = vloss
            torch.save(head.state_dict(), f"{R}/{out_name}.pt")
        torch.save({"model": head.state_dict(), "optimizer": opt.state_dict(),
                    "best_val": best_val, "epoch": ep,
                    "eval_mean": eval_mean.cpu(), "eval_std": eval_std.cpu()},
                   ckpt_path)

    # save engine format (evals-based, not activations)
    sd = head.state_dict()
    gates_np = sd["gates"].cpu().numpy()
    lw = sd["lin.weight"].cpu().numpy().flatten()
    lb = sd["lin.bias"].cpu().numpy().flatten()
    em = eval_mean.cpu().numpy()
    es = eval_std.cpu().numpy()
    path = f"{R}/{out_name}.evh"
    with open(path, "wb") as f:
        f.write(struct.pack("<IIIII", 0x45564C48, 1, E_n, HCE_D, N_DOM))
        f.write(gates_np.astype(np.float32).tobytes())
        f.write(lw.astype(np.float32).tobytes())
        f.write(lb.astype(np.float32).tobytes())
        f.write(em.astype(np.float32).tobytes())
        f.write(es.astype(np.float32).tobytes())
    print(f"DONE best_val={best_val:.6f}, engine: {path}", flush=True)
    print(f"gates:\n{gates_np}")


if __name__ == "__main__":
    main()
