"""V4-TEACHER TARGET GENERATION — streaming over sidecar parts.

For each part: re-derive FENs from the bin (row-aligned), compute HCE + domain,
run v4-raw forward on GPU, write (fen, v4_eval) pairs to a new bins file.
Output: one .txt per part (fen|v4eval) for student packing.
"""
import sys, os, glob, time
sys.path.insert(0, "/srv/workspace/flychess/src/chess-lab/tools")
import numpy as np
import torch
import chess
import audit_packer

R = "/mnt/cold-raid6/chess-audit"
SF = f"{R}/sidecars_full"
OUT = f"{R}/v4_targets"
E_N, ACT_D, FT_D, HCE_D = 13, 1024, 86896, 36
TEACHER = f"{R}/selfplay_rl/dense_v4_ckpts3/best.pt"

def wp(x):
    return 1.0 / (1.0 + torch.exp(-x / 361.0))

class ClippedReLU(torch.nn.Module):
    def __init__(self, hi=63.0):
        super().__init__()
        self.hi = hi
    def forward(self, x):
        return torch.clamp(x, 0.0, self.hi)

class DenseV4(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.emb = torch.nn.Embedding(FT_D, ACT_D, sparse=True)
        self.lin_ev = torch.nn.Linear(E_N, 1, bias=False)
        self.tail = torch.nn.Sequential(
            torch.nn.Linear(E_N*ACT_D + ACT_D + HCE_D + E_N, 64), ClippedReLU(),
            torch.nn.Linear(64, 32), ClippedReLU(),
            torch.nn.Linear(32, 1),
        )
        self.register_buffer("ev_mean", torch.zeros(E_N))
        self.register_buffer("ev_std", torch.ones(E_N))
        self.register_buffer("a_mean", torch.zeros(E_N, ACT_D))
        self.register_buffer("a_std", torch.ones(E_N, ACT_D))
    def forward(self, ev, act, hce, oh, ftidx):
        x_ev = (ev - self.ev_mean) / self.ev_std
        x_act = (act.float() - self.a_mean) / self.a_std
        mask = (ftidx >= 0).float().unsqueeze(-1)
        raw = (self.emb(ftidx.clamp(min=0)) * mask).sum(dim=1)
        feats = torch.cat([x_act.view(len(ev), -1), raw, hce, oh], dim=1)
        return self.tail(feats).squeeze(-1) + self.lin_ev(x_ev).squeeze(-1)

def hce_fn(board):
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
        files = [0]*8
        for s2 in board.pieces(chess.PAWN, side):
            files[chess.square_file(s2)] += 1
        f += [x/2.0 for x in files]
        f.append(sum(1 for x in files if x >= 2)/4.0)
    return f

def main():
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    os.makedirs(OUT, exist_ok=True)
    model = DenseV4()
    sd = torch.load(TEACHER, map_location="cpu", weights_only=False)
    model.load_state_dict(sd)
    model = model.to(dev).eval()
    from score_experts import domain as domain_of
    DOMS = ["balanced_l0","balanced_l1","balanced_l2","balanced_l3","nvb",
            "nvr","bvr","rv2m","qvmat","oppb","dvoretsky","exchanges","tb"]
    D2I = {d: i for i, d in enumerate(DOMS)}

    parts = sorted(glob.glob(f"{SF}/*_p*.npz"))
    print(f"{len(parts)} parts", flush=True)
    t00 = time.time()
    total = 0
    for pi, part_f in enumerate(parts):
        name = os.path.basename(part_f)[:-4]
        out_f = f"{OUT}/{name}.txt"
        if os.path.exists(out_f):
            continue
        bin_name = name.rsplit("_p", 1)[0]
        bin_f = f"{R}/expert_bins_both/{bin_name}.bin"
        raw = open(bin_f, "rb").read()
        d = np.load(part_f)
        n = d["evals"].shape[0]
        part_no = int(name.rsplit("_p", 1)[1])
        st = part_no * 200000
        fens = []
        hces = np.zeros((n, HCE_D), dtype=np.float32)
        doms = np.zeros(n, dtype=np.int64)
        for i in range(n):
            b = audit_packer.unpack_sfen(raw[(st+i)*40:(st+i)*40+32])[0]
            fens.append(b.fen())
            hces[i] = hce_fn(b)
            try:
                doms[i] = D2I.get(domain_of(b.fen()), 12)
            except Exception:
                doms[i] = 12
        oh = np.zeros((n, E_N), dtype=np.float32)
        oh[np.arange(n), doms] = 1.0
        acts = torch.from_numpy(d["acts"]).to(dev)  # already engine scale in parts
        evals = torch.tensor(d["evals"]).to(dev)
        ft = torch.tensor(d["ft"].astype(np.int64)).to(dev)
        outs = []
        with torch.no_grad():
            for s2 in range(0, n, 8192):
                e = min(s2 + 8192, n)
                pred = model(evals[s2:e], acts[s2:e],
                             torch.tensor(hces[s2:e]).to(dev),
                             torch.tensor(oh[s2:e]).to(dev), ft[s2:e])
                outs.append(pred.cpu().numpy())
        out = np.concatenate(outs)
        with open(out_f, "w") as f:
            for fen, v in zip(fens, out):
                f.write(f"{fen}|{int(round(float(v)))}\n")
        total += n
        if pi % 10 == 0:
            rate = total / max(time.time() - t00, 1)
            print(f"part {pi}/{len(parts)}: {total:,} positions ({rate:.0f}/s)", flush=True)
        del d, acts, evals, ft
        torch.cuda.empty_cache()
    print("V4_TARGETS_DONE", flush=True)

if __name__ == "__main__":
    main()
