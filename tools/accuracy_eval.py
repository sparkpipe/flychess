"""ACCURACY EVAL — 1000 random positions, win-prob correlation vs SF-depth-20 reference.

Candidates (all evaluated in PyTorch where possible, no engine needed for speed):
  nQ (routed expert eval — the incumbent)
  raw v4 head (400k-supervised, ensemble)
  RL-i1 v4
  linear sQ head (the committee)
  full-corpus student will be added when trained
Reference: SF17.1 depth 20 (neutral judge), win-prob space.
Metric: win-prob RMSE + Pearson correlation vs reference.
"""
import sys, os, glob, time, subprocess, random, json
sys.path.insert(0, "/srv/workspace/flychess/src/nnue-pytorch")
sys.path.insert(0, "/srv/workspace/flychess/src/chess-lab/tools")
import numpy as np
import torch
import chess

R = "/mnt/cold-raid6/chess-audit"
SP = f"{R}/selfplay_rl"
E_N, ACT_D, FT_D = 13, 1024, 86896
NPOS = 1000

def wp(cp):
    return 1.0 / (1.0 + np.exp(-np.array(cp, dtype=np.float64) / 361.0))

def sample_positions():
    """1000 random positions from the corpus bins (stratified across bins)."""
    rng = random.Random(1234)
    import audit_packer
    bins = sorted(glob.glob(f"{R}/expert_bins_both/*.bin"))
    fens = []
    per_bin = NPOS // len(bins) + 1
    for b in bins:
        raw = open(b, "rb").read()
        n = len(raw) // 40
        idxs = rng.sample(range(n), min(per_bin, n))
        for i in idxs:
            fens.append(audit_packer.unpack_sfen(raw[i*40:i*40+32])[0].fen())
    rng.shuffle(fens)
    return fens[:NPOS]

def sf20_reference(fens):
    """SF17.1 depth 20, stm-perspective cp -> win prob."""
    out = []
    p = subprocess.Popen(["/usr/games/stockfish"], stdin=subprocess.PIPE,
                         stdout=subprocess.PIPE, stderr=subprocess.DEVNULL,
                         text=True, bufsize=1)
    p.stdin.write("uci\nisready\n"); p.stdin.flush()
    while "readyok" not in p.stdout.readline():
        pass
    for i, fen in enumerate(fens):
        p.stdin.write(f"position fen {fen}\ngo depth 20\n"); p.stdin.flush()
        cp = None
        for _ in range(3000):
            line = p.stdout.readline()
            if line.startswith("info") and " score cp " in line and " pv " in line:
                import re
                m = re.search(r"score cp (-?\d+)", line)
                if m:
                    cp = int(m.group(1))
            elif line.startswith("bestmove"):
                break
        # convert stm-cp to white-persp
        stm = chess.Board(fen).turn
        cp_white = cp if stm == chess.WHITE else -cp if cp is not None else None
        out.append(cp_white)
        if (i + 1) % 200 == 0:
            print(f"  sf20: {i+1}/{len(fens)}", flush=True)
    p.stdin.write("quit\n"); p.stdin.flush()
    return out

def eval_candidates(fens):
    """Each candidate returns white-perspective cp per fen."""
    import extract_stack_features as E
    import data_loader
    from score_experts import domain as domain_of
    nets = [E.load_expert(e) for e in
            ["balanced_l0","balanced_l1","balanced_l2","balanced_l3",
             "nvb","nvr","bvr","rv2m","qvmat","oppb",
             "dvoretsky","exchanges","tactics"]]
    caps = [[] for _ in nets]
    hooks = []
    for mdl, cap in zip(nets, caps):
        def mk(buf):
            def h(mod, inp, out):
                o = out[0] if isinstance(out, tuple) else out
                buf.append(o.detach().cpu().numpy().copy())
            return h
        hooks.append(mdl.input.register_forward_hook(mk(cap)))

    evals = np.zeros((len(fens), E_N), dtype=np.float32)
    acts = np.zeros((len(fens), E_N, ACT_D), dtype=np.uint8)
    doms = np.zeros(len(fens), dtype=np.int64)
    DOMS = ["balanced_l0","balanced_l1","balanced_l2","balanced_l3","nvb",
            "nvr","bvr","rv2m","qvmat","oppb","dvoretsky","exchanges","tb"]
    D2I = {d: i for i, d in enumerate(DOMS)}
    B = 256
    for s in range(0, len(fens), B):
        chunk = fens[s:s+B]
        bs = data_loader.get_sparse_batch_from_fens(
            "Full_Threats+PP_3Wide+HalfKAv2_hm", chunk,
            [0]*len(chunk), [1]*len(chunk), [0]*len(chunk))
        us, them, wi, bi, o, sc, pc = bs.contents.get_tensors("cuda")
        for cap in caps:
            cap.clear()
        with torch.no_grad():
            for mdl in nets:
                mdl.forward(us, them, wi, bi, pc)
        for ei in range(E_N):
            arr = caps[ei][-1]
            acts[s:s+len(chunk), ei, :] = (arr[:, 0, :ACT_D] if arr.ndim == 3 else arr[:, :ACT_D])
            del caps[ei][:-1]
        for ei, mdl in enumerate(nets):
            with torch.no_grad():
                vs = mdl.forward(us, them, wi, bi, pc)
            evals[s:s+len(chunk), ei] = [float(v) * mdl.quantization.nnue2score for v in vs]
        data_loader.destroy_sparse_batch(bs)
    for h in hooks:
        h.remove()
    for i, fen in enumerate(fens):
        try:
            d = domain_of(fen)
        except Exception:
            d = "tb"
        doms[i] = D2I.get(d, 12)
    acts_eng = acts >> 1  # engine scale
    return evals, acts_eng, doms

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
            torch.nn.Linear(E_N*ACT_D + ACT_D + 36 + E_N, 64), ClippedReLU(),
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
    def load(self, sd):
        self.load_state_dict(sd)

def rawft_all(fens):
    out = []
    p = subprocess.Popen(["/srv/workspace/flychess/src/Stockfish-act/src/stockfish"],
                         stdin=subprocess.PIPE, stdout=subprocess.PIPE, text=True, bufsize=1)
    p.stdin.write("uci\n"); p.stdin.flush(); time.sleep(1)
    for fen in fens:
        p.stdin.write(f"rawft {fen}\n"); p.stdin.flush()
        line = p.stdout.readline()
        while line and not line.startswith("rawft"):
            line = p.stdout.readline()
        out.append([int(t) for t in line.split()[1:]])
    p.stdin.write("quit\n"); p.stdin.flush()
    return out

def main():
    fens = sample_positions()
    print(f"{len(fens)} positions sampled", flush=True)

    ref = sf20_reference(fens)
    valid = [i for i, c in enumerate(ref) if c is not None]
    ref_wp = wp([ref[i] for i in valid])
    print(f"reference done: {len(valid)} valid", flush=True)

    evals, acts, doms = eval_candidates([fens[i] for i in valid])
    ftlists = rawft_all([fens[i] for i in valid])
    hces = np.array([hce_fn(chess.Board(fens[i])) for i in valid], dtype=np.float32)
    oh = np.zeros((len(valid), E_N), dtype=np.float32)
    oh[np.arange(len(valid)), doms] = 1.0

    results = {}
    def score(name, cp_stm_white):
        p = wp(cp_stm_white)
        rmse = float(np.sqrt(((p - ref_wp) ** 2).mean()))
        corr = float(np.corrcoef(p, ref_wp)[0, 1])
        results[name] = {"wp_rmse": round(rmse, 4), "corr": round(corr, 4)}
        print(f"{name:22s} wp-RMSE {rmse:.4f}  corr {corr:.4f}", flush=True)

    # nQ: routed expert eval (the slot domain picks), white-persp
    routed = evals[np.arange(len(valid)), doms] * np.where(
        np.array([1 if chess.Board(fens[i]).turn else -1 for i in valid]) > 0, 1, -1)
    score("nQ (routed)", routed)
    # 13-expert consensus (white persp)
    cons = evals.mean(axis=1) * np.where(
        np.array([1 if chess.Board(fens[i]).turn else -1 for i in valid]) > 0, 1, -1)
    score("consensus-13", cons)

    # v4 heads
    MAXF = 160
    ft = np.full((len(valid), MAXF), -1, dtype=np.int64)
    for k, lst in enumerate(ftlists):
        c = min(len(lst), MAXF)
        ft[k, :c] = lst[:c]
    for name, ck in [("v4-raw(400k)", f"{SP}/dense_v4_ckpts3/best.pt"),
                     ("v4-RL-i1", f"{R}/rl_v4/ckpt_i1.pt")]:
        if not os.path.exists(ck):
            print(f"{name}: missing", flush=True)
            continue
        sd = torch.load(ck, map_location="cpu", weights_only=False)
        m = DenseV4(); m.load(sd); m.eval()
        with torch.no_grad():
            pred = m(torch.tensor(evals), torch.from_numpy(acts), torch.tensor(hces),
                     torch.tensor(oh), torch.tensor(ft)).numpy()
        # pred is stm-persp cp (targets were stm) -> white persp
        stm_sign = np.array([1 if chess.Board(fens[i]).turn else -1 for i in valid])
        score(name, pred * stm_sign)

    # linear sQ head (the committee) via .evh weights
    sys.path.insert(0, "/srv/workspace/flychess/src/chess-lab/tools")
    import rl_iter as RI
    g, lw, lb, mean, std = RI.parse_evh(f"{R}/selfplay_rl/head_sQ_gain.evh"
                                        if os.path.exists(f"{R}/selfplay_rl/head_sQ_gain.evh")
                                        else f"{R}/selfplay_rl/head_sQ.evh")
    norm = (evals - mean[None, :]) / std[None, :]
    gated = norm * g[doms]
    ohv = np.zeros((len(valid), E_N), dtype=np.float32)
    ohv[np.arange(len(valid)), doms] = 1.0
    X = np.concatenate([gated, hces, ohv], axis=1)
    lin_out = X @ lw + lb
    score("sQ-linear(gain)", lin_out * stm_sign)

    json.dump(results, open(f"{SP}/accuracy_eval.json", "w"), indent=1)
    print("RESULTS_SAVED", flush=True)

if __name__ == "__main__":
    main()
