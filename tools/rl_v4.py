"""V4 SELF-PLAY RL — Chess960 loop for the dense nonlinear head.

Cycle: self-play (act-fork + current v4 head, Chess960 book) →
  extract (13 evals + 13x1024 acts + rawft + HCE + outcome targets) →
  fine-tune v4 checkpoint (bootstrap lineage, low LR, wp-of-outcome loss) →
  export .evh v4 → report weight delta.
Usage: rl_v4.py <tc_seconds> <n_rounds> [--resume ckpt]
"""
import sys, os, glob, time, json, subprocess, struct, math
sys.path.insert(0, "/srv/workspace/flychess/src/nnue-pytorch")
sys.path.insert(0, "/srv/workspace/flychess/src/chess-lab/tools")
import numpy as np
import torch, torch.nn as nn
import chess, chess.pgn

R = "/mnt/cold-raid6/chess-audit"
SP = f"{R}/selfplay_rl"
RL = f"{R}/rl_v4"
ACT_BIN = "/srv/workspace/flychess/src/Stockfish-act/src/stockfish"
FC = "/srv/workspace/flychess/src/arena/fastchess-linux-x86-64/fastchess"
E_N, HCE_D, ACT_D, FT_D = 13, 36, 1024, 86896
H1, H2 = 64, 32
MAXF = 160

EXPERTS = ["balanced_l0","balanced_l1","balanced_l2","balanced_l3",
           "nvb","nvr","bvr","rv2m","qvmat","oppb",
           "dvoretsky","exchanges","tactics"]

def wp(x):
    return 1.0 / (1.0 + torch.exp(-x / 361.0))

class ClippedReLU(nn.Module):
    def __init__(self, hi=63.0):
        super().__init__()
        self.hi = hi
    def forward(self, x):
        return torch.clamp(x, 0.0, self.hi)

class DenseV4(nn.Module):
    def __init__(self, ev_mean, ev_std, a_mean, a_std):
        super().__init__()
        self.emb = nn.Embedding(FT_D, ACT_D, sparse=True)
        self.lin_ev = nn.Linear(E_N, 1, bias=False)
        self.tail = nn.Sequential(
            nn.Linear(E_N * ACT_D + ACT_D + HCE_D + E_N, H1), ClippedReLU(),
            nn.Linear(H1, H2), ClippedReLU(),
            nn.Linear(H2, 1),
        )
        self.register_buffer("ev_mean", ev_mean.clone())
        self.register_buffer("ev_std", ev_std.clone())
        self.register_buffer("a_mean", a_mean.clone().view(E_N, ACT_D))
        self.register_buffer("a_std", a_std.clone().view(E_N, ACT_D))
    def forward(self, ev, act, hce, oh, ftidx):
        x_ev = (ev - self.ev_mean) / self.ev_std
        x_act = (act.float() - self.a_mean) / self.a_std
        mask = (ftidx >= 0).float().unsqueeze(-1)
        raw = (self.emb(ftidx.clamp(min=0)) * mask).sum(dim=1)
        feats = torch.cat([x_act.view(len(ev), -1), raw, hce, oh], dim=1)
        return self.tail(feats).squeeze(-1) + self.lin_ev(x_ev).squeeze(-1)

def opts():
    N = f"{R}/nets"
    o = [f"option.EvalFile={N}/balanced_l0.nnue"]
    o += [f"option.EvalFile{i+2}={N}/{e}.nnue" for i, e in enumerate(EXPERTS[1:])]
    return " ".join(o)

def gen_games(tc, n_rounds, head_evh, gdir):
    os.makedirs(gdir, exist_ok=True)
    cmd = [FC,
           "-engine", "name=sp_a", f"cmd={ACT_BIN}", *opts().split(),
           f"option.StackHead={head_evh}",
           "-engine", "name=sp_b", f"cmd={ACT_BIN}", *opts().split(),
           f"option.StackHead={head_evh}",
           "-each", "proto=uci", f"st={tc}", "timemargin=200",
           "-rounds", str(n_rounds), "-repeat",
           "-openings", f"file={R}/chess960_book.epd", "format=epd", "order=sequential",
           "-pgnout", f"file={gdir}/games.pgn",
           "-concurrency", "10"]
    print(f"gen: {n_rounds*2} games @ {tc}s (StackHead={os.path.basename(head_evh)})", flush=True)
    subprocess.run(cmd, capture_output=True, text=True, timeout=14400)

def parse_games(gdir):
    fens, zs = [], []
    for pgn_f in glob.glob(f"{gdir}/*.pgn"):
        with open(pgn_f) as f:
            while True:
                g = chess.pgn.read_game(f)
                if g is None:
                    break
                res = g.headers.get("Result", "*")
                ow = 1.0 if res == "1-0" else 0.0 if res == "0-1" else 0.5
                b = g.board(); ply = 0
                for node in g.mainline():
                    if ply % 4 == 0:
                        fens.append(b.fen())
                        zs.append(ow if b.turn == chess.WHITE else 1.0 - ow)
                    b.push(node.move); ply += 1
    print(f"parse: {len(fens)} positions", flush=True)
    return fens, zs

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
        files = [0] * 8
        for s2 in board.pieces(chess.PAWN, side):
            files[chess.square_file(s2)] += 1
        f += [x / 2.0 for x in files]
        f.append(sum(1 for x in files if x >= 2) / 4.0)
    return f

def extract_features(fens):
    """All four groups, engine-scale acts, one pass."""
    import extract_stack_features as E
    import data_loader
    from score_experts import domain as domain_of

    nets = [E.load_expert(e) for e in EXPERTS]
    caps = [[] for _ in nets]
    hooks = []
    for mdl, cap in zip(nets, caps):
        def mk(buf):
            def h(mod, inp, out):
                o = out[0] if isinstance(out, tuple) else out
                buf.append(o.detach().cpu().numpy().copy())
            return h
        hooks.append(mdl.input.register_forward_hook(mk(cap)))

    eng = subprocess.Popen([ACT_BIN], stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                           text=True, bufsize=1)
    eng.stdin.write("uci\n"); eng.stdin.flush(); time.sleep(1.0)

    evals = np.zeros((len(fens), E_N), dtype=np.float32)
    acts = np.zeros((len(fens), E_N, ACT_D), dtype=np.uint8)
    hces = np.zeros((len(fens), HCE_D), dtype=np.float32)
    doms = np.zeros(len(fens), dtype=np.int64)
    DOMS = ["balanced_l0","balanced_l1","balanced_l2","balanced_l3","nvb",
            "nvr","bvr","rv2m","qvmat","oppb","dvoretsky","exchanges","tb"]
    D2I = {d: i for i, d in enumerate(DOMS)}

    B = 512
    for s2 in range(0, len(fens), B):
        chunk = fens[s2:s2+B]
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
            acts[s2:s2+len(chunk), ei, :] = (arr[:, 0, :ACT_D] if arr.ndim == 3 else arr[:, :ACT_D])
            del caps[ei][:-1]
        for ei, mdl in enumerate(nets):
            with torch.no_grad():
                vs = mdl.forward(us, them, wi, bi, pc)
            evals[s2:s2+len(chunk), ei] = [float(v) * mdl.quantization.nnue2score for v in vs]
        data_loader.destroy_sparse_batch(bs)
    for h in hooks:
        h.remove()
    # NOTE: acts from the python hook are cache-scale (2x engine). Convert:
    acts = acts >> 1
    for i, fen in enumerate(fens):
        b = chess.Board(fen)
        hces[i] = hce_fn(b)
        try:
            d = domain_of(fen)
        except Exception:
            d = "tb"
        doms[i] = D2I.get(d, 12)
    ft = np.full((len(fens), MAXF), -1, dtype=np.int32)
    cnt = np.zeros(len(fens), dtype=np.int32)
    for k, fen in enumerate(fens):
        eng.stdin.write(f"rawft {fen}\n"); eng.stdin.flush()
        line = eng.stdout.readline()
        while line and not line.startswith("rawft"):
            line = eng.stdout.readline()
        toks = line.split()[1:]
        c = min(len(toks), MAXF)
        ft[k, :c] = [int(t) for t in toks[:c]]
        cnt[k] = c
    eng.stdin.write("quit\n"); eng.stdin.flush()
    return evals, acts, hces, doms, ft, cnt

def load_model(ckpt_path, ev_mean, ev_std, a_mean, a_std):
    m = DenseV4(ev_mean, ev_std, a_mean, a_std)
    sd = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    m.load_state_dict(sd)
    return m

def fine_tune(model, evals, acts, hces, doms, ft, cnt, zs, epochs=30):
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    model = model.to(dev)
    N = len(zs)
    Y = torch.tensor(np.array(zs, dtype=np.float32))
    X_ev = torch.tensor(evals)
    X_act = torch.from_numpy(acts)
    X_hce = torch.tensor(hces)
    oh = np.zeros((N, E_N), dtype=np.float32)
    oh[np.arange(N), doms] = 1.0
    X_oh = torch.tensor(oh)
    X_ft = torch.tensor(ft.astype(np.int64))

    opt_emb = torch.optim.SparseAdam([model.emb.weight], lr=1e-4)
    opt = torch.optim.AdamW([p for n, p in model.named_parameters()
                             if not n.startswith("emb.")], lr=5e-5, weight_decay=1e-5)
    old = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()
           if k.startswith("tail.") or k.startswith("lin_ev")}
    BS = 2048
    for ep in range(epochs):
        model.train()
        perm = np.random.permutation(N)
        tl, nb = 0.0, 0
        for i in range(0, N, BS):
            idx = torch.tensor(perm[i:i+BS])
            pred = model(X_ev[idx].to(dev), X_act[idx].to(dev), X_hce[idx].to(dev),
                         X_oh[idx].to(dev), X_ft[idx].to(dev))
            loss = ((wp(pred) - Y[idx].to(dev)) ** 2).mean()
            opt_emb.zero_grad(); opt.zero_grad()
            loss.backward()
            opt_emb.step(); opt.step()
            tl += loss.item(); nb += 1
        if ep == 0 or ep == epochs - 1:
            print(f"  rl_epoch {ep}: wp-loss {tl/max(nb,1):.6f}", flush=True)
    model = model.cpu()
    change = math.sqrt(sum(((old[k].float() - model.state_dict()[k].float()) ** 2).sum().item()
                           for k in old))
    return model, change

def main():
    tc = int(sys.argv[1]) if len(sys.argv) > 1 else 1
    n_rounds = int(sys.argv[2]) if len(sys.argv) > 2 else 960
    resume = None
    if "--resume" in sys.argv:
        resume = sys.argv[sys.argv.index("--resume") + 1]
    os.makedirs(RL, exist_ok=True)
    state_f = f"{RL}/state.json"
    state = json.load(open(state_f)) if os.path.exists(state_f) else {"iter": 0}

    base_ckpt = resume or f"{SP}/dense_v4_ckpts3/best.pt"
    ckpt = base_ckpt
    head_evh = f"{RL}/head_rl.evh"

    for iteration in range(state["iter"], state["iter"] + 1):  # one cycle per invocation
        # 0. export current ckpt so self-play uses it
        subprocess.run(["python3",
                        "/srv/workspace/flychess/src/chess-lab/tools/export_dense_v4.py",
                        ckpt, head_evh], check=True, capture_output=True)
        # 1. self-play
        gdir = f"{RL}/games_i{iteration}"
        gen_games(tc, n_rounds, head_evh, gdir)
        # 2. parse
        fens, zs = parse_games(gdir)
        MIN_POS = int(os.environ.get("RL_MIN_POS", "500"))
        if len(fens) < MIN_POS:
            print("too few positions, abort"); return
        # 3. features
        evals, acts, hces, doms, ft, cnt = extract_features(fens)
        # 4. fine-tune from current ckpt
        sd = torch.load(ckpt, map_location="cpu", weights_only=False)
        model = load_model(ckpt, sd["ev_mean"], sd["ev_std"], sd["a_mean"].flatten(), sd["a_std"].flatten())
        model, change = fine_tune(model, evals, acts, hces, doms, ft, cnt, zs)
        new_ckpt = f"{RL}/ckpt_i{iteration+1}.pt"
        torch.save(model.state_dict(), new_ckpt)
        print(f"  weight change L2: {change:.4f}", flush=True)
        if os.path.exists(f"{SP}/gate_cache/features.npz"):
            os.system(f"python3 /srv/workspace/flychess/src/chess-lab/tools/accuracy_gate.py {new_ckpt} rl-i{iteration+1} >> /mnt/cold-raid6/chess-audit/selfplay_rl/gate_runs.log 2>&1")
        state["iter"] = iteration + 1
        state["last_change"] = change
        json.dump(state, open(state_f, "w"))
        ckpt = new_ckpt
    print("RL_CYCLE_DONE", flush=True)

if __name__ == "__main__":
    main()
