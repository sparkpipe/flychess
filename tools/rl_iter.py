"""SELF-PLAY RL ITERATION — trains the DEPLOYABLE stacked eval head (.evh).

One cycle:
  1. self-play games WITH the stacked head on (option.StackHead) — on-policy
  2. parse positions + side-to-move outcomes (win-prob targets)
  3. extract 13 expert scalar evals on GPU (STM-relative, engine-consistent —
     same pipeline the deployed head was trained on)
  4. fine-tune gates + linear head on win-prob loss vs game outcome
  5. export engine .evh + measure weight L2 change (TC-advancement signal)

Engine head format (eval_head.h, magic EVLH 0x45564C48):
  gates[n_dom][13] + lin_w[62 = 13 evals + 36 hce + 13 onehot] + lin_b
  + eval_mean[13] + eval_std[13].  Stats stay FROZEN from the bootstrap head;
  RL only moves gates/lin/bias.  Domains are router slots 0..12 (12 = EX_TB;
  python "tb"/"none"/unknown fall there).
"""
import subprocess, os, sys, json, time, math, glob, struct, shutil
import numpy as np
import torch
import chess
import chess.pgn

R = "/mnt/cold-raid6/chess-audit"
SP = f"{R}/selfplay_rl"
ARENA = "/srv/workspace/flychess/src/arena"
TOOLS = "/srv/workspace/flychess/src/chess-lab/tools"
FORK = "/srv/workspace/flychess/src/Stockfish/src/stockfish"
BOOT_EVH = f"{R}/head_evals_v1.evh"      # deployed 6-0 head (bootstrap)
CUR_EVH = f"{SP}/head_rl_current.evh"      # latest RL head (self-play uses this)

EXPERTS = ["balanced_l0", "balanced_l1", "balanced_l2", "balanced_l3",
           "nvb", "nvr", "bvr", "rv2m", "qvmat", "oppb",
           "dvoretsky", "exchanges", "tactics"]
E_N = len(EXPERTS)          # 13
HCE_D = 36
N_DOM = 13                  # router slots 0..12 (slot 12 = EX_TB)
EVH_MAGIC = 0x45564C48

# router-domain names; index == engine slot.  slot 12 holds tactics.nnue but
# is the EX_TB routing branch — python domain() returns "tb"/"none" there.
DOMS = EXPERTS[:12] + ["tb"]
D2I = {d: i for i, d in enumerate(DOMS)}


def engine_opts(head_evh):
    o = [f"option.EvalFile={R}/nets/{EXPERTS[0]}.nnue"]
    o += [f"option.EvalFile{i + 2}={R}/nets/{e}.nnue" for i, e in enumerate(EXPERTS[1:])]
    o.append(f"option.StackHead={head_evh}")
    return o


def parse_evh(path):
    """Read an engine .evh → gates(13,13), lin_w(62), lin_b, mean(13), std(13).
    Files written with 12 domain rows get row 12 (EX_TB) filled at 0.3/1.0."""
    b = open(path, "rb").read()
    magic, ver, n_e, hce, n_d = struct.unpack_from("<IIIII", b, 0)
    assert magic == EVH_MAGIC and ver == 1 and n_e == E_N and hce == HCE_D
    off = 20
    gates = np.frombuffer(b, dtype="<f4", count=n_d * n_e, offset=off).reshape(n_d, n_e).copy()
    off += n_d * n_e * 4
    lin_w = np.frombuffer(b, dtype="<f4", count=E_N + HCE_D + E_N, offset=off).copy()
    off += (E_N + HCE_D + E_N) * 4
    lin_b = float(struct.unpack_from("<f", b, off)[0]); off += 4
    mean = np.frombuffer(b, dtype="<f4", count=n_e, offset=off).copy(); off += n_e * 4
    std = np.frombuffer(b, dtype="<f4", count=n_e, offset=off).copy()
    if n_d < N_DOM:  # bootstrap file has 12 rows — give EX_TB a sane init
        row = np.full(E_N, 0.3, dtype=np.float32); row[E_N - 1] = 1.0
        gates = np.vstack([gates, row[None, :]])
    return gates, lin_w, lin_b, mean, std


def write_evh(path, gates, lin_w, lin_b, mean, std):
    with open(path, "wb") as f:
        f.write(struct.pack("<IIIII", EVH_MAGIC, 1, E_N, HCE_D, N_DOM))
        f.write(np.ascontiguousarray(gates, dtype="<f4").tobytes())
        f.write(np.ascontiguousarray(lin_w, dtype="<f4").tobytes())
        f.write(struct.pack("<f", lin_b))
        f.write(np.ascontiguousarray(mean, dtype="<f4").tobytes())
        f.write(np.ascontiguousarray(std, dtype="<f4").tobytes())


def gen_games(tc, n_rounds, head_evh):
    ts = time.strftime("%Y%m%d_%H%M%S")
    gdir = f"{SP}/games_{ts}"
    os.makedirs(gdir, exist_ok=True)
    opts = engine_opts(head_evh)
    cmd = [f"{ARENA}/fastchess-linux-x86-64/fastchess",
           "-engine", "name=sp_a", f"cmd={FORK}", *opts,
           "-engine", "name=sp_b", f"cmd={FORK}", *opts,
           "-each", "proto=uci", f"st={tc}", "timemargin=200",
           "-rounds", str(n_rounds), "-repeat",
           "-openings", f"file={R}/chess960_book.epd", "format=epd", "order=sequential",
           "-pgnout", f"file={gdir}/games.pgn",
           "-concurrency", "10"]
    print(f"gen: {n_rounds * 2} games at {tc}s/move (StackHead={os.path.basename(head_evh)})", flush=True)
    subprocess.run(cmd, capture_output=True, text=True, timeout=7200)
    return gdir


def parse_games(gdir):
    """Positions (every 4th ply) + outcome from the SIDE TO MOVE's view."""
    fens, zs = [], []
    for pgn_f in glob.glob(f"{gdir}/*.pgn"):
        with open(pgn_f) as f:
            while True:
                g = chess.pgn.read_game(f)
                if g is None:
                    break
                res = g.headers.get("Result", "*")
                ow = 1.0 if res == "1-0" else 0.0 if res == "0-1" else 0.5
                board = g.board()
                ply = 0
                for node in g.mainline():
                    if ply % 4 == 0:
                        fens.append(board.fen())
                        zs.append(ow if board.turn == chess.WHITE else 1.0 - ow)
                    board.push(node.move)
                    ply += 1
    print(f"parse: {len(fens)} positions from games", flush=True)
    return fens, zs


def extract_expert_evals(fens):
    """13 expert scalar evals per position. STM-relative (verified:
    white-up-Q/black-to-move -> -2501cp). Same pipeline as the deployed head."""
    sys.path.insert(0, TOOLS)
    import extract_stack_features as E
    import data_loader

    nets = [E.load_expert(e) for e in EXPERTS]
    assert all(n is not None for n in nets), "expert load failure"
    evals = np.zeros((len(fens), E_N), dtype=np.float32)
    BATCH = 512
    for start in range(0, len(fens), BATCH):
        chunk = fens[start:start + BATCH]
        bs = data_loader.get_sparse_batch_from_fens(
            "Full_Threats+PP_3Wide+HalfKAv2_hm", chunk,
            [0] * len(chunk), [1] * len(chunk), [0] * len(chunk))
        us, them, wi, bi, o, s, pc = bs.contents.get_tensors("cuda")
        for ei, mdl in enumerate(nets):
            with torch.no_grad():
                vs = mdl.forward(us, them, wi, bi, pc)
            evals[start:start + len(chunk), ei] = [
                float(v) * mdl.quantization.nnue2score for v in vs]
        data_loader.destroy_sparse_batch(bs)
    del nets
    torch.cuda.empty_cache()
    return evals


def hce_fn(board):
    """Mirror of eval_head.h compute_hce() — same order, same scales."""
    f = []
    for pt, v in [(chess.PAWN, 1), (chess.KNIGHT, 3), (chess.BISHOP, 3),
                  (chess.ROOK, 5), (chess.QUEEN, 9)]:
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


def compute_hce_batch(fens):
    from score_experts import domain as domain_of
    hces = np.zeros((len(fens), HCE_D), dtype=np.float32)
    doms = np.zeros(len(fens), dtype=np.int64)
    for i, fen in enumerate(fens):
        b = chess.Board(fen)
        hces[i] = hce_fn(b)
        try:
            d = domain_of(fen)
        except Exception:
            d = "tb"
        doms[i] = D2I.get(d, 12)   # unknown/tb/none -> slot 12 (EX_TB)
    assert doms.min() >= 0 and doms.max() <= 12
    return hces, doms


def build_head(bootstrap=None, prev_pt=None):
    """Engine-format head. gates/lin trainable; eval stats frozen buffers."""
    import torch.nn as nn

    class Head(nn.Module):
        def __init__(self):
            super().__init__()
            self.gates = nn.Parameter(torch.zeros(N_DOM, E_N))
            self.lin = nn.Linear(E_N + HCE_D + E_N, 1)
            self.register_buffer("eval_mean", torch.zeros(E_N))
            self.register_buffer("eval_std", torch.ones(E_N))

        def forward(self, ev_raw, hce, dom):
            norm = (ev_raw - self.eval_mean) / self.eval_std
            gated = norm * self.gates[dom]
            oh = torch.zeros(ev_raw.shape[0], E_N, device=ev_raw.device)
            oh.scatter_(1, dom[:, None], 1.0)
            return self.lin(torch.cat([gated, hce, oh], dim=1)).squeeze(-1)

    head = Head()
    if prev_pt and os.path.exists(prev_pt):
        head.load_state_dict(torch.load(prev_pt, map_location="cpu", weights_only=False))
        src = f"prev ({os.path.basename(prev_pt)})"
    elif bootstrap:
        g, lw, lb, mean, std = parse_evh(bootstrap)
        with torch.no_grad():
            head.gates.copy_(torch.from_numpy(g))
            head.lin.weight.copy_(torch.from_numpy(lw)[None, :])
            head.lin.bias.fill_(lb)
            head.eval_mean.copy_(torch.from_numpy(mean))
            head.eval_std.copy_(torch.from_numpy(std))
        src = f"bootstrap ({os.path.basename(bootstrap)})"
    else:
        raise RuntimeError("need bootstrap or prev head")
    print(f"  head init from {src}", flush=True)
    return head


def rl_train_step(evals, hces, doms, zs, prev_pt, epochs):
    head = build_head(bootstrap=BOOT_EVH, prev_pt=prev_pt)

    X_ev = torch.tensor(evals, dtype=torch.float32)
    X_hce = torch.tensor(hces, dtype=torch.float32)
    X_dom = torch.tensor(doms, dtype=torch.long)
    Y = torch.tensor(np.array(zs, dtype=np.float32))

    old = {k: v.clone() for k, v in head.state_dict().items()
           if k in ("gates", "lin.weight", "lin.bias")}

    opt = torch.optim.AdamW([
        {"params": [head.gates], "lr": 5e-5},
        {"params": head.lin.parameters(), "lr": 2e-4},
    ], weight_decay=1e-5)

    def wp(cp):
        return 1.0 / (1.0 + torch.exp(-cp / 361.0))

    # 10% validation split — convergence evidence, and best-val checkpoint wins
    N = X_ev.shape[0]
    g = torch.Generator().manual_seed(42)
    perm_all = torch.randperm(N, generator=g)
    n_val = max(500, N // 10)
    val_idx, tr_idx = perm_all[:n_val], perm_all[n_val:]

    best_val, best_state, stale = 1e9, None, 0
    BS = 4096
    for ep in range(epochs):
        perm = tr_idx[torch.randperm(len(tr_idx))]
        tl, nb = 0.0, 0
        for i in range(0, len(perm), BS):
            idx = perm[i:i + BS]
            pred = head(X_ev[idx], X_hce[idx], X_dom[idx])
            loss = ((wp(pred) - Y[idx]) ** 2).mean()
            opt.zero_grad(); loss.backward(); opt.step()
            tl += loss.item(); nb += 1
        head.eval()
        with torch.no_grad():
            vp = head(X_ev[val_idx], X_hce[val_idx], X_dom[val_idx])
            vl = ((wp(vp) - Y[val_idx]) ** 2).mean().item()
        head.train()
        if vl < best_val - 1e-6:
            best_val, stale = vl, 0
            best_state = {k: v.detach().clone() for k, v in head.state_dict().items()}
        else:
            stale += 1
        if ep == 0 or ep == epochs - 1 or ep % 25 == 0:
            print(f"  rl_epoch {ep}: train={tl / max(nb, 1):.6f} val={vl:.6f}", flush=True)
        if stale >= 50:  # converged, no val improvement for 50 epochs
            print(f"  early stop at epoch {ep} (val plateau {best_val:.6f})", flush=True)
            break
    if best_state is not None:
        head.load_state_dict(best_state)

    with torch.no_grad():
        tr_mask = torch.zeros(N, dtype=torch.bool); tr_mask[tr_idx] = True
        corr = np.corrcoef(wp(head(X_ev[tr_mask], X_hce[tr_mask], X_dom[tr_mask])).numpy(),
                           Y[tr_mask].numpy())[0, 1]
    change = math.sqrt(sum(((old[k].float() - head.state_dict()[k].float()) ** 2).sum().item()
                           for k in old))
    print(f"  win-prob corr: {corr:.4f}", flush=True)
    return head, change


def main():
    smoke = "--smoke" in sys.argv
    from_dir = None
    if "--from-dir" in sys.argv:
        from_dir = sys.argv[sys.argv.index("--from-dir") + 1]
    tc = int(sys.argv[1]) if len(sys.argv) > 1 and sys.argv[1].isdigit() else 1
    n_rounds = 2 if smoke else (int(sys.argv[2]) if len(sys.argv) > 2 and sys.argv[2].isdigit() else 960)
    epochs = 5 if smoke else 300

    os.makedirs(SP, exist_ok=True)
    state_f = f"{SP}/state.json"
    state = {"tc": tc, "iteration": 0, "streak": 0, "history": []}
    if os.path.exists(state_f):
        state = json.load(open(state_f))
    state["tc"] = tc
    it = state["iteration"]

    print(f"=== RL ITERATION {it}: TC={tc}s, {n_rounds * 2} games ===", flush=True)

    # 1. self-play with the current best stacked head
    head_evh = CUR_EVH if os.path.exists(CUR_EVH) else BOOT_EVH
    if from_dir:
        gdir = from_dir
        print(f"gen: SKIPPED (--from-dir {gdir})", flush=True)
    else:
        t0 = time.time()
        gdir = gen_games(tc, n_rounds, head_evh)
        print(f"  games generated in {time.time() - t0:.0f}s", flush=True)

    # 2. parse
    fens, zs = parse_games(gdir)
    if len(fens) < 100:
        print("  ERROR: too few positions, skipping", flush=True)
        return

    # 3. expert evals (GPU)  4. HCE + domains (CPU)
    print("  extracting expert evals...", flush=True)
    t0 = time.time()
    evals = extract_expert_evals(fens)
    print(f"  evals done in {time.time() - t0:.0f}s", flush=True)
    hces, doms = compute_hce_batch(fens)
    print(f"  domains: {np.bincount(doms, minlength=13).tolist()}", flush=True)

    # 5. train  6. export
    prev_pt = f"{SP}/head_rl_iter{it}.pt"
    print("  training head on outcomes...", flush=True)
    head, change = rl_train_step(evals, hces, doms, zs,
                                 prev_pt if it > 0 else None, epochs)
    print(f"  weight change L2: {change:.6f}", flush=True)

    torch.save(head.state_dict(), f"{SP}/head_rl_iter{it + 1}.pt")
    g = head.gates.detach().numpy(); lw = head.lin.weight.detach().numpy().flatten()
    lb = float(head.lin.bias.item())
    write_evh(f"{SP}/head_rl_iter{it + 1}.evh", g, lw, lb,
              head.eval_mean.numpy(), head.eval_std.numpy())
    shutil.copy(f"{SP}/head_rl_iter{it + 1}.evh", CUR_EVH)

    # 7. state
    state["iteration"] += 1
    state["streak"] = state["streak"] + 1 if change < 1e-4 else 0
    state["history"].append({"iter": it, "tc": tc, "pos": len(fens), "change": round(change, 6)})
    json.dump(state, open(state_f, "w"), indent=1)

    print(f"=== ITERATION {state['iteration']} COMPLETE "
          f"(streak={state['streak']}, change={change:.6f}) ===", flush=True)


if __name__ == "__main__":
    main()
