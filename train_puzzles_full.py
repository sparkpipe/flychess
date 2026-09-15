#!/usr/bin/env python3
"""Train FlyV3 on the ENTIRE lichess puzzle DB (6.1M puzzles), FULL LINES:
CE on EVERY solver move all the way to the end of each puzzle line
(teacher forcing: walk the recorded line, opponent replies included).

PLUS a 3-class position pheromone readout (for the side to move):
  2 = WON   (mateIn*/crushing puzzles, solver plies; KQ/KR-vs-K anchors)
  1 = PLAYABLE (equal-ish, 40-60% — NOT drawn)
  0 = LOST  (opponent plies of winning puzzles; checkmate-in-position anchors)
Drawn/unwinnable positions get their own anchor set (KvK/KNvK/KBvK, stalemate)
and map to class 1's "draw-in-hand" extreme via synthetic dead-draw data.

Pipeline: fork workers extract features per (puzzle, solver-ply) instance on
CPU; the GPU consumer trains batched propagation (B positions per step).

Env: B=256, WORKERS=12, SMOKE=1, EPOCHS=1000, LR=3e-4, RESUME=1,
EVAL_EVERY=500000, W_CLS=0.5.
"""
import numpy as np, torch, scipy.sparse as sp, chess, time, json, sys, os, zlib, random
import multiprocessing as mp
from collections import deque

CSV   = "/home/spec/chess-lab/puzzles.csv"
STATE = "/home/spec/chess-lab/flyv3_state.pt"
LOGF  = "/home/spec/chess-lab/v3_puzzle_log.jsonl"
DEV   = "cuda"
PROP_STEPS, LEAK, CAP = 2, 0.5, 20.0
MAXL  = 256                       # pad legal-move axis
WIN_THEMES = {"mateIn1", "mateIn2", "mateIn3", "mateIn4", "mateIn5", "crushing"}

sys.path.insert(0, "/home/spec/chess-lab")
import fly_v3_full as v3
import flyfeat                   # vectorized extractor, bit-identical features

# ---------------- model: v3 + 3-neuron class pheromone head ----------------
class FlyV3C(v3.FlyV3):
    def __init__(self, feature_keys, n_sens):
        super().__init__(feature_keys, n_sens)
        # identical rng -> identical readout permutation; take 3 more neurons
        z = np.load(v3.BRAIN)
        m = sp.csr_matrix((z["data"], z["indices"], z["indptr"]),
                          shape=tuple(z["shape"]))
        sens = np.load(v3.SENS)
        nprng = np.random.default_rng(20260912)
        free = np.setdiff1d(np.arange(m.shape[0]), sens)
        p2 = nprng.permutation(len(free))
        self.cls_idx = torch.from_numpy(free[p2[4096:4099]].astype(np.int64)).to(DEV)
        self.theta_cls = torch.nn.Parameter(torch.zeros(3))

    def class_logits(self, a):
        """a: (B,N) or (N,) activations -> (3,) or (B,3) pheromone logits."""
        if a.dim() == 1:
            return self.theta_cls * a[self.cls_idx]
        return self.theta_cls.unsqueeze(0) * a[:, self.cls_idx]

# ---------------- worker side (CPU extraction) ----------------
def init_worker():
    v3.feat_vec(chess.Board())    # seeds v3.FEATURE_KEYS deterministically
    flyfeat.feat_vec(chess.Board())

def instance(job):
    """Walk one puzzle line; return one record per SOLVER/OPP ply.
    lichess format: FEN is before the opponent's move moves[0].
    Record: (fv, slots, tgt, mf, cls)  cls for side-to-move at that ply.
    Solver ply of a winning puzzle = WON(2); opponent ply = LOST(0);
    everything else = PLAYABLE(1). tgt=-1 rows train class only."""
    fen, moves, themes = job
    out = []
    win = any(t in WIN_THEMES for t in (themes or "").split())
    try:
        b = chess.Board(fen)
        ms = moves.split()
        b.push(chess.Move.from_uci(ms[0]))
        for i in range(1, len(ms)):
            mv = chess.Move.from_uci(ms[i])
            if mv not in b.legal_moves:
                return out
            fv = flyfeat.feat_vec(b)
            mvs = list(b.legal_moves)
            slots = np.array([m.from_square * 64 + m.to_square for m in mvs],
                             dtype=np.int64)
            mf = np.stack([flyfeat.move_feats(b, m) for m in mvs]).astype(np.float32)
            if i % 2 == 1:
                out.append((fv, slots, mvs.index(mv), mf, 2 if win else 1))
            else:
                out.append((fv, slots, -1, mf, 0 if win else 1))
            b.push(mv)
    except Exception:
        return out
    return out

def synth_pool(n=4000):
    """Dead-position anchors for the pheromone scale. KQ/KR-vs-K = WON,
    KvK/KNvK/KBvK = DRAWN-in-hand. Terminal boards (no legal moves) are
    skipped — LOST pheromone is trained by opponent plies of winning puzzles."""
    rng = random.Random(99)
    out = []
    def one(pieces):
        b = chess.Board(None)
        sqs = rng.sample(range(64), len(pieces))
        for sq, (pt, col) in zip(sqs, pieces):
            b.set_piece_at(sq, chess.Piece(pt, col))
        if not b.is_valid(): return None
        b.turn = rng.random() < 0.5
        return b
    specs = []
    for _ in range(n // 5):
        specs.append(([(chess.KING, True), (chess.KING, False),
                       (chess.QUEEN, True)], 2))
        specs.append(([(chess.KING, True), (chess.KING, False),
                       (chess.ROOK, True)], 2))
        specs.append(([(chess.KING, True), (chess.KING, False)], 1))
        specs.append(([(chess.KING, True), (chess.KING, False),
                       (chess.KNIGHT, True)], 1))
        specs.append(([(chess.KING, True), (chess.KING, False),
                       (chess.BISHOP, False)], 1))
    for pieces, cls in specs:
        b = one(pieces)
        if b is None: continue
        try:
            mvs = list(b.legal_moves)
            if not mvs: continue           # terminal boards: skip
            fv = flyfeat.feat_vec(b)
            slots = np.array([m.from_square * 64 + m.to_square for m in mvs],
                             dtype=np.int64)
            mf = np.stack([flyfeat.move_feats(b, m) for m in mvs]).astype(np.float32)
            out.append((fv, slots, -1, mf, cls))
        except Exception:
            continue
    return out

def mating_anchors(n=600):
    """KQvK / KRvK technique positions WITH target moves (the fly could draw
    but never convert). Target = mate-in-1 if present, else the move that
    minimizes the defender king's freedom without stalemating."""
    rng = random.Random(77)
    out = []
    tries = 0
    while len(out) < n and tries < n * 12:
        tries += 1
        b = chess.Board(None)
        sqs = rng.sample(range(64), 3)
        b.set_piece_at(sqs[0], chess.Piece(chess.KING, chess.WHITE))
        b.set_piece_at(sqs[1], chess.Piece(chess.KING, chess.BLACK))
        pt = chess.QUEEN if rng.random() < 0.5 else chess.ROOK
        b.set_piece_at(sqs[2], chess.Piece(pt, chess.WHITE))
        if not b.is_valid():
            continue
        b.turn = chess.WHITE
        mvs = list(b.legal_moves)
        if not mvs:
            continue
        best, best_mv = -1e9, None
        for mv in mvs:
            b.push(mv)
            oc = b.outcome(claim_draw=False)
            if oc is not None and oc.winner is not None:
                sc = 10000.0                      # mate in 1
            elif b.is_stalemate():
                sc = -10000.0
            else:
                bk = b.king(chess.BLACK)
                box = max(chess.square_file(bk), 7 - chess.square_file(bk)) + \
                    max(chess.square_rank(bk), 7 - chess.square_rank(bk))
                sc = 10 * (14 - box) - len(list(b.legal_moves)) * 0.1 \
                    + (1.0 if b.is_check() else 0.0)
                if b.is_repetition(2):
                    sc -= 50.0
            b.pop()
            if sc > best:
                best, best_mv = sc, mv
        if best_mv is None or best < -50:
            continue
        try:
            fv = flyfeat.feat_vec(b)
            slots = np.array([m.from_square * 64 + m.to_square for m in mvs],
                             dtype=np.int64)
            mf = np.stack([flyfeat.move_feats(b, m) for m in mvs]).astype(np.float32)
            out.append((fv, slots, mvs.index(best_mv), mf, 2))
        except Exception:
            continue
    return out

# ---------------- parent side (GPU training) ----------------
def batch_forward(model, fvb, slotb, mfb, maskb):
    # column layout a=(N,B): WT @ a hits the fast contiguous cuSPARSE path
    x = torch.from_numpy(fvb).to(DEV, non_blocking=True)
    s = torch.clamp(model.W_sens(x), -6, 6)
    B = x.shape[0]
    a = torch.zeros(model.N, B, device=DEV)
    a[model.sensory_idx] = s.T
    for _ in range(PROP_STEPS):
        a = torch.clamp((1 - LEAK) * a + LEAK * (model.WT @ a), -CAP, CAP)
    cols = torch.arange(B, device=DEV).unsqueeze(1)
    slots = torch.from_numpy(slotb).to(DEV)
    rid = model.readout_idx[slots]
    T = model.theta[slots] * a[rid, cols] \
        + torch.from_numpy(mfb).to(DEV) @ model.theta_mv
    T = T.masked_fill(~torch.from_numpy(maskb).to(DEV), -1e9)
    cls = model.theta_cls.unsqueeze(0) * a[model.cls_idx].T
    return torch.log_softmax(T, dim=1), cls, a.detach().T

def collate(buf):
    B = len(buf)
    F = buf[0][0].shape[0]
    fvb = np.zeros((B, F), dtype=np.float32)
    slotb = np.zeros((B, MAXL), dtype=np.int64)
    mfb = np.zeros((B, MAXL, 8), dtype=np.float32)
    maskb = np.zeros((B, MAXL), dtype=bool)
    tgtb = np.full(B, -1, dtype=np.int64)
    clb = np.zeros(B, dtype=np.int64)
    for i, (fv, slots, tgt, mf, cl) in enumerate(buf):
        K = len(slots)
        fvb[i] = fv
        slotb[i, :K] = slots
        mfb[i, :K] = mf
        maskb[i, :K] = True
        tgtb[i] = tgt
        clb[i] = cl
    return fvb, slotb, mfb, maskb, tgtb, clb

def eval_batch(model, jobs):
    buf = []
    for job in jobs:
        buf.extend(instance(job))
    if not buf: return None
    tot = cor = ctot = ccor = 0
    with torch.no_grad():
        for i in range(0, len(buf), 256):
            fvb, slotb, mfb, maskb, tgtb, clb = collate(buf[i:i+256])
            logp, cls_logits, _ = batch_forward(model, fvb, slotb, mfb, maskb)
            pred = torch.argmax(logp, dim=1).cpu().numpy()
            cpred = torch.argmax(cls_logits, dim=1).cpu().numpy()
            mv = tgtb >= 0
            cor += int((pred[mv] == tgtb[mv]).sum()); tot += int(mv.sum())
            ccor += int((cpred == clb).sum()); ctot += len(clb)
    return cor / max(tot, 1), ccor / max(ctot, 1)

def instance_chunk(jobs):
    """Process a chunk of puzzle jobs into one flat record list (bounded IPC)."""
    out = []
    for job in jobs:
        out.extend(instance(job))
    return out

def ho_eval(model, ho, syn, full_n=300, first_n=1200, m2_n=800):
    """held-out: first-move acc + class acc, mateIn2 subset, full-line acc."""
    model.eval()
    r1 = eval_batch(model, [(f, m, t) for f, m, t in ho[:first_n]])
    jobs2 = [(f, m, t) for f, m, t in ho if "mateIn2" in t][:m2_n]
    r2 = eval_batch(model, jobs2) if jobs2 else (0.0, 0.0)
    # synthetic pheromone check (the absolute W/D/L anchors)
    sbuf = [(fv, slots, tgt, mf, cl) for (fv, slots, tgt, mf, cl) in syn[:600]]
    ctot = ccor = 0
    with torch.no_grad():
        for i in range(0, len(sbuf), 256):
            fvb, slotb, mfb, maskb, tgtb, clb = collate(sbuf[i:i+256])
            _, cls_logits, _ = batch_forward(model, fvb, slotb, mfb, maskb)
            cpred = torch.argmax(cls_logits, dim=1).cpu().numpy()
            ccor += int((cpred == clb).sum()); ctot += len(clb)
    resfull = []
    for fen, moves, _ in ho[:full_n]:
        ok = True
        try:
            b = chess.Board(fen); ms = moves.split(); b.push(chess.Move.from_uci(ms[0]))
            for i in range(1, len(ms)):
                mv = chess.Move.from_uci(ms[i])
                if mv not in b.legal_moves: ok = False; break
                if i % 2 == 1:
                    mvs, T, _ = model.scores(b)
                    if int(torch.argmax(T)) != mvs.index(mv): ok = False; break
                b.push(mv)
        except Exception:
            ok = False
        resfull.append(ok)
    model.train()
    return (r1[0], r1[1], r2[0], ccor / max(ctot, 1), float(np.mean(resfull)))

def main():
    import pandas as pd
    B = int(os.environ.get("B", "256"))
    WORKERS = int(os.environ.get("WORKERS", "20"))
    EPOCHS = int(os.environ.get("EPOCHS", "1000"))
    LR = float(os.environ.get("LR", "3e-4"))
    W_CLS = float(os.environ.get("W_CLS", "1.5"))
    smoke = os.environ.get("SMOKE", "0") == "1"

    print("loading csv ...", flush=True)
    df = pd.read_csv(CSV,
                     usecols=["PuzzleId", "FEN", "Moves", "Rating", "Themes"],
                     dtype={"PuzzleId": str, "FEN": str, "Moves": str,
                            "Rating": int, "Themes": str})
    if smoke:
        df = df.iloc[:30000]
    ids = df.PuzzleId.values
    ho_mask = np.array([zlib.crc32(p.encode()) % 100 < 2 for p in ids])
    order = np.argsort(df.Rating.values[~ho_mask], kind="stable")
    arr = df[~ho_mask].reset_index(drop=True)
    rng = np.random.default_rng(7)
    for blk in range(0, len(order), 4096):
        seg = order[blk:blk + 4096].copy(); rng.shuffle(seg)
        order[blk:blk + 4096] = seg
    fens = arr.FEN.values; movs = arr.Moves.values; thms = arr.Themes.fillna("").values
    print(f"train puzzles={len(order)}  holdout={int(ho_mask.sum())}", flush=True)

    ho_df = df[ho_mask]
    ho = list(zip(ho_df.FEN.values, ho_df.Moves.values,
                  ho_df.Themes.fillna("").values))
    random.Random(5).shuffle(ho)
    del df, ho_df, ids, ho_mask          # free the 6.1M-row frame

    init_worker()
    print("building synthetic anchors ...", flush=True)
    syn = synth_pool(4000) + mating_anchors(800)
    print(f"synthetic anchors: {len(syn)}", flush=True)
    ctx = mp.get_context("fork")      # COW-shared pages; workers never touch CUDA
    pool = ctx.Pool(WORKERS, initializer=init_worker, maxtasksperchild=512)

    model = FlyV3C(sorted(v3.FEATURE_KEYS), 26933).to(DEV)
    if os.environ.get("RESUME", "0") == "1" and os.path.exists(STATE):
        model.load_state_dict(torch.load(STATE, weights_only=True), strict=False)
        print("resumed from flyv3_state.pt", flush=True)
    opt = torch.optim.Adam(model.parameters(), lr=LR)

    if not os.path.exists("/home/spec/chess-lab/flyv3_state_pre_puzzleDB.pt"):
        import shutil
        if os.path.exists(STATE):
            shutil.copy(STATE, "/home/spec/chess-lab/flyv3_state_pre_puzzleDB.pt")
            print("backed up pre-puzzleDB state", flush=True)

    t0 = time.time(); seen = 0; puzzles = 0; loss_acc = 0.0; nloss = 0; nstep = 0
    buf = []; last_ck = time.time(); last_eval_seen = 0; last_ev = {}
    t_wait = t_col = t_gpu = 0.0          # pipeline phase timers
    logf = open(LOGF, "a")
    syn_rng = random.Random(3)

    def checkpoint():
        tmp = STATE + ".tmp"
        torch.save(model.state_dict(), tmp)
        os.replace(tmp, STATE)

    def do_step(chunk):
        nonlocal seen, loss_acc, nloss, last_ck, last_eval_seen, t_col, t_gpu, nstep, last_ev
        # mix in ~5% synthetic pheromone anchors
        for _ in range(max(1, B // 10)):
            chunk[syn_rng.randrange(len(chunk))] = syn[syn_rng.randrange(len(syn))]
        _t = time.time()
        fvb, slotb, mfb, maskb, tgtb, clb = collate(chunk)
        t_col += time.time() - _t
        _t = time.time()
        logp, cls_logits, _ = batch_forward(model, fvb, slotb, mfb, maskb)
        tgt = torch.from_numpy(tgtb).to(DEV)
        clst = torch.from_numpy(clb).to(DEV)
        row_loss = -logp.gather(1, tgt.clamp(min=0)[:, None]).squeeze(1)
        mv = tgt >= 0
        mv_term = row_loss[mv].mean() if int(mv.sum()) else torch.zeros((), device=DEV)
        loss = mv_term \
            + torch.nn.functional.cross_entropy(cls_logits, clst) * W_CLS
        opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        torch.cuda.synchronize()
        t_gpu += time.time() - _t
        seen += len(chunk); loss_acc += float(loss.item()); nloss += 1; nstep += 1
        do_eval = nstep == 1 or seen - last_eval_seen >= int(os.environ.get("EVAL_EVERY", "500000"))
        do_print = nstep == 1 or nstep % 200 == 0
        if do_eval:
            r1m, r1c, r2m, rs_c, rf = ho_eval(model, ho, syn)
            ev = {"ho_first": round(r1m, 4), "ho_cls": round(r1c, 4),
                  "ho_mateIn2": round(r2m, 4), "syn_cls": round(rs_c, 4),
                  "ho_full": round(rf, 4)}
            last_eval_seen = seen
            last_ev = ev
        else:
            ev = {}
        if do_print:
            el = time.time() - t0
            rss_mb = int(open("/proc/self/statm").read().split()[1]) * 4096 >> 20
            rec = {"ep": ep, "seen": seen, "puzzles": puzzles,
                   "loss": round(loss_acc / max(nloss, 1), 4),
                   "ips": round(seen / el, 1),
                   "gpu_ms": round(t_gpu * 1000 / max(1, nstep)),
                   "col_ms": round(t_col * 1000 / max(1, nstep)),
                   "rss_gb": round(rss_mb / 1024, 1)}
            rec.update(last_ev if not ev else ev)
            print(json.dumps(rec), flush=True)
            logf.write(json.dumps(rec) + "\n"); logf.flush()
            loss_acc = 0.0; nloss = 0
        if time.time() - last_ck > 1200:
            checkpoint(); last_ck = time.time()
            print("checkpoint saved", flush=True)

    for ep in range(EPOCHS):
        if ep > 0:                              # reshuffle between epochs
            rng.shuffle(order)
        # bounded-submission pipeline: at most 2*WORKERS chunk results alive
        jobs_iter = iter(order)
        pending = deque()
        MAXIN = 2 * WORKERS + 2
        CH = 64

        def submit():
            while len(pending) < MAXIN:
                i0 = next(jobs_iter, None)
                if i0 is None:
                    return
                chunk_jobs = []
                for _ in range(CH):
                    j = next(jobs_iter, None)
                    if j is None:
                        break
                    chunk_jobs.append((fens[j], movs[j], thms[j]))
                if chunk_jobs:
                    pending.append(pool.apply_async(instance_chunk,
                                                    (chunk_jobs,)))
        submit()
        while pending:
            _t = time.time()
            res = pending.popleft().get()
            t_wait += time.time() - _t
            buf.extend(res)
            puzzles += CH
            submit()
            while len(buf) >= B:
                chunk, buf = buf[:B], buf[B:]
                do_step(chunk)
                res = None
        while len(buf) >= B:                     # epoch-end flush of the remainder
            chunk, buf = buf[:B], buf[B:]
            do_step(chunk)
        checkpoint()
        print(f"EPOCH {ep} DONE puzzles={puzzles} seen={seen} "
              f"mins={round((time.time()-t0)/60,1)}", flush=True)
    checkpoint()
    pool.close()

if __name__ == "__main__":
    main()
