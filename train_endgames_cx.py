#!/usr/bin/env python3
"""CX endgame lane: the endgame trains through a DIFFERENT part of the fly
brain — features inject into the 963 central-complex neurons (the fly's
stepwise goal-vector machinery) instead of the 26,933 sensory periphery.

Features: flyfeat_end (Dvoretsky-taxonomy concepts: opposition, key squares,
rule of the square, zugzwang proxies, rook cut-off/behind-passer/7th rank,
wrong/bad bishop, opposite-bishop drawishness, exchange-down-for-pawn).
Data: full-line lichess endgame puzzles + synthetic exact endings whose
WON/DRAWN labels come from endgame THEORY (the class pheromone learns
'exchange down for a pawn = drawn' here).
Output: flyv3cx_state.pt (separate lane; main lane untouched).
Env: B=256, WORKERS=10, SMOKE=1, RESUME=1, W_CLS=1.5, EVAL_EVERY=300000.
"""
import numpy as np, torch, chess, time, json, os, sys, zlib, random
import multiprocessing as mp
from collections import deque

sys_p = "/home/spec/chess-lab"
sys.path.insert(0, sys_p)
import fly_v3_full as v3
import flyfeat, flyfeat_end
from train_puzzles_full import FlyV3C, collate as base_collate, MAXL

CSV   = f"{sys_p}/puzzles.csv"
STATE = f"{sys_p}/flyv3cx_state.pt"
LOGF  = f"{sys_p}/v3cx_log.jsonl"
DEV   = "cuda"
PROP_STEPS, LEAK, CAP = 2, 0.5, 20.0
END_THEMES = ("endgame", "rookEndgame", "pawnEndgame", "bishopEndgame",
              "knightEndgame", "queenEndgame")
WIN_THEMES = {"mateIn1", "mateIn2", "mateIn3", "mateIn4", "mateIn5", "crushing"}

CX_IDX = np.load(f"{sys_p}/cx_idx.npy")


class FlyV3CX(FlyV3C):
    """Endgame senses: features -> 963 CX neurons -> same graph/readout.
    Readout neurons (move slots + class) are SELECTED by activation variance
    across diverse boards at build time — random readouts receive ~zero
    signal because 963 injected neurons dilute ~1000x in two steps."""
    def __init__(self, n_end_feats, sel_boards=None):
        v3.feat_vec(chess.Board())          # seed FEATURE_KEYS for parent class
        super().__init__(sorted(v3.FEATURE_KEYS), 26933)
        self.cx_idx = torch.from_numpy(CX_IDX.astype(np.int64)).to(DEV)
        self.W_cx = torch.nn.Linear(n_end_feats, len(CX_IDX))
        self.cx_gain = 80.0
        torch.nn.init.zeros_(self.theta)
        torch.nn.init.zeros_(self.theta_mv)
        torch.nn.init.zeros_(self.theta_cls)
        self.cls_cx = torch.from_numpy(CX_IDX[[100, 400, 700]].astype(np.int64)).to(DEV)
        if sel_boards:
            self._select_readouts(sel_boards)

    @torch.no_grad()
    def _prop_all(self, boards):
        import flyfeat_end
        fvb = np.stack([flyfeat_end.feat_vec_end(b)[0] for b in boards])
        x = torch.from_numpy(fvb).to(DEV)
        s = torch.clamp(self.W_cx(x), -6, 6) * self.cx_gain
        a = torch.zeros(self.N, x.shape[0], device=DEV)
        a[self.cx_idx] = s.T
        for _ in range(PROP_STEPS):
            a = torch.clamp((1 - LEAK) * a + LEAK * (self.WT @ a), -CAP, CAP)
        return a

    def _select_readouts(self, boards, n_move=4096, n_cls=3):
        self.to(DEV)                                  # selection runs on GPU
        a = self._prop_all(boards)                    # (N, B)
        var = a.var(dim=1).cpu().numpy()
        var[self.cx_idx.cpu().numpy()] = -1.0         # injected neurons saturate
        order = np.argsort(var)[::-1]
        self.readout_idx = torch.from_numpy(
            order[:n_move].astype(np.int64)).to(DEV)
        self.cls_cx = torch.from_numpy(
            order[n_move:n_move + n_cls].astype(np.int64)).to(DEV)
        print(f"readout selected: var top={var[order[0]]:.3f} "
              f"4096th={var[order[n_move-1]]:.4f} "
              f"cls={var[order[n_move:n_move+n_cls]].round(4)}", flush=True)

    def batch_forward(self, fvb, slotb, mfb, maskb):
        x = torch.from_numpy(fvb).to(DEV, non_blocking=True)
        s = torch.clamp(self.W_cx(x), -6, 6) * self.cx_gain
        B = x.shape[0]
        a = torch.zeros(self.N, B, device=DEV)
        a[self.cx_idx] = s.T
        for _ in range(PROP_STEPS):
            a = torch.clamp((1 - LEAK) * a + LEAK * (self.WT @ a), -CAP, CAP)
        cols = torch.arange(B, device=DEV).unsqueeze(1)
        slots = torch.from_numpy(slotb).to(DEV)
        T = self.theta[slots] * a[self.readout_idx[slots], cols] \
            + torch.from_numpy(mfb).to(DEV) @ self.theta_mv
        T = T.masked_fill(~torch.from_numpy(maskb).to(DEV), -1e9)
        cls = self.theta_cls.unsqueeze(0) * torch.tanh(a[self.cls_cx].T / 4.0)
        return torch.log_softmax(T, dim=1), cls


# ---------------- exact endings (Dvoretsky "precise positions") ----------------
def exact_endings():
    """Hand-built theoretical positions with theory-correct class labels.
    Class for side to move: 2=WON, 1=DRAWN-in-hand, 0=LOST. Move CE masked."""
    E = []
    def add(fen, cls):
        b = chess.Board(fen)
        if not b.is_valid():
            return
        if b.is_game_over():
            oc = b.outcome()
            if not (oc and oc.termination == chess.Termination.INSUFFICIENT_MATERIAL):
                return                      # keep dead draws as anchors
        E.append((fen, cls))
        E.append((b.mirror().fen(), cls))   # color-mirrored twin, same STM class
    # trivial draws
    add("8/8/8/4k3/8/8/8/K7 w - - 0 1", 1)          # KvK
    add("8/8/8/4k3/8/8/8/K5N1 w - - 0 1", 1)        # KNvK
    add("8/8/8/4k3/8/8/8/K5B1 w - - 0 1", 1)        # KBvK
    # trivial wins
    add("8/8/8/4k3/8/8/8/K2Q4 w - - 0 1", 2)        # KQvK
    add("8/8/8/4k3/8/8/8/K2R4 w - - 0 1", 2)        # KRvK
    add("8/8/8/4k3/8/8/8/K2Q2R1 w - - 0 1", 2)
    # KP: defender outside the square -> unstoppable runner (rule of the square)
    add("7k/8/8/8/8/8/P7/K7 w - - 0 1", 2)          # pawn a2, black king h8 far
    add("8/8/8/8/8/2k5/P7/K7 w - - 0 1", 1)         # defender inside the square
    # Lucena-style: pawn on 7th, own king in front, enemy king cut off -> WON
    add("1K6/1P6/8/8/7R/8/6k1/7r w - - 0 1", 2)
    add("2K5/2P5/8/8/6R1/8/5k2/6r1 w - - 0 1", 2)
    # Philidor 3rd-rank defense: defender king in FRONT of pawn, rook on the
    # attack rank until the pawn advances -> DRAWN
    add("k7/8/K7/1P6/8/6r1/8/6R1 w - - 0 1", 1)       # Kb8 in front of b5?K a6+Pb5
    add("k7/8/K7/P7/8/6r1/8/6R1 w - - 0 1", 1)        # same with a5 pawn
    # short-side defense with pawn on 6th -> DRAWN
    add("k7/8/K7/P7/8/6r1/8/6R1 b - - 0 1", 1)
    # R vs B corner fortress: bishop controls the corner, king sheltered -> DRAWN
    add("rk6/pb6/1K6/8/8/8/6R1/8 w - - 0 1", 1)
    add("rk6/1b6/1K6/8/8/8/6R1/8 w - - 0 1", 1)
    # R vs R+wrong... exchange-down-but-drawn archetype: R vs B+N-ish material
    # (R each side; one side has B+N vs two pawns -> usually drawn without passers)
    add("1k6/8/8/8/8/2nb4/8/1K1R4 w - - 0 1", 1)
    # lost archetypes: bare rook vs pawn on 7th with defending king cut away
    add("k7/P7/1K6/8/8/8/8/7r w - - 0 1", 2)        # pawn a7, king supports -> WON
    add("8/8/8/8/8/1k6/2r5/K7 w - - 0 1", 0)        # K vs R+K: STM lost
    add("8/8/8/8/8/1k6/2q5/K7 w - - 0 1", 0)        # K vs Q+K: STM lost
    return E


def synth_instances(exact):
    out = []
    for fen, cls in exact:
        b = chess.Board(fen)
        mvs = list(b.legal_moves)
        if not mvs:
            continue
        fv, _ = flyfeat_end.feat_vec_end(b)
        slots = np.array([m.from_square * 64 + m.to_square for m in mvs], np.int64)
        mf = np.stack([flyfeat.move_feats(b, m) for m in mvs]).astype(np.float32)
        out.append((fv, slots, -1, mf, cls))       # class-only instance
    return out


# ---------------- worker: full-line endgame puzzle instances ----------------
def init_worker():
    flyfeat_end.feat_vec_end(chess.Board())
    flyfeat.feat_vec(chess.Board())


def instance(job):
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
            fv, _ = flyfeat_end.feat_vec_end(b)
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


def instance_chunk(jobs):
    out = []
    for job in jobs:
        out.extend(instance(job))
    return out


def main():
    import pandas as pd
    B = int(os.environ.get("B", "256"))
    WORKERS = int(os.environ.get("WORKERS", "10"))
    EPOCHS = int(os.environ.get("EPOCHS", "1000"))
    W_CLS = float(os.environ.get("W_CLS", "1.5"))
    smoke = os.environ.get("SMOKE", "0") == "1"

    print("loading csv ...", flush=True)
    df = pd.read_csv(CSV, usecols=["PuzzleId", "FEN", "Moves", "Rating", "Themes"])
    m = df.Themes.fillna("")
    mask = np.zeros(len(df), dtype=bool)
    for t in END_THEMES:
        mask |= m.str.contains(t).values
    df = df[mask].reset_index(drop=True)
    if smoke:
        df = df.iloc[:20000]
    ids = df.PuzzleId.values
    ho_mask = np.array([zlib.crc32(p.encode()) % 100 < 2 for p in ids])
    order = np.argsort(df.Rating.values[~ho_mask], kind="stable")
    arr = df[~ho_mask].reset_index(drop=True)
    rng = np.random.default_rng(11)
    for blk in range(0, len(order), 4096):
        seg = order[blk:blk + 4096].copy(); rng.shuffle(seg)
        order[blk:blk + 4096] = seg
    fens = arr.FEN.values; movs = arr.Moves.values; thms = arr.Themes.fillna("").values
    print(f"endgame puzzles: train={len(order)} ho={int(ho_mask.sum())}", flush=True)
    ho_df = df[ho_mask]
    ho = list(zip(ho_df.FEN.values, ho_df.Moves.values, ho_df.Themes.fillna("").values))
    random.Random(6).shuffle(ho)
    del df, ho_df, ids, ho_mask, arr

    init_worker()
    exact = exact_endings()
    syn = synth_instances(exact)
    print(f"exact endings: {len(syn)} instances from {len(exact)} positions", flush=True)
    ctx = mp.get_context("fork")
    pool = ctx.Pool(WORKERS, initializer=init_worker, maxtasksperchild=512)

    v0, NAMES = flyfeat_end.feat_vec_end(chess.Board())
    # diverse boards for readout selection: exact endings + random playouts
    sel = [chess.Board(fen) for fen, _ in exact]
    rsel = random.Random(13)
    for _ in range(240):
        bb = chess.Board()
        for _ in range(rsel.randrange(10, 80)):
            mm = list(bb.legal_moves)
            if not mm:
                break
            bb.push(rsel.choice(mm))
        if not bb.is_game_over():
            sel.append(bb)
    model = FlyV3CX(len(NAMES), sel_boards=sel).to(DEV)
    if os.environ.get("RESUME", "0") == "1" and os.path.exists(STATE):
        model.load_state_dict(torch.load(STATE, weights_only=True), strict=False)
        print("resumed", flush=True)
    opt = torch.optim.Adam(model.parameters(), lr=float(os.environ.get("LR", "3e-4")))
    if not os.path.exists(STATE.replace(".pt", "_orig.pt")) and os.path.exists(STATE):
        import shutil; shutil.copy(STATE, STATE.replace(".pt", "_orig.pt"))

    def checkpoint():
        tmp = STATE + ".tmp"
        torch.save(model.state_dict(), tmp)
        os.replace(tmp, STATE)

    t0 = time.time(); seen = 0; puzzles = 0; nloss = 0; nstep = 0
    loss_acc = 0.0; buf = []; last_ck = time.time(); last_eval_seen = 0; last_ev = {}
    logf = open(LOGF, "a"); syn_rng = random.Random(4)

    def do_step(chunk):
        nonlocal seen, loss_acc, nloss, nstep, last_ck, last_eval_seen, last_ev
        for _ in range(max(1, B // 4)):        # 25% exact endings
            chunk[syn_rng.randrange(len(chunk))] = syn[syn_rng.randrange(len(syn))]
        fvb, slotb, mfb, maskb, tgtb, clb = base_collate(chunk)
        _t = time.time()
        logp, cls_logits = model.batch_forward(fvb, slotb, mfb, maskb)
        tgt = torch.from_numpy(tgtb).to(DEV)
        clst = torch.from_numpy(clb).to(DEV)
        row_loss = -logp.gather(1, tgt.clamp(min=0)[:, None]).squeeze(1)
        mv = tgt >= 0
        mv_term = row_loss[mv].mean() if int(mv.sum()) else torch.zeros((), device=DEV)
        loss = mv_term + torch.nn.functional.cross_entropy(cls_logits, clst) * W_CLS
        opt.zero_grad(); loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        torch.cuda.synchronize()
        seen += len(chunk); loss_acc += float(loss.item()); nloss += 1; nstep += 1
        do_eval = nstep == 1 or seen - last_eval_seen >= int(os.environ.get("EVAL_EVERY", "300000"))
        do_print = nstep == 1 or nstep % 200 == 0
        if do_eval:
            r1 = eval_batch(model, [(f, m2, t) for f, m2, t in ho[:800]])
            sc = class_only_eval(model, syn)
            ev = {"ho_first": round(r1[0], 4), "ho_cls": round(r1[1], 4),
                  "syn_cls": round(sc, 4)}
            last_eval_seen = seen; last_ev = ev
        else:
            ev = {}
        if do_print:
            rss = int(open("/proc/self/statm").read().split()[1]) * 4096 >> 20
            rec = {"ep": ep, "seen": seen, "puzzles": puzzles,
                   "loss": round(loss_acc / max(nloss, 1), 4),
                   "ips": round(seen / (time.time() - t0), 1),
                   "rss_gb": round(rss / 1024, 1)}
            rec.update(last_ev if not ev else ev)
            print(json.dumps(rec), flush=True)
            logf.write(json.dumps(rec) + "\n"); logf.flush()
            loss_acc = 0.0; nloss = 0
        if time.time() - last_ck > 1200:
            checkpoint(); last_ck = time.time(); print("checkpoint saved", flush=True)

    def eval_batch(model, jobs):
        buf = []
        for job in jobs:
            buf.extend(instance(job))
        if not buf: return (0.0, 0.0)
        tot = cor = ctot = ccor = 0
        with torch.no_grad():
            for i in range(0, len(buf), B):
                fvb, slotb, mfb, maskb, tgtb, clb = base_collate(buf[i:i+B])
                logp, cls_logits = model.batch_forward(fvb, slotb, mfb, maskb)
                pred = torch.argmax(logp, dim=1).cpu().numpy()
                cpred = torch.argmax(cls_logits, dim=1).cpu().numpy()
                m2 = tgtb >= 0
                cor += int((pred[m2] == tgtb[m2]).sum()); tot += int(m2.sum())
                ccor += int((cpred == clb).sum()); ctot += len(clb)
        return cor / max(tot, 1), ccor / max(ctot, 1)

    def class_only_eval(model, syn):
        ccor = ctot = 0
        with torch.no_grad():
            for i in range(0, len(syn), B):
                fvb, slotb, mfb, maskb, tgtb, clb = base_collate(syn[i:i+B])
                _, cls_logits = model.batch_forward(fvb, slotb, mfb, maskb)
                cpred = torch.argmax(cls_logits, dim=1).cpu().numpy()
                ccor += int((cpred == clb).sum()); ctot += len(clb)
        return ccor / max(ctot, 1)

    for ep in range(EPOCHS):
        jobs_iter = iter(order)
        pending = deque()
        MAXIN = 2 * WORKERS + 2
        CH = 64
        def submit():
            while len(pending) < MAXIN:
                i0 = next(jobs_iter, None)
                if i0 is None:
                    return
                cj = []
                for _ in range(CH):
                    j = next(jobs_iter, None)
                    if j is None:
                        break
                    cj.append((fens[j], movs[j], thms[j]))
                if cj:
                    pending.append(pool.apply_async(instance_chunk, (cj,)))
        submit()
        while pending:
            res = pending.popleft().get()
            buf.extend(res)
            puzzles += CH
            submit()
            while len(buf) >= B:
                chunk, buf = buf[:B], buf[B:]
                do_step(chunk)
        checkpoint()
        print(f"EPOCH {ep} DONE puzzles={puzzles} seen={seen} "
              f"mins={round((time.time()-t0)/60,1)}", flush=True)


if __name__ == "__main__":
    main()
