#!/usr/bin/env python3
"""FliPy-Leela: PUCT Monte-Carlo search over the fly brain.

Policy prior  : softmax of the fly's per-move readout scores (move ranking).
Value         : the MENU COMPARISON — top-k mean of the mover's ranked moves
                vs top-k mean of the opponent's ranked moves after the best
                reply, on the fly's shared readout scale:
                    v(STM) = tanh((topk(T_stm) - topk(T_opp)) / tau)
                plus the 3-neuron class pheromone (lost/playable/won) as the
                absolute won-drawn-lost signal.
Trap layer    : at the root, among SAFE moves (not within delta of worse than
                best, opponent's class readout not 'won'), prefer the child
                that maximizes the opponent's chance to err:
                    D(child) = 0.5 * H_norm(softmax(T_opp))
                             + 0.5 * (1 - t_opp_topk / tau_cap)
                flat + low-quality opponent menus = many ways to go wrong.
                Sacrifice a few percentage points of EV while the draw (or
                better) stays in hand.

UCI engine; also importable: bestmove(board, sims=800, **opts).
"""
import sys, os, time, math, chess
import numpy as np
import torch

sys.path.insert(0, "/home/spec/chess-lab")
import fly_v3_full as v3
from train_puzzles_full import FlyV3C, synth_pool  # noqa: E402

DEV = "cuda"
STATE = "/home/spec/chess-lab/flyv3_state.pt"
CONTEMPT = 0.25          # draws are scored as -CONTEMPT: avoid them (operator:
                         # prefer 45% win / 50% loss / 5% draw over 90% draw)

MODEL = None


def model():
    global MODEL
    if MODEL is None:
        v3.feat_vec(chess.Board())
        MODEL = FlyV3C(sorted(v3.FEATURE_KEYS), 26933).to(DEV)
        MODEL.load_state_dict(torch.load(STATE, weights_only=True, map_location=DEV),
                              strict=False)
        MODEL.eval()
    return MODEL


# ---------------- fly evaluation of a single position ----------------
def fly_eval(board):
    """Return (mvs, T numpy, class_probs numpy) for board; STM perspective."""
    m = model()
    if not any(board.generate_legal_moves()):
        return [], np.zeros(0), np.array([0.0, 1.0, 0.0])  # dead position
    with torch.no_grad():
        mvs, T, a = m.scores(board)
        cls = torch.softmax(m.class_logits(a), dim=0)
    return mvs, T.cpu().numpy(), cls.cpu().numpy()


def menu_quality(T, topk):
    k = min(topk, len(T))
    if k == 0: return 0.0
    return float(np.sort(T)[-k:].mean())


def node_value(board, topk, tau):
    """Menu-comparison value for the side to move, in [-1, 1].
    Also returns the opponent-menu stats for the trap layer when the best
    reply position is evaluable."""
    mvs, T, _ = fly_eval(board)
    if not mvs:
        return (-1.0 if board.is_checkmate() else 0.0), None, None
    t_us = menu_quality(T, topk)
    best = mvs[int(np.argmax(T))]
    child = board.copy(stack=False)
    child.push(best)
    if child.is_game_over(claim_draw=False):
        oc = child.outcome(claim_draw=False)
        t_opp = 0.0
        opp_stats = {"H": 0.0, "t": 0.0, "terminal": True}
    else:
        mvs2, T2, _ = fly_eval(child)
        t_opp = menu_quality(T2, topk)
        p = np.exp(T2 - T2.max()); p /= p.sum()
        opp_stats = {"H": float(-(p * np.log(p + 1e-9)).sum() / max(1.0, np.log(len(p)))),
                     "t": t_opp, "terminal": False}
    return math.tanh((t_us - t_opp) / tau), T, opp_stats


class Node:
    __slots__ = ("P", "N", "W", "children", "expanded", "v_own")

    def __init__(self, P=0.0):
        self.P = P; self.N = 0; self.W = 0.0
        self.children = {}; self.expanded = False; self.v_own = None


class Search:
    def __init__(self, board, sims=800, cpuct=1.6, topk=3, tau=2.0,
                 trap_lambda=0.0, trap_delta=0.25, dir_alpha=0.3, dir_frac=0.0,
                 deadline=None, swarm=True, swarm_seed=0):
        self.b0 = board.copy(stack=False)
        self.swarm = swarm; self.swarm_seed = swarm_seed
        self.sims = sims; self.cpuct = cpuct; self.topk = topk; self.tau = tau
        self.trap_lambda = trap_lambda; self.trap_delta = trap_delta
        self.dir_alpha = dir_alpha; self.dir_frac = dir_frac
        self.deadline = deadline
        self.root = Node()
        self.cache = {}                     # epd -> (T, class) reuse

    def fly_cached(self, board):
        key = board.epd()
        hit = self.cache.get(key)
        if hit is None:
            mvs, T, cls = fly_eval(board)
            hit = (mvs, T, cls)
            if len(self.cache) > 20000:
                self.cache.clear()
            self.cache[key] = hit
        return hit

    def expand(self, board, node, ply):
        """Expand node at `board`; returns value for STM at `board`."""
        if board.is_game_over(claim_draw=False):
            oc = board.outcome(claim_draw=False)
            v = -1.0 if (oc and oc.winner is not None) else -CONTEMPT
            node.N = 1; node.W = v
            return v
        if board.is_fifty_moves() or board.is_repetition(3):
            node.N = 1; node.W = -CONTEMPT
            return -CONTEMPT
        mvs, T, cls = self.fly_cached(board)
        p = np.exp(T - T.max()); p /= p.sum()
        node.children = {mv: Node(float(p[i])) for i, mv in enumerate(mvs)}
        node.expanded = True
        if ply == 0 and self.dir_frac > 0:
            d = np.random.default_rng().dirichlet(
                [self.dir_alpha] * len(mvs))
            for i, mv in enumerate(mvs):
                node.children[mv].P = (1 - self.dir_frac) * node.children[mv].P \
                    + self.dir_frac * float(d[i])
        v, T2, opp = node_value(board, self.topk, self.tau)
        node.N = 1; node.W = v
        return v

    def simulate(self, board, node, ply=0):
        """Returns value for STM at this ply."""
        if not node.expanded:
            return self.expand(board, node, ply)
        # PUCT select
        best, best_score = None, -1e18
        sqn = math.sqrt(max(1, node.N))
        for mv, ch in node.children.items():
            q = -(ch.W / ch.N) if ch.N else 0.0   # child value -> our perspective
            s = q + self.cpuct * ch.P * sqn / (1 + ch.N)
            if s > best_score:
                best, best_score = mv, s
        child = board.copy(stack=False)
        child.push(best)
        ch = node.children[best]
        if child.is_game_over(claim_draw=False):
            oc = child.outcome(claim_draw=False)
            v = -1.0 if (oc and oc.winner is not None) else 0.0
            ch.N += 1; ch.W += v
            v_stm = v
        else:
            v_stm = -self.simulate(child, ch, ply + 1)   # negamax
        node.N += 1; node.W += v_stm
        return v_stm

    def run(self):
        if self.b0.is_game_over(claim_draw=False):
            return None, {}
        self.expand(self.b0, self.root, 0)
        if self.sims >= 48:
            return self.run_batched()
        for _ in range(self.sims):
            if self.deadline and time.time() > self.deadline:
                break
            self.simulate(self.b0, self.root, 0)
        stats = {}
        for mv, ch in self.root.children.items():
            stats[mv] = (ch.N, -(ch.W / ch.N) if ch.N else 0.0, ch.P)
        return None, stats

    # ---------------- batched GPU path ----------------
    def select_vl(self, vl=3):
        """Descend by PUCT to an unexpanded leaf; virtual loss on the way."""
        board = self.b0.copy(stack=False)
        node = self.root
        stack = []
        while node.expanded and node.children:
            best, best_score = None, -1e18
            sqn = math.sqrt(max(1, node.N))
            for mv, ch in node.children.items():
                q = -(ch.W / ch.N) if ch.N else 0.0
                s = q + self.cpuct * ch.P * sqn / (1 + ch.N)
                if s > best_score:
                    best, best_score = mv, s
            ch = node.children[best]
            stack.append((node, ch))
            ch.N += vl; ch.W -= vl
            board.push(best)
            node = ch
            if board.is_game_over(claim_draw=False) or board.is_repetition(3):
                break
        return board, node, stack

    def run_batched(self, L=24):
        import flyfeat, random as _rnd
        m = model()
        done = 0
        VL = 3                                  # virtual loss
        gen = _rnd.Random(self.swarm_seed)
        noise = ({"drop": gen.uniform(0.02, 0.06),
                  "jitter": gen.uniform(0.0, 0.04)}) if self.swarm else None
        while done < self.sims:
            if self.deadline and time.time() > self.deadline:
                break
            if self.swarm and self.root.children:
                Ns = [c.N for c in self.root.children.values()]
                top_frac = max(Ns) / max(1, sum(Ns))
                L = max(8, min(48, int(round(24 * (2.0 - top_frac)))))
            n = min(L, self.sims - done)
            done += n
            paths = [self.select_vl(VL) for _ in range(n)]
            leaf_boards = [b for b, _, _ in paths]
            p1 = batch_scores(leaf_boards, m, flyfeat, noise=noise)
            child_boards = []; info = []
            for (b, node, stack), (mvs, T, cls) in zip(paths, p1):
                oc = b.outcome(claim_draw=False)
                if b.is_game_over(claim_draw=False) or b.is_fifty_moves():
                    v = -1.0 if (oc and oc.winner is not None) else -CONTEMPT
                    info.append((node, stack, None, None, v))
                else:
                    p = np.exp(T - T.max()); p /= p.sum()
                    node.children = {mv: Node(float(p[i])) for i, mv in enumerate(mvs)}
                    node.expanded = True
                    best = mvs[int(np.argmax(T))]
                    cb = b.copy(stack=False)
                    cb.push(best)
                    child_boards.append(cb)
                    info.append((node, stack, T, cb, None))
            pend = []
            opp_T = {}
            mate_v = {}
            for k, e in enumerate(info):
                if e[4] is None:
                    oc = e[3].outcome(claim_draw=False)
                    if oc is not None:            # mating / drawing move
                        mate_v[k] = 1.0 if oc.winner is not None else -CONTEMPT
                    else:
                        pend.append((k, e[3]))
            if pend:
                for (k, cb), (_, T2, _) in zip(pend, batch_scores(
                        [cb for _, cb in pend], m, flyfeat, noise=noise)):
                    opp_T[k] = T2
            for k, (node, stack, Tl, cb, v_fixed) in enumerate(info):
                if v_fixed is not None:
                    v = v_fixed
                elif k in mate_v:
                    v = mate_v[k]                 # the move DELIVERS mate
                else:
                    t_us = menu_quality(Tl, self.topk)
                    t_opp = menu_quality(opp_T[k], self.topk)
                    v = math.tanh((t_us - t_opp) / self.tau)
                # each edge node stores ITS OWN mover's value exactly once
                v_child = v
                for parent, edge in reversed(stack):
                    edge.N -= VL; edge.W += VL      # remove virtual loss
                    edge.N += 1; edge.W += v_child
                    v_child = -v_child              # parent sees the flip
            self.root.N += n                        # root visits grow per sim
        stats = {}
        for mv, ch in self.root.children.items():
            stats[mv] = (ch.N, -(ch.W / ch.N) if ch.N else 0.0, ch.P)
        return None, stats


MAXL = 256


def batch_scores(boards, m, flyfeat, noise=None):
    """Batched column-layout propagate; [(mvs, T, cls)] per board.
    noise={'drop','jitter'} turns each batch into a swarm of slightly
    different flies: sensory dropout + propagation jitter."""
    Lb = len(boards)
    if not Lb:
        return []
    fvb = np.stack([flyfeat.feat_vec(b) for b in boards])
    slotb = np.zeros((Lb, MAXL), np.int64)
    mfb = np.zeros((Lb, MAXL, 8), np.float32)
    maskb = np.zeros((Lb, MAXL), bool)
    keep = []
    for i, b in enumerate(boards):
        mvs = list(b.legal_moves)
        keep.append(mvs[:MAXL])
        for j, mv in enumerate(keep[i]):
            slotb[i, j] = mv.from_square * 64 + mv.to_square
            mfb[i, j] = flyfeat.move_feats(b, mv)
            maskb[i, j] = True
    with torch.no_grad():
        x = torch.from_numpy(fvb).to(DEV)
        if noise is not None:
            drop_mask = (torch.rand_like(x) > noise["drop"]).float()
            x = x * drop_mask
        s = torch.clamp(m.W_sens(x), -6, 6)
        a = torch.zeros(m.N, Lb, device=DEV)
        a[m.sensory_idx] = s.T
        for _ in range(2):
            a = 0.5 * a + 0.5 * (m.WT @ a)
            if noise is not None:
                a = a + noise["jitter"] * torch.randn_like(a)
            a = torch.clamp(a, -20.0, 20.0)
        cols = torch.arange(Lb, device=DEV).unsqueeze(1)
        slots = torch.from_numpy(slotb).to(DEV)
        T = m.theta[slots] * a[m.readout_idx[slots], cols] \
            + torch.from_numpy(mfb).to(DEV) @ m.theta_mv
        T = T.masked_fill(~torch.from_numpy(maskb).to(DEV), -1e9)
        cls = torch.softmax(m.theta_cls.unsqueeze(0) * a[m.cls_idx].T, dim=1)
    Tn = T.cpu().numpy(); Cn = cls.cpu().numpy()
    out = []
    for i, b in enumerate(boards):
        K = int(maskb[i].sum())
        out.append((keep[i], Tn[i][:K], Cn[i]))
    return out


# ---------------- stance controller + phase blending ----------------
PVALS = (chess.PAWN, chess.KNIGHT, chess.BISHOP, chess.ROOK, chess.QUEEN)


def composition(board):
    """Full material-composition encoding — handles ALL imbalances: both
    armies as (P,N,B,R,Q), signed difference vector, conventional balance,
    and the sharp archetypes (R vs 2 minors, Q vs army, 2R vs Q)."""
    def army(color):
        return tuple(bin(board.pieces_mask(pt, color)).count("1")
                     for pt in PVALS)
    mine, theirs = army(chess.WHITE), army(chess.BLACK)
    diff = tuple(x - y for x, y in zip(mine, theirs))
    vals = (1, 3, 3, 5, 9)
    balance = sum(d * v for d, v in zip(diff, vals))
    same = mine == theirs
    q_us, q_them = mine[4], theirs[4]
    minors_us, minors_them = mine[1] + mine[2], theirs[1] + theirs[2]
    q_vs_army = ((q_us >= 1 and q_them == 0 and (theirs[3] + minors_them) >= 2)
                 or (q_them >= 1 and q_us == 0 and (mine[3] + minors_us) >= 2))
    r_vs_2minors = ((mine[3] - theirs[3] == 1 and minors_us - minors_them == -2)
                    or (theirs[3] - mine[3] == 1 and minors_us - minors_them == 2))
    r_vs_minor_pawn = (abs(mine[3] - theirs[3]) == 1
                       and abs(minors_us - minors_them) == 1)
    two_rooks_vs_queen = (abs(mine[3] - theirs[3]) == 2
                          and abs(q_us - q_them) == 1)
    imbalance = sum(abs(x - y) for x, y in zip(mine, theirs))
    return {"mine": mine, "theirs": theirs, "diff": diff,
            "balance": balance / 12.0, "same_army": same,
            "q_vs_army": q_vs_army, "r_vs_2minors": r_vs_2minors,
            "r_vs_minor_pawn": r_vs_minor_pawn,
            "two_rooks_vs_queen": two_rooks_vs_queen,
            "asymmetry": min(imbalance, 6) / 6.0}


def phase_weights(board):
    """Normalized (opening, middlegame, endgame) blend. Material does most of
    it, but tactical material (queens/rooks on) keeps middlegame weight alive
    in thin positions (operator: some endgames are very tactical)."""
    pm = board.piece_map()
    n = len(pm)
    queens = sum(1 for p in pm.values() if p.piece_type == chess.QUEEN)
    rooks = sum(1 for p in pm.values() if p.piece_type == chess.ROOK)
    minors = sum(1 for p in pm.values() if p.piece_type in (chess.KNIGHT, chess.BISHOP))
    pawns = sum(1 for p in pm.values() if p.piece_type == chess.PAWN)
    # opening: full army and pieces still on home ranks
    home = sum(1 for sq, p in pm.items()
               if (p.color == chess.WHITE and chess.square_rank(sq) <= 1)
               or (p.color == chess.BLACK and chess.square_rank(sq) >= 6))
    undeveloped = home / max(n, 1)
    w_open = max(0.0, (n >= 24) * 0.7 + 0.6 * undeveloped * (n / 32.0))
    # endgame: thin material; queens off matters most
    comp = composition(board)
    sharp = queens > 0 or rooks >= 2 or comp["r_vs_2minors"] \
        or comp["two_rooks_vs_queen"] or comp["q_vs_army"]
    w_end = max(0.0, min(1.0, (16 - n) / 10.0)) * (1.0 if queens == 0 else 0.35)
    if pawns == 0 and minors + rooks <= 3 and not sharp:
        w_end = max(w_end, 0.8)
    w_mid = max(0.0, 1.0 - w_open - w_end)
    if sharp and w_mid < 0.3:
        w_mid = 0.3
    tot = w_open + w_mid + w_end
    return (w_open / tot, w_mid / tot, w_end / tot)


def stance(board, cls):
    """(trap_lambda, trap_delta, mode) for this position.
    Operator philosophy:
      winning + endgame  -> nurture advantage, play for two results
      balanced           -> maximum chaos (the trap algo)
      behind             -> desperation: big sacrifices, nothing to lose
      winning + midgame  -> tactics-first, moderate chaos"""
    p_loss, p_play, p_win = float(cls[0]), float(cls[1]), float(cls[2])
    w_open, w_mid, w_end = phase_weights(board)
    comp = composition(board)
    if p_win > 0.6 and w_end > 0.5:
        return (0.1, 0.10, "nurture")
    if p_loss > 0.55 or (p_win < 0.2 and p_play < 0.55):
        return (1.6, 0.75, "desperation")
    lam = 1.0 + 0.5 * w_mid + 0.4 * comp["asymmetry"] \
        + (0.2 if comp["q_vs_army"] else 0.0) \
        + (0.3 if comp["r_vs_2minors"] or comp["two_rooks_vs_queen"] else 0.0) \
        + (0.1 if abs(comp["balance"]) < 0.08 and not comp["same_army"] else 0.0)
    return (min(lam, 2.0), 0.45, "chaos")


def choose_root_move(board, stats, cls_root, trap_lambda, trap_delta,
                     topk, tau, tau_cap=6.0, stance_auto=True):
    """Trap layer + stance controller. Safe set = within delta of best value
    AND opponent class readout at child says opponent is not winning.
    Among safe moves prefer max opponent-difficulty D — except in nurture
    stance (winning endgame), where the best Q wins outright."""
    if not stats: return None
    if stance_auto:
        trap_lambda, trap_delta, mode = stance(board, cls_root)
    else:
        mode = "chaos"
    N = np.array([s[0] for s in stats.values()], dtype=np.float64)
    Q = np.array([s[1] for s in stats.values()], dtype=np.float64)
    mvs = list(stats.keys())
    v_best = float(Q.max())
    safe = []
    for i, mv in enumerate(mvs):
        if Q[i] < v_best - trap_delta:
            continue
        if N[i] < max(3, N.max() * 0.05):      # not enough evidence
            continue
        child = board.copy(stack=False)
        child.push(mv)
        if child.is_insufficient_material():
            continue                           # never trade into a dead draw
        oc0 = child.outcome(claim_draw=False)
        if oc0 is not None and oc0.winner is None:
            continue                           # no immediate stalemate draws
        mvs2, T2, opp_cls = fly_eval(child)
        p_opp_win = float(opp_cls[2])          # opponent to move: their p_win
        if p_opp_win > 0.85:                   # never walk into their win
            continue
        if len(T2) == 0:                       # terminal child: Q decides alone
            safe.append((float(Q[i]), mv, float(Q[i]), 0.0))
            continue
        p2 = np.exp(T2 - T2.max()); p2 /= p2.sum()
        H = float(-(p2 * np.log(p2 + 1e-9)).sum() / max(1.0, np.log(len(p2))))
        t_opp = menu_quality(T2, topk)
        D = 0.5 * H + 0.5 * max(0.0, 1.0 - t_opp / tau_cap)
        safe.append((float(Q[i]) + trap_lambda * D, mv, float(Q[i]), D))
    if not safe:                               # nothing safe: pure best move
        return mvs[int(np.argmax(Q))]
    if mode == "nurture":                      # play for two results: max Q
        return max(safe, key=lambda t: t[2])[1]
    safe.sort(key=lambda t: t[0], reverse=True)
    return safe[0][1]


def bestmove(board, sims=800, cpuct=1.6, topk=3, tau=2.0, trap_lambda=0.5,
             trap_delta=0.25, dir_frac=0.0, movetime=None, swarm=True,
             swarm_seed=0):
    deadline = time.time() + movetime if movetime else None
    s = Search(board, sims=sims, cpuct=cpuct, topk=topk, tau=tau,
               trap_lambda=0.0, trap_delta=trap_delta, dir_frac=dir_frac,
               deadline=deadline, swarm=swarm, swarm_seed=swarm_seed)
    _, stats = s.run()
    mvs0, T0, cls0 = fly_eval(board)
    mv = choose_root_move(board, stats, cls0, trap_lambda, trap_delta, topk, tau)
    return (mv.uci() if mv else None), stats


# ---------------- UCI ----------------
def uci_loop():
    opts = {"Simulations": 800, "CPuct": 1.6, "TopK": 3, "TauMenu": 2.0,
            "TrapLambda": 0.5, "TrapDelta": 0.25, "DirFrac": 0.0,
            "MoveOverhead": 200, "Swarm": True, "SwarmSeed": 0}
    board = chess.Board()
    print("id name FliPy-Leela-0.1")
    print("id author SparkPipe")
    for k, v in opts.items():
        t = "check" if isinstance(v, bool) else \
            ("spin" if isinstance(v, int) else "string")
        if k in ("CPuct", "TauMenu", "TrapLambda", "TrapDelta", "DirFrac"):
            t = "string"                       # float values; fastchess-safe
        if t == "spin":
            lo, hi = (0, 10**6) if k in ("MoveOverhead", "SwarmSeed") else (1, 10**6)
            print(f'option name {k} type spin default {v} min {lo} max {hi}')
        else:
            print(f'option name {k} type {t} default {v}')
    print("uciok", flush=True)
    for line in sys.stdin:
        line = line.strip()
        if not line: continue
        cmd, *rest = line.split(maxsplit=1)
        arg = rest[0] if rest else ""
        if cmd == "isready":
            model(); print("readyok", flush=True)
        elif cmd == "ucinewgame":
            board = chess.Board()
        elif cmd == "position":
            board = chess.Board()
            if arg.startswith("fen "):
                parts = arg[4:].split(" moves ")
                board = chess.Board(parts[0])
                moves = parts[1] if len(parts) > 1 else ""
            else:
                moves = arg.split("moves ", 1)[1] if "moves " in arg else ""
            for mv in moves.split():
                board.push_uci(mv)
        elif cmd == "go":
            toks = arg.split()
            def val(k):
                return int(toks[toks.index(k) + 1]) if k in toks else None
            mt = None
            if "movetime" in toks:
                mt = val("movetime") - opts["MoveOverhead"]
            elif "wtime" in toks or "btime" in toks:
                myt = val("wtime") if board.turn == chess.WHITE else val("btime")
                inc = val("winc") if board.turn == chess.WHITE else val("binc")
                if myt:
                    mt = myt / 22 + 0.8 * (inc or 0) - opts["MoveOverhead"]
                    if myt > 4000:                     # clock healthy: think
                        mt = max(1000, mt)
            if mt is not None:
                deadline = time.time() + max(50, mt) / 1000
            else:
                deadline = None
            s = Search(board, sims=opts["Simulations"], cpuct=opts["CPuct"],
                       topk=opts["TopK"], tau=opts["TauMenu"], trap_lambda=0.0,
                       trap_delta=opts["TrapDelta"], dir_frac=opts["DirFrac"],
                       deadline=deadline, swarm=opts["Swarm"],
                       swarm_seed=int(opts["SwarmSeed"]))
            _, stats = s.run()
            mvs0, T0, cls0 = fly_eval(board)
            mv = choose_root_move(board, stats, cls0, opts["TrapLambda"],
                                  opts["TrapDelta"], opts["TopK"], opts["TauMenu"])
            if mv is None:
                print("bestmove (none)", flush=True)
            else:
                print(f"bestmove {mv.uci()}", flush=True)
        elif cmd == "setoption":
            kv = arg.replace("name ", "").split(" value ")
            if len(kv) == 2 and kv[0] in opts:
                old = opts[kv[0]]
                if isinstance(old, bool):
                    opts[kv[0]] = kv[1].strip().lower() == "true"
                elif isinstance(old, int):
                    opts[kv[0]] = int(float(kv[1]))
                elif isinstance(old, float):
                    opts[kv[0]] = float(kv[1])
                else:
                    opts[kv[0]] = kv[1]
        elif cmd == "quit":
            break


if __name__ == "__main__":
    uci_loop()
