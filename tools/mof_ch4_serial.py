"""ch4 size-proof (operator ruling): train ch4's sections SERIALLY to
the 98% exhaustive gate; if a section caps below 98 and is bigger than
MIN_SPLIT, split it in half (feature-space) and continue — the bisection
finds the section size at which a dedicated fly reaches 98%. Everything
serial, one GPU, PASS markers per leaf.
"""
import os
import sys
import json
import random
import numpy as np
import torch
import chess

sys.path.insert(0, "/home/spec/chess-lab")
import fly_curriculum as fc
import flyfeat_cb
from fly_curriculum import DEV, load_pools, FAM_SCORE

CAP = int(os.environ.get("CAP_STEPS", "12000"))
MIN_SPLIT = 8


def train_to_gate(rows, tag, state):
    torch.manual_seed(0)
    rmap = fc.build_retino_map(mode="geo")
    m = fc.FlyCB(len(flyfeat_cb.FEATURE_KEYS), sel_boards=None,
                 readout="variance").to(DEV)
    m.retino = rmap
    m.retino_gain = torch.nn.Parameter(torch.ones(7) * 2.0).to(DEV)
    if os.path.exists(state):
        m.load_state_dict(torch.load(state, weights_only=True),
                          strict=False)
    opt = torch.optim.Adam(m.parameters(), lr=3e-4)
    rng = random.Random(6000 + hash(tag) % 100000)
    FAM_SCORE.update({rows[0].get("pool", tag): 0.0})
    step = 0
    best = 0.0
    best_sd = None
    best_solved = frozenset()        # snapshot at the PEAK (operator ruling): the
                          # final state may sit below the best; the split
                          # must use the fly's best knowledge
    while step < CAP:
        for _ in range(500):
            step += 1
            fc.tb_step(m, opt, rows, rng)
        torch.save(m.state_dict(), state + ".tmp")
        os.replace(state + ".tmp", state)
        pair, _, _ = fc.gate_tb(m, rows, random.Random(777),
                                exhaustive=True)
        FAM_SCORE.update({rows[0].get("pool", tag): pair})
        if pair > best:
            best = pair
            best_sd = {k: v.detach().cpu().clone()
                       for k, v in m.state_dict().items()}
            # the snapshot carries its own answer set (operator ruling):
            # the split uses the RECORDED solved-set from the peak; the
            # later re-eval is verification, never the selector
            sv, fl = eval_solved(m, rows)
            best_solved = frozenset(e["fen"] for e in sv)
            print(json.dumps({"tag": tag, "step": step, "NEW-BEST":
                              round(pair, 4),
                              "solved": len(best_solved),
                              "of": len(rows)}), flush=True)
        print(json.dumps({"tag": tag, "step": step,
                          "exhaustive": round(pair, 4),
                          "best": round(best, 4)}), flush=True)
        if pair >= 0.98:
            return True, best, m, best_solved
    if best_sd is not None:
        m.load_state_dict(best_sd)
        sv, fl = eval_solved(m, rows)
        now = frozenset(e["fen"] for e in sv)
        print(json.dumps({"tag": tag, "VERIFY": now == best_solved,
                          "recorded": len(best_solved),
                          "replayed": len(now)}), flush=True)
    return False, best, m, best_solved


def eval_solved(model, rows):
    """Failure-driven split (operator ruling): the positions the capped
    fly SOLVES form one subsection; the ones it FAILS form another."""
    solved, failed = [], []
    FLIP = {"win": "loss", "loss": "win", "draw": "draw",
            "cursed_win": "cursed_loss", "cursed_loss": "cursed_win"}
    model.eval()
    with torch.no_grad():
        for ci in range(0, len(rows), 64):
            chunk = rows[ci:ci + 64]
            keep = []
            for e in chunk:
                try:
                    b = chess.Board(e["fen"])
                except Exception:
                    continue
                if b.is_game_over() or not list(b.legal_moves):
                    continue
                keep.append(e)
            if not keep:
                continue
            Bn = len(keep)
            boards = [chess.Board(e["fen"]) for e in keep]
            M = max(len(list(b.legal_moves)) for b in boards)
            F = flyfeat_cb.MOVE_DIMS
            fvb = np.stack([flyfeat_cb.feat_vec(b)[0] for b in boards])
            slotb = np.zeros((Bn, M), np.int64)
            pcrowb = np.zeros((Bn, M), np.int64)
            mfb = np.zeros((Bn, M, F), np.float32)
            maskb = np.zeros((Bn, M), bool)
            psb = np.zeros((Bn, M), np.float32)
            ps2b = np.zeros((Bn, M), np.float32)
            thb = np.zeros((Bn, M), np.float32)
            for i, b in enumerate(boards):
                mvs = list(b.legal_moves)
                p2 = {(pm.from_square, pm.to_square)
                      for pm in b.pseudo_legal_moves}
                for j, mv in enumerate(mvs):
                    slotb[i, j] = mv.from_square * 64 + mv.to_square
                    pc = b.piece_at(mv.from_square)
                    pcrowb[i, j] = fc._PC_IDX[pc.piece_type] if pc else 0
                    mfb[i, j] = flyfeat_cb.move_feats(b, mv)
                    maskb[i, j] = True
                    psb[i, j] = 1.0 if (
                        b.attacks_mask(mv.from_square)
                        & chess.BB_SQUARES[mv.to_square]) else 0.0
                    ps2b[i, j] = 1.0 if (mv.from_square,
                                         mv.to_square) in p2 else 0.0
                    b.push(mv)
                    thb[i, j] = min(
                        bin(b.attacks_mask(mv.to_square)
                            & b.occupied_co[b.turn]).count("1"), 4) / 4.0
                    b.pop()
            T, logp, T_all, clsg = fc.forward(model, fvb, slotb, pcrowb,
                                              mfb, maskb, psb, thb, ps2b)
            picks = torch.argmax(T, dim=1).tolist()
            for i, e in enumerate(keep):
                ch = e.get("children", {})
                opt = {"win": "loss", "cursed_win": "loss",
                       "draw": "draw", "cursed_loss": "win",
                       "loss": "win"}[e["cat"]]
                optset = {u for u, c in ch.items()
                          if FLIP.get(c.get("cat")) == opt}
                mvs = list(boards[i].legal_moves)
                hit = mvs[picks[i]].uci() in optset
                (solved if hit else failed).append(e)
    model.train()
    return solved, failed


def write_pool(rows, pool):
    path = f"/home/spec/chess-lab/tbpools/{pool}.jsonl"
    with open(path, "w") as f:
        for e in rows:
            f.write(json.dumps(dict(e, pool=pool)) + "\n")
    return path


def run(rows, tag, depth=0, model=None):
    state = f"/home/spec/chess-lab/flies_mof/l0_{tag}.pt"
    ok, best, m, best_solved = train_to_gate(rows, tag, state)
    if ok:
        print(f"PROOF {tag}: PASSED (n={len(rows)})", flush=True)
        return [(tag, len(rows), "PASS")]
    if len(rows) > MIN_SPLIT * 2 and depth < 4:
        solved = [e for e in rows if e["fen"] in best_solved]
        failed = [e for e in rows if e["fen"] not in best_solved]
        if len(solved) < MIN_SPLIT or len(failed) < MIN_SPLIT:
            print(f"PROOF {tag}: FAIL (n={len(rows)}, best {best:.3f}, "
                  f"outcome split too small "
                  f"({len(solved)}+{len(failed)})", flush=True)
            return [(tag, len(rows), f"FAIL@{best:.3f}")]
        print(f"SPLIT-BY-OUTCOME {tag} (n={len(rows)}, best "
              f"{best:.3f}) -> solved {len(solved)} + failed "
              f"{len(failed)}", flush=True)
        pa = run([dict(e, pool=f"{tag}p") for e in solved],
                 f"{tag}p", depth + 1)
        pb = run([dict(e, pool=f"{tag}f") for e in failed],
                 f"{tag}f", depth + 1)
        return pa + pb
    print(f"PROOF {tag}: FAIL (n={len(rows)}, best {best:.3f})",
          flush=True)
    return [(tag, len(rows), f"FAIL@{best:.3f}")]


def main():
    flyfeat_cb.feat_vec(chess.Board())
    import glob as _g
    results = []
    for path in sorted(_g.glob("/home/spec/chess-lab/tbpools/"
                               "DEGM_Ch4_s*.jsonl")):
        nm = os.path.basename(path).replace(".jsonl", "")
        rows = load_pools([nm])
        if not rows:
            continue
        results += run(rows, nm)
    print("CH4-PROOF-COMPLETE " + json.dumps(results), flush=True)


if __name__ == "__main__":
    main()
