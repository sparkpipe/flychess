#!/usr/bin/env python3
"""Champion-fly post-training: fine-tune the generalist flybrain on the games
of ONE player (Tal, Morphy, Petrosian, Capablanca, ...).

Input:  the HF chess_games parquet (14M games, 1600-2024, has player names)
        + a player name.
Method: positions where the player is to move, ply 8..60; CE on THEIR move.
        The generalist keeps its class pheromone; a low-LR fine-tune bends the
        policy toward the champion's choices -> <player>_fly.pt.
Output: ~/chess-lab/<player>_fly.pt

Usage: python3 posttrain_champions.py PARQUET "Mikhail Tal" [steps]
"""
import sys, os, json, time, chess
import numpy as np, torch
import multiprocessing as mp

sys.path.insert(0, "/home/spec/chess-lab")
import fly_v3_full as v3
import flyfeat
from train_puzzles_full import FlyV3C, collate, batch_forward

DEV = "cuda"
BASE = "/home/spec/chess-lab/flyv3_state.pt"


def player_positions(parquet, player, max_games=20000, plies=(8, 60)):
    """Yield (fen, move_uci) pairs where the player is to move."""
    import pandas as pd, pyarrow.parquet as pq
    df = pq.read_table(parquet, columns=["white", "black", "moves"]).to_pandas()
    m = (df["white"] == player) | (df["black"] == player)
    df = df[m].head(max_games)
    print(f"{player}: {len(df)} games", flush=True)
    out = []
    for _, row in df.iterrows():
        as_white = row["white"] == player
        mvlist = row["moves"].split() if isinstance(row["moves"], str) else []
        b = chess.Board()
        plies_done = 0
        for tok in mvlist:
            if tok in ("1-0", "0-1", "1/2-1/2", "*"):
                break
            if tok[0].isdigit() and tok.endswith("."):
                continue
            try:
                mv = b.parse_san(tok)
            except Exception:
                break
            mover_is_player = (b.turn == chess.WHITE) == as_white
            if mover_is_player and plies_done >= plies[0] and plies_done <= plies[1]:
                out.append((b.fen(), mv.uci()))
            b.push(mv)
            plies_done += 1
            if b.is_game_over() or plies_done > plies[1]:
                break
    print(f"{player}: {len(out)} training positions", flush=True)
    return out


def instance(job):
    fen, uci = job
    try:
        b = chess.Board(fen)
        mv = chess.Move.from_uci(uci)
        if mv not in b.legal_moves:
            return []
        fv = flyfeat.feat_vec(b)
        mvs = list(b.legal_moves)
        slots = np.array([m.from_square * 64 + m.to_square for m in mvs], np.int64)
        mf = np.stack([flyfeat.move_feats(b, m) for m in mvs]).astype(np.float32)
        return [(fv, slots, mvs.index(mv), mf, 1)]
    except Exception:
        return []


def main():
    parquet = sys.argv[1]
    player = sys.argv[2]
    steps = int(sys.argv[3]) if len(sys.argv) > 3 else 4000
    jobs = player_positions(parquet, player)
    import random
    random.Random(1).shuffle(jobs)
    v3.feat_vec(chess.Board())
    flyfeat.feat_vec(chess.Board())
    model = FlyV3C(sorted(v3.FEATURE_KEYS), 26933).to(DEV)
    model.load_state_dict(torch.load(BASE, weights_only=True), strict=False)
    print("loaded generalist", flush=True)
    opt = torch.optim.Adam(model.parameters(), lr=1e-4)   # gentle fine-tune
    B = 128
    buf = []
    t0 = time.time()
    n = 0
    rng = random.Random(0)
    while n < steps:
        j = jobs[rng.randrange(len(jobs))]
        buf.extend(instance(j))
        while len(buf) >= B:
            chunk, buf = buf[:B], buf[B:]
            fvb, slotb, mfb, maskb, tgtb, clb = collate(chunk)
            logp, cls_logits, _ = batch_forward(model, fvb, slotb, mfb, maskb)
            tgt = torch.from_numpy(tgtb).to(DEV)
            loss = -logp.gather(1, tgt[:, None]).mean()
            opt.zero_grad(); loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            n += 1
            if n % 500 == 0:
                print(json.dumps({"step": n, "loss": round(float(loss.item()), 4),
                                  "mps": round(n / (time.time() - t0), 1)}), flush=True)
            if n >= steps:
                break
    out = f"/home/spec/chess-lab/{player.split()[-1].lower()}_fly.pt"
    torch.save(model.state_dict(), out)
    print("saved", out, flush=True)


if __name__ == "__main__":
    main()
