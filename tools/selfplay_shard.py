"""Self-play shard runner for spark CPU fleet (aarch64 fork).

Plays every opening in the shard file twice (each color) between two
identical stacked-head engines at movetime control. Writes one PGN.

Usage: selfplay_shard.py SHARD.epd OUT.pgn MOVETIME_S SLOTS [ENGINE_PATH] [HEAD]
"""
import sys, os, chess, chess.engine, chess.pgn

FORK = os.environ.get("SF_BIN", os.path.expanduser("~/flychess/Stockfish/src/stockfish"))
R = "/mnt/cold-raid6/chess-audit"
NETS = os.environ.get("NETS_DIR", os.path.expanduser("~/flychess/nets"))
HEAD = os.environ.get("SF_HEAD", os.path.expanduser("~/flychess/head.evh"))
EXPERTS = ["balanced_l0", "balanced_l1", "balanced_l2", "balanced_l3",
           "nvb", "nvr", "bvr", "rv2m", "qvmat", "oppb",
           "dvoretsky", "exchanges", "tactics"]


def engine_opts():
    o = {"EvalFile": f"{NETS}/{EXPERTS[0]}.nnue"}
    for i, e in enumerate(EXPERTS[1:]):
        o[f"EvalFile{i + 2}"] = f"{NETS}/{e}.nnue"
    o["StackHead"] = HEAD
    return o


def play_pair(opening_fen, idx, mt, log):
    """Two games: A white then A black. Returns (idx, results)."""
    outs = []
    for flip in (False, True):
        ea = chess.engine.SimpleEngine.popen_uci(FORK)
        eb = chess.engine.SimpleEngine.popen_uci(FORK)
        try:
            ea.configure(engine_opts()); eb.configure(engine_opts())
            board = chess.Board(opening_fen)
            game = chess.pgn.Game()
            game.headers["FEN"] = opening_fen
            node = game
            plies = 0
            while not board.is_game_over(claim_draw=True) and plies < 400:
                eng = eb if (board.turn == chess.WHITE) == flip else ea
                mv = eng.play(board, chess.engine.Limit(time=mt)).move
                board.push(mv)
                node = node.add_variation(mv)
                plies += 1
            res = board.result(claim_draw=True)
            game.headers["Result"] = res
            game.headers["White"] = "spA" if not flip else "spB"
            game.headers["Black"] = "spB" if not flip else "spA"
            game.headers["PlyCount"] = str(plies)
            outs.append((res, plies, game))
        finally:
            ea.quit(); eb.quit()
    with open(log, "a") as f:
        for res, plies, game in outs:
            print(game, file=f, end="\n\n")
    return idx, [o[0] for o in outs], [o[1] for o in outs]


def main():
    shard, out, mt, slots = sys.argv[1], sys.argv[2], float(sys.argv[3]), int(sys.argv[4])
    fens = [" ".join(l.split()[:4]) for l in open(shard)
            if l.strip() and not l.startswith("#")]
    # sequential round-robin over slots keeps every core busy without oversub
    from concurrent.futures import ThreadPoolExecutor
    done = 0
    with ThreadPoolExecutor(max_workers=slots) as ex:
        futs = [ex.submit(play_pair, fen, i, mt, out) for i, fen in enumerate(fens)]
        for fu in futs:
            idx, results, plies = fu.result()
            done += 1
            if done % 16 == 0:
                print(f"  {done}/{len(fens)} openings ({sum(plies)} last plies)", flush=True)
    print(f"SHARD_DONE: {len(fens)} openings -> {out}", flush=True)


if __name__ == "__main__":
    main()
