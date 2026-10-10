"""Quick strength test: our net vs a weak opponent (limited-strength SF),
UCI-based match runner. No cutechess needed.

Plays N games at T seconds/move, alternating colors, reports W/L/D.
"""
import sys
import os
import subprocess
import random
import time
import chess
import chess.engine

GAMES = int(os.environ.get("GAMES", "20"))
TIME_PER_MOVE = float(os.environ.get("TPM", "0.5"))
OUR_SF = os.environ.get("OUR_SF",
                        "/home/spec/Stockfish/src/stockfish")
OUR_NET = os.environ.get("OUR_NET",
                         "/home/spec/chess-lab/our_net_v2.nnue")
OPP_SF = os.environ.get("OPP_SF", "/usr/games/stockfish")
# weaken the opponent: limit skill + depth
OPP_SKILL = os.environ.get("OPP_SKILL", "5")
OPP_DEPTH = os.environ.get("OPP_DEPTH", "3")


def play_game(our_engine, opp_engine, our_white, game_num):
    board = chess.Board()
    # randomize opening
    rng = random.Random(game_num)
    for _ in range(rng.randrange(4, 8)):
        if not list(board.legal_moves):
            break
        board.push(rng.choice(list(board.legal_moves)))

    move_num = 0
    while not board.is_game_over() and move_num < 200:
        if (board.turn == chess.WHITE) == our_white:
            result = our_engine.play(board,
                                     chess.engine.Limit(
                                         time=TIME_PER_MOVE))
        else:
            result = opp_engine.play(board,
                                     chess.engine.Limit(
                                         time=TIME_PER_MOVE))
        board.push(result.move)
        move_num += 1

    result = board.result()
    if result == '1/2-1/2' or result == '*':
        return 0.5
    if (result == '1-0') == our_white:
        return 1.0
    return 0.0


def main():
    our = chess.engine.SimpleEngine.popen_uci(OUR_SF)
    our.configure({"EvalFile": OUR_NET, "Threads": 1, "Hash": 64})
    opp = chess.engine.SimpleEngine.popen_uci(OPP_SF)
    opp.configure({"Skill Level": int(OPP_SKILL), "Threads": 1,
                   "Hash": 64})

    scores = []
    for i in range(GAMES):
        our_white = (i % 2 == 0)
        s = play_game(our, opp, our_white, i)
        scores.append(s)
        w = sum(1 for x in scores if x == 1.0)
        l = sum(1 for x in scores if x == 0.0)
        d = sum(1 for x in scores if x == 0.5)
        print(f"game {i+1}/{GAMES}: "
              f"{'W' if s == 1 else ('L' if s == 0 else 'D')} "
              f"(total: {w}W {l}L {d}D, "
              f"score={sum(scores)/len(scores):.2f})", flush=True)

    our.quit()
    opp.quit()
    total = sum(scores) / len(scores)
    w = sum(1 for x in scores if x == 1.0)
    l = sum(1 for x in scores if x == 0.0)
    d = sum(1 for x in scores if x == 0.5)
    elo_gap = -400 * (1 / (1 + 10 ** (0 - 0)) - total) if total != 0.5 else 0
    print(f"\n=== FINAL: {w}W {l}L {d}D of {GAMES} games, "
          f"score={total:.2f}", flush=True)
    print(f"=== opponent: SF 17.1 skill={OPP_SKILL} depth={OPP_DEPTH} "
          f"t={TIME_PER_MOVE}s/move", flush=True)
    print("MATCH-COMPLETE", flush=True)


if __name__ == "__main__":
    main()
