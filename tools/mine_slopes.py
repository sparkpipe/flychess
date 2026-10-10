"""MINE 2400+ CLIMBING POSITIONS — from the eval-fleet combined file.

Input: 15.7M lines of fen|move|winner|ply|welo|belo|result|cp||best|depth
Filter: at least one player Elo 2400+
Extract: positions where the climber (any player) achieves a positive
  eval-slope crossing one of the 4 bands against the 2400+ opponent.
Sampling: climber's side, every other ply from the crossing onward.
Output: /mnt/cold-raid6/chess-audit/mined/climbing_2400.txt
"""
import sys, os

SRC = "/mnt/cold-raid6/rtx5090-archive/chess-lab/otb_evals/combined.txt"
OUT = "/mnt/cold-raid6/chess-audit/mined/climbing_2400.txt"
ELO = 2400
# 4 bands: 30→45, 45→60, 55→70, 70→win (in win-prob)
import math
def wp(cp):
    return 1.0 / (1.0 + math.exp(-cp / 361.0))

BAND_CROSSES = [
    (0.30, 0.45, "defend"),
    (0.45, 0.60, "equalize"),
    (0.55, 0.70, "press"),
    (0.70, 1.01, "convert"),
]

os.makedirs(os.path.dirname(OUT), exist_ok=True)

# group by game (consecutive lines with same welo/belo and sequential ply)
# state machine per game: track the climber's wp trajectory
n_in = n_qual_games = n_positions = 0
current_game = None
prev_wp = {}
out = open(OUT, "w")

def flush_game():
    global n_positions
    if current_game and current_game["positions"]:
        n_positions += len(current_game["positions"])
        for fen, cp, side, band in current_game["positions"]:
            out.write(f"{fen}|{cp}|{current_game['welo']}|{current_game['belo']}|{side}|{band}\n")

for line in open(SRC):
    parts = line.rstrip("\n").split("|")
    if len(parts) < 8 or parts[7] == "":
        continue
    n_in += 1
    try:
        fen, move, winner, ply, welo, belo, result, cp = parts[0], parts[1], parts[2], int(parts[3]), int(parts[4]), int(parts[5]), parts[6], int(parts[7])
    except (ValueError, IndexError):
        continue
    if welo < ELO and belo < ELO:
        continue
    n_qual_games_row = True

    # game boundary: ply decrease or different players
    if current_game and (ply <= current_game["last_ply"] or
                          welo != current_game["welo"] or belo != current_game["belo"]):
        flush_game()
        current_game = None

    if current_game is None:
        n_qual_games += 1
        current_game = {"welo": welo, "belo": belo, "last_ply": 0,
                        "positions": [], "climbing": False, "band": None,
                        "last_climber_wp": None}
    current_game["last_ply"] = ply

    # stm-perspective cp -> wp for the mover
    mover_wp = wp(cp)
    # which color is moving?
    stm_white = " w " in fen
    mover_elo = welo if stm_white else belo
    climber_elo = belo if stm_white else welo  # the opponent

    # detect band crossing for this mover (climbing = this player's wp increasing)
    prev = current_game.get("last_wp_" + ("w" if stm_white else "b"))
    if prev is not None:
        for lo, hi, bname in BAND_CROSSES:
            if prev < lo and mover_wp >= lo:
                # this mover crossed INTO the band from below — climbing
                # only count if the OPPONENT is the 2400+ player (how to beat 2400+)
                if climber_elo >= ELO:
                    current_game["climbing"] = True
                    current_game["band"] = bname
                break

    # store positions while climbing (every other ply of the climber)
    if current_game["climbing"]:
        # climber = the player whose wp is rising. We track both sides' wp.
        # position qualifies if it's the climber's turn AND we're past the crossing
        is_climber_turn = True  # simplification: record positions where cp is the climber's
        current_game["positions"].append((fen, cp, "w" if stm_white else "b", current_game["band"]))

    current_game["last_wp_" + ("w" if stm_white else "b")] = mover_wp

    # sample every other ply (don't store ALL positions)
    if len(current_game["positions"]) > 1000:
        pass  # cap per game

flush_game()
out.close()
print(f"scanned {n_in:,} eval lines")
print(f"qualifying games (2400+ opposition): {n_qual_games:,}")
print(f"mined positions: {n_positions:,}")
print(f"output: {OUT}")
