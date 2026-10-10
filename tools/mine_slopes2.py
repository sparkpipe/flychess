"""MINE BOTH-SIDE SLOPE POSITIONS — positive and anti (negative slope).

POSITIVE: how any player gained eval against a 2400+ opponent.
  Opponent must be 2400+. Climber can be any rating.
ANTI: how strong players went on a negative slope (any opponent).
  The declining player must be 2400+. Opponent can be any rating.
  Anti bands: 70%→55%, 55%→40%, 45%→30% (includes start positions:
  white starts at 55%, black at 45%).

Game result is irrelevant — only the within-game segments matter.

Input: eval-fleet combined.txt (15.7M winner-side positions with cp).
Note: fleet has WINNER-side only. We need BOTH sides.
The original dump (otb_complete_dump_both.txt, 126M lines) has all
positions but no cp. We match fleet evals to dump positions by position-part.

Since matching 126M lines is expensive, we use the fleet's cp values
for winner-side positions and infer loser-side from consecutive
winner cps (loser's cp ≈ -winner's cp at the same position).
Actually: the fleet has positions at every other ply (winner to move).
Loser positions are the OTHER plies. We can compute loser cp at
the loser's ply as approximately -(winner cp at next winner ply) evaluated
from the loser's perspective... this is complex.

Simpler approach: use the position_all.txt (7.2M) which has both sides
(sampled every ply), plus the depth-DB's cp@d20 evaluations. The depth-DB
is nearly complete (5.58M remaining, ~35 min). Once done, we join with
game results and ElOs from the dump metadata.

For NOW: mine from what we have — the fleet's winner-side positions.
For each game, the winner's wp trajectory gives us:
  - positive slope segments (winner climbing) = positive signal
  - anti segments: we need LOSER climbing, which requires loser-side cps.
    Infer from winner cps: if winner's wp DROPS, the loser was climbing.
    Anti = positions where the (any-side) player's wp crossed downward
    through the anti bands.
"""
import sys, os, math, collections

SRC = "/mnt/cold-raid6/rtx5090-archive/chess-lab/otb_evals/combined.txt"
OUT_POS = "/mnt/cold-raid6/chess-audit/mined/positive_2400.txt"
OUT_ANTI = "/mnt/cold-raid6/chess-audit/mined/anti_2400.txt"
ELO = 2400

def wp(cp):
    return 1.0 / (1.0 + math.exp(-cp / 361.0))

POS_BANDS = [(0.30, 0.45, "p30to45"), (0.45, 0.60, "p45to60"), (0.55, 0.70, "p55to70"), (0.70, 1.01, "p70toW")]
ANTI_BANDS = [(0.70, 0.55, "a70to55"), (0.55, 0.40, "a55to40"), (0.45, 0.30, "a45to30")]

os.makedirs(os.path.dirname(OUT_POS), exist_ok=True)

# init: white starts at 0.55, black at 0.45 (operator ruling)
INIT_WP = {"w": 0.55, "b": 0.45}

current_game = None
n_games = 0
n_pos_mined = 0
n_anti_mined = 0
out_pos = open(OUT_POS, "w")
out_anti = open(OUT_ANTI, "w")

def flush(game):
    global n_pos_mined, n_anti_mined
    if not game or not game["positions"]:
        return
    for entry in game["positions"]:
        if entry["signal"] == "pos":
            out_pos.write(f"{entry['fen']}|{entry['cp']}|{game['welo']}|{game['belo']}|{entry['side']}|{entry['band']}\n")
            n_pos_mined += 1
        else:
            out_anti.write(f"{entry['fen']}|{entry['cp']}|{game['welo']}|{game['belo']}|{entry['side']}|{entry['band']}\n")
            n_anti_mined += 1

for line in open(SRC):
    parts = line.rstrip("\n").split("|")
    if len(parts) < 8 or not parts[7] or parts[7] == "None":
        continue
    try:
        fen, move, winner, ply, welo, belo, result, cp = (
            parts[0], parts[1], parts[2], int(parts[3]),
            int(parts[4]), int(parts[5]), parts[6], int(parts[7]))
    except (ValueError, IndexError):
        continue

    # game boundary
    if current_game and (ply <= current_game["last_ply"] or
                          welo != current_game["welo"] or belo != current_game["belo"]):
        flush(current_game)
        current_game = None

    if current_game is None:
        n_games += 1
        current_game = {"welo": welo, "belo": belo, "last_ply": 0,
                        "positions": [], "prev_wp": INIT_WP[winner],
                        "pos_active": False, "anti_active": False,
                        "pos_band": None, "anti_band": None}
    current_game["last_ply"] = ply

    # fleet cp is from the WINNER's perspective (stm = winner)
    mover_wp = wp(cp)
    prev = current_game["prev_wp"]

    # POSITIVE signal: climber (any rating) gains eval vs 2400+ opponent
    # the mover here IS the game winner (fleet is winner-side)
    # the opponent must be 2400+
    climber_elo = welo if winner == "w" else belo
    opponent_elo = belo if winner == "w" else welo
    if opponent_elo >= ELO:
        for lo, hi, _ in POS_BANDS:
            if prev < lo and mover_wp >= lo:
                current_game["pos_active"] = True
                current_game["pos_band"] = f"{int(lo*100)}to{int(hi*100)}"
                break

    # ANTI signal: strong player (2400+) goes on negative slope
    # winner's wp DROPPING means the loser was climbing... but we want
    # the opposite: a strong player DECLINING. If the WINNER is 2400+
    # and their wp drops, that's the strong player declining.
    if climber_elo >= ELO:
        for hi, lo, _ in ANTI_BANDS:
            if prev >= hi and mover_wp < lo:
                current_game["anti_active"] = True
                current_game["anti_band"] = f"{int(hi*100)}to{int(lo*100)}"
                break

    if current_game["pos_active"]:
        current_game["positions"].append({"fen": fen, "cp": cp, "side": winner,
                                           "band": current_game["pos_band"], "signal": "pos"})
    if current_game["anti_active"]:
        current_game["positions"].append({"fen": fen, "cp": cp, "side": winner,
                                           "band": current_game["anti_band"], "signal": "anti"})

    current_game["prev_wp"] = mover_wp

flush(current_game)
out_pos.close()
out_anti.close()
print(f"games: {n_games:,}")
print(f"positive-signal positions: {n_pos_mined:,}")
print(f"anti-signal positions: {n_anti_mined:,}")
print(f"outputs: {OUT_POS} and {OUT_ANTI}")
