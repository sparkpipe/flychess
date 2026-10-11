"""TRAJECTORY SEGMENT EXTRACTOR: the curated specialist data builder.

Input: 15.7M OTB positions with depth-12 evals
  (fen|played|winner|ply|welo|belo|result|cp|wdl|best|depth)

Process:
  1. Group by game (we track game boundaries via ply resets)
  2. Convert cp to win probability: W = 1/(1+exp(-cp/361))
  3. Detect blunders (eval change > BLUNDER_CP in one move)
  4. Split game into segments at each blunder
  5. For each segment, determine:
     a. Stable material configuration (after early-trade settling)
     b. Win% at segment start and end
     c. Trajectory type: press(55→70), convert(70→win),
        equalize(45→55), defend(30→45)
  6. Tag every position in the segment with (config, trajectory_type)
  7. Output: classified positions ready for per-specialist .bin packing

Usage: python3 segment_extractor.py < input_file > output_file
"""
import sys
import os
import math
import json
import struct
import time
from collections import defaultdict

# thresholds
BLUNDER_CP = 120        # eval swing that starts a new segment
MIN_SEGMENT = 4         # minimum positions for a valid segment
# win% bands (converted from cp using W = 1/(1+exp(-cp/361)))
# 55% ≈ +31cp, 70% ≈ +90cp, 45% ≈ -31cp, 30% ≈ -90cp
BANDS = {
    "press":     (0.55, 0.70),   # small edge to real advantage
    "convert":   (0.70, 0.95),   # winning position to victory
    "equalize":  (0.45, 0.55),   # slightly worse to balanced
    "defend":    (0.30, 0.45),   # clearly worse to fighting
    "win":       (0.95, 1.01),   # technically won
    "collapse":  (0.00, 0.30),   # falling apart
}

import chess


def cp_to_winprob(cp):
    return 1.0 / (1.0 + math.exp(-cp / 361.0))


def piece_config(board):
    """Stable material configuration string, e.g. 'QRvQRR' or 'BvN'."""
    pieces = {"w": [], "b": []}
    for sq in range(64):
        pc = board.piece_at(sq)
        if pc is None:
            continue
        symbol = pc.symbol().upper()
        if symbol == "K":
            continue
        pieces["w" if pc.color == chess.WHITE else "b"].append(symbol)
    # sort by value: Q > R > B > N > P
    order = {"Q": 0, "R": 1, "B": 2, "N": 3, "P": 4}
    w = "".join(sorted(pieces["w"], key=lambda p: order[p]))
    b = "".join(sorted(pieces["b"], key=lambda p: order[p]))
    return f"{w}v{b}" if w or b else "empty"


def bishop_colors(board):
    """Detect opposite-colored bishops."""
    wb = bb = None
    for sq in range(64):
        pc = board.piece_at(sq)
        if pc and pc.piece_type == chess.BISHOP:
            color = (chess.square_file(sq) + chess.square_rank(sq)) % 2
            if pc.color == chess.WHITE:
                wb = color
            else:
                bb = color
    if wb is not None and bb is not None:
        return "opp_bishops" if wb != bb else "same_bishops"
    return ""


def classify_trajectory(start_wp, end_wp):
    """Classify a segment by its probability transition."""
    for name, (lo, hi) in BANDS.items():
        if lo <= start_wp < hi:
            # find the end band
            for end_name, (elo, ehi) in BANDS.items():
                if elo <= end_wp < ehi:
                    if start_wp < end_wp:  # improving
                        return f"{name}_to_{end_name}"
                    else:
                        return f"{end_name}_to_{name}"  # declining
            return f"{name}_to_{name}"  # no change
    return "unknown"


def main():
    t0 = time.time()
    n_in = 0
    n_segments = 0
    segment_counts = defaultdict(int)
    config_counts = defaultdict(int)

    # read all lines, group by game (ply resets indicate game boundaries)
    current_game = []
    game_id = 0

    def process_game(positions, gid):
        nonlocal n_segments
        if len(positions) < MIN_SEGMENT:
            return

        # compute win probs
        wps = []
        for p in positions:
            wp = cp_to_winprob(p["cp"])
            # flip to white perspective for consistency
            if p["winner"] == "b":
                wp = 1.0 - wp
            wps.append(wp)

        # detect blunders and split into segments
        segments = []
        seg_start = 0
        for i in range(1, len(wps)):
            delta = abs(wps[i] - wps[i - 1])
            if delta * 100 > BLUNDER_CP / 361.0 * 100:  # rough blunder
                segments.append((seg_start, i))
                seg_start = i
        segments.append((seg_start, len(wps)))

        # classify each segment
        for start, end in segments:
            if end - start < MIN_SEGMENT:
                continue
            seg_wp_start = wps[start]
            seg_wp_end = wps[end - 1]
            traj = classify_trajectory(seg_wp_start, seg_wp_end)

            # get stable material config from the middle of the segment
            mid = (start + end) // 2
            board = chess.Board(positions[mid]["fen"])
            config = piece_config(board)
            bc = bishop_colors(board)
            if bc:
                config += f"_{bc}"

            n_segments += 1
            segment_counts[traj] += 1
            config_counts[config] += 1

            # output every position in this segment with its tags
            for i in range(start, end):
                p = positions[i]
                print(json.dumps({
                    "gid": gid,
                    "fen": p["fen"],
                    "played": p["played"],
                    "winner": p["winner"],
                    "ply": p["ply"],
                    "cp": p["cp"],
                    "wp": round(wps[i], 4),
                    "config": config,
                    "traj": traj,
                    "seg_start_wp": round(seg_wp_start, 4),
                    "seg_end_wp": round(seg_wp_end, 4),
                }), flush=False)

    for line in sys.stdin:
        parts = line.strip().split("|")
        if len(parts) < 8:
            continue
        fen, played, winner, ply = parts[0], parts[1], parts[2], \
            int(parts[3])
        cp = int(parts[7]) if parts[7] and parts[7] != 'None' else 0

        # game boundary: ply decreases (new game starts)
        if current_game and ply <= current_game[-1]['ply']:
            process_game(current_game, game_id)
            game_id += 1
            current_game = []

        current_game.append({
            "fen": fen, "played": played, "winner": winner,
            "ply": ply, "cp": cp,
        })
        n_in += 1
        if n_in % 1000000 == 0:
            print(f"processed {n_in} positions, "
                  f"{n_segments} segments, "
                  f"{time.time()-t0:.0f}s", file=sys.stderr,
                  flush=True)

    # process last game
    if current_game:
        process_game(current_game, game_id)

    print(f"TOTAL: {n_in} positions, {game_id+1} games, "
          f"{n_segments} segments, {time.time()-t0:.0f}s",
          file=sys.stderr, flush=True)

    # summary to stderr
    print("SEGMENT TYPES:", file=sys.stderr)
    for k, v in sorted(segment_counts.items(), key=lambda x: -x[1]):
        print(f"  {k}: {v}", file=sys.stderr)
    print("TOP CONFIGS:", file=sys.stderr)
    for k, v in sorted(config_counts.items(), key=lambda x: -x[1])[:20]:
        print(f"  {k}: {v}", file=sys.stderr)
    print("SEGMENT-EXTRACT-COMPLETE", file=sys.stderr, flush=True)


if __name__ == "__main__":
    main()
