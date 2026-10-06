"""Generate all 960 Chess960 opening positions as an EPD book.
Removes book memorization — forces the head to learn general chess principles."""
import chess
import sys

OUT = sys.argv[1] if len(sys.argv) > 1 else "/mnt/cold-raid6/chess-audit/chess960_book.epd"

def generate_all_960():
    """Generate all 960 valid Chess960 starting positions."""
    positions = []
    for pos_num in range(960):
        # Chess960 back rank: generate from position number
        # Standard algorithm: place pieces according to Scharnagl numbering
        rank = [""] * 8
        n = pos_num

        # Place bishops (must be on opposite colors)
        bishop_dark = n % 4
        n //= 4
        bishop_light = n % 4
        n //= 4
        # Dark squares on rank 1: b(1), d(3), f(5), h(7)
        dark_squares = [1, 3, 5, 7]
        light_squares = [0, 2, 4, 6]
        # Place queen
        queen_pos = n % 6
        n //= 6
        # Place knight (2 remaining spots after queen)
        knight1 = n % 5
        n //= 5
        knight2 = n % 4

        # Build the rank
        # First place bishops
        occupied = set()
        rank[dark_squares[bishop_dark]] = "b"
        occupied.add(dark_squares[bishop_dark])
        rank[light_squares[bishop_light]] = "b"
        occupied.add(light_squares[bishop_light])

        # Place queen in first available spot
        remaining = [i for i in range(8) if i not in occupied]
        q_sq = remaining[queen_pos]
        rank[q_sq] = "q"
        occupied.add(q_sq)

        # Place knights
        remaining = [i for i in range(8) if i not in occupied]
        n1_sq = remaining[knight1]
        rank[n1_sq] = "n"
        occupied.add(n1_sq)
        remaining = [i for i in range(8) if i not in occupied]
        n2_sq = remaining[knight2]
        rank[n2_sq] = "n"
        occupied.add(n2_sq)

        # Remaining 3 squares: R, K, R (king between rooks)
        remaining = [i for i in range(8) if i not in occupied]
        rank[remaining[0]] = "r"
        rank[remaining[1]] = "k"
        rank[remaining[2]] = "r"

        # Build FEN
        back_rank = "".join(rank)
        # Mirror for Black (uppercase = white pieces, lowercase for white FEN)
        # White pieces are uppercase in FEN
        wr = back_rank.upper()
        # Black pieces (uppercase in rank 8 FEN)
        br = back_rank.upper()

        fen = f"{br}/pppppppp/8/8/8/8/PPPPPPPP/{wr} w KQkq - 0 1"

        # Validate
        board = chess.Board(fen)
        if not board.is_valid():
            continue

        positions.append((pos_num, fen))

    return positions


def main():
    positions = generate_all_960()
    print(f"Generated {len(positions)} Chess960 positions")
    with open(OUT, "w") as f:
        for num, fen in positions:
            board = chess.Board(fen)
            f.write(f"{board.epd()} bm -;\n")
    print(f"Saved to {OUT}")


if __name__ == "__main__":
    main()
