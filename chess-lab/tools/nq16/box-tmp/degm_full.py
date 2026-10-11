"""DEGM FULL EXTRACTION — every position in the book, no filtering.

Traverses ALL variations of every game/line in DEGM.pgn, emits every
position reached (unique by fen), both sides to move as they occur.
Output: degm_full.tsv (fen) — routed downstream.
"""
import chess, chess.pgn, io, sys

PGN = "/srv/workspace/flychess/src/chess-lab/books/DEGM.pgn"
OUT = "/mnt/cold-raid6/chess-audit/wp_fit/trainsets/sources/degm_full.tsv"

def walk(node, seen, out, stats):
    for var in node.variations:
        b = var.board()
        fen = b.fen()
        stats["total"] += 1
        if fen not in seen:
            seen.add(fen)
            out.write(fen + "\n")
            stats["unique"] += 1
        walk(var, seen, out, stats)

seen = set()
stats = {"total": 0, "unique": 0, "games": 0, "parse_errors": 0}
out = open(OUT, "w")
with open(PGN, encoding="utf-8", errors="replace") as f:
    while True:
        try:
            g = chess.pgn.read_game(f)
        except Exception:
            stats["parse_errors"] += 1
            # resync: skip to next blank line after movetext
            try:
                while True:
                    ln = f.readline()
                    if not ln:
                        break
            except Exception:
                break
            continue
        if g is None:
            break
        stats["games"] += 1
        try:
            root = g.board()
        except Exception:
            # malformed doubled-FEN header: repair by taking the first valid fen substring
            import re
            hdr = g.headers.get("FEN", "")
            m = re.match(r"([pnbrqkPNBRQK1-8]+(?:/[pnbrqkPNBRQK1-8]+){7} [wb] [KQkq-]+ - \d+ \d+)", hdr)
            if not m:
                stats["parse_errors"] += 1
                continue
            try:
                root = chess.Board(m.group(1))
            except Exception:
                stats["parse_errors"] += 1
                continue
            stats["repaired"] = stats.get("repaired", 0) + 1
        root_fen = root.fen()
        stats["total"] += 1
        if root_fen not in seen:
            seen.add(root_fen)
            out.write(root_fen + "\n")
            stats["unique"] += 1
        walk(g, seen, out, stats)
out.close()
print(f"games={stats['games']:,} positions_traversed={stats['total']:,} "
      f"unique={stats['unique']:,} parse_errors={stats['parse_errors']}")
