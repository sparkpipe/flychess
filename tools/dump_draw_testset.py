"""UNSEEN TEST SET: positions from DRAWN OTB games (excluded from all
training data by the decisive-only prefilter — genuinely unseen).
Byte-scan the archive for draw blocks, parse a spread sample (every
Nth matching game), emit positions sampled across plies 10-120.

Output: fen|draw|w|ply|welo|belo|1/2-1/2
Usage: dump_draw_testset.py <archive.pgn> <out.txt> <n_games>
"""
import sys
import random
import chess
import chess.pgn

PGN, OUT, NGAMES = sys.argv[1], sys.argv[2], int(sys.argv[3])
rng = random.Random(20260929)

# pass 1: byte-scan draw blocks (cheap), collect file offsets
offsets = []
with open(PGN, "rb") as f:
    pos = 0
    block_start = 0
    in_block = False
    for line in f:
        if line.startswith(b"[Event "):
            block_start = pos
            in_block = True
        elif in_block and line.startswith(b'[Result "1/2-1/2"]'):
            offsets.append(block_start)
            in_block = False
        pos += len(line)
print("draw blocks found:", len(offsets), flush=True)

# spread sample across the archive order (eras)
stride = max(1, len(offsets) // NGAMES)
picked = offsets[::stride][:NGAMES]
print("picked:", len(picked), flush=True)

n_pos = n_games = 0
with open(PGN, "rb") as f, open(OUT, "w") as w:
    for off in picked:
        f.seek(off)
        # read block until blank-line-after-movetext heuristic: just parse
        # via a bounded reader
        data = f.read(20000)
        import io
        gh = chess.pgn.read_game(io.TextIOWrapper(io.BytesIO(data),
                                                  encoding="utf-8",
                                                  errors="replace"))
        if gh is None:
            continue
        try:
            wel = int(gh.headers.get("WhiteElo", 0) or 0)
            bel = int(gh.headers.get("BlackElo", 0) or 0)
        except Exception:
            continue
        if wel < 2200 or bel < 2200:
            continue
        board = gh.board()
        plies = []
        for node in gh.mainline():
            board.push(node.move)
            plies.append(board.fen())
            if len(plies) > 120:
                break
        if len(plies) < 20:
            continue
        take = list(range(9, len(plies), 4))  # every 4th ply from move 5
        for p in take:
            w.write("%s|draw|w|%d|%d|%d|1/2-1/2\n"
                    % (plies[p], p + 1, wel, bel))
            n_pos += 1
        n_games += 1
        if n_games % 500 == 0:
            print("games %d, positions %d" % (n_games, n_pos), flush=True)
        if n_games >= NGAMES:
            break
print("DRAW TEST SET: %d games -> %d positions" % (n_games, n_pos))
print("DRAW-DUMP-COMPLETE")
