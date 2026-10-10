"""PROBE — is the fleet->Lumbras join working at all?

1. Reconstruct the first fleet game from combined.txt consecutive lines.
2. Find candidate Lumbras games (same welo/belo/result) in games_all.tsv.
3. Replay each candidate; count board-part hits at matching plies.
"""
import chess, chess.pgn, io

FLEET = "/mnt/cold-raid6/rtx5090-archive/chess-lab/otb_evals/combined.txt"
GAMES = "/mnt/cold-raid6/chess-audit/wp_fit/games_all.tsv"
PGN = "/mnt/cold-raid6/rtx5090-archive/chess-lab/games/LumbrasGigaBase_OTB_Complete.pgn"

game_positions = []
ref = None
n_games = 0
for line in open(FLEET):
    p = line.rstrip("\n").split("|")
    if len(p) < 8 or not p[7]:
        continue
    ply = int(p[3])
    if ply <= 1 and game_positions:
        n_games += 1
        if n_games >= 1:
            ref = game_positions
            break
    if not game_positions or ply > int(game_positions[-1][0]) or p[4] != game_positions[0][4]:
        pass
    if ply == 1:
        game_positions = []
    game_positions.append((ply, p[0], p[2], p[7], p[4], p[5], p[6]))

print(f"ref game: {len(ref)} positions, welo={ref[0][4]} belo={ref[0][5]} result={ref[0][6]}")
ref_boards = {fen.split(' ')[0]: ply for ply, fen, stm, cp, w, b, r in ref}
print(f"first fen: {ref[0][1]}")

cands = []
for line in open(GAMES):
    f_ = line.rstrip("\n").split("\t")
    if f_[0] == "game_id":
        continue
    if f_[6] == ref[0][4] and f_[7] == ref[0][5] and f_[5] == ref[0][6]:
        cands.append(f_)
        if len(cands) >= 12:
            break
print(f"candidates in games_all: {len(cands)}")

pgn_f = open(PGN, "rb")
for c in cands:
    off, ln = int(c[1]), int(c[2])
    pgn_f.seek(off)
    blob = pgn_f.read(max(ln, 1) if ln > 0 else 65536)
    movetext = b""
    for lne in blob.split(b"\n"):
        if not lne.startswith(b"[") and lne.strip():
            movetext += lne + b" "
    g = chess.pgn.read_game(io.StringIO('[Result "%s"]\n\n' % ref[0][6] + movetext.decode('utf-8', 'replace')))
    if g is None:
        print(f"  gid={c[0]} parse fail")
        continue
    hits = total = 0
    node, ply = g, 0
    while not node.is_end() and ply < 60:
        node = node.variations[0]
        ply += 1
        total += 1
        if node.board().fen().split(" ")[0] in ref_boards:
            hits += 1
    print(f"  gid={c[0]} {c[3][:20]} vs {c[4][:20]} {c[8]}: hits {hits}/{total}")
