"""ASSEMBLE16 — nQ.16B training bins. All sources, routerB, 16 experts.

Stages (idempotent, .done markers in /extnvme/active/train16/):
  1 puzzles   ~/backlog_shards/puzzle_eval/w*.tsv + lichess solutions CSV
  2 degm      degm.tsv (cp/100) + tb.tsv (wdl -> wp-inversion cp)
  3 segments  pos_*.clean/draws.tsv (sig=pos) + played-move replay
  4 miniatures miniature_rows.tsv + miniature_labels.tsv (d12 cp)

Output: /extnvme/active/train16/<expert>.{train,val}.bin  (40-byte nodchip)
"""
import sys, os, csv, io, glob, struct, json, hashlib
import chess, chess.pgn
sys.path.insert(0, "/srv/workspace/flychess/src/chess-lab/tools")
sys.path.insert(0, "/tmp")
from pack_expert_bins import pack_sfen, pack_move
from routerB import routeB

OUT = "/extnvme/active/anti16"
SEG = "/extnvme/segments"
GAMES = "/mnt/cold-raid6/chess-audit/wp_fit/games_all.tsv"
PGN = "/mnt/cold-raid6/rtx5090-archive/chess-lab/games/LumbrasGigaBase_OTB_Complete.pgn"
SHARDS = os.path.expanduser("~/backlog_shards/puzzle_eval")
CSV = "/srv/workspace/flychess/src/chess-lab/puzzles/lichess_db_puzzle.csv"
DEGM = "/mnt/cold-raid6/chess-audit/wp_fit/trainsets/sources/degm.tsv"
TBTSV = "/mnt/cold-raid6/chess-audit/wp_fit/trainsets/sources/tb.tsv"
MINI = "/extnvme/active/miniature_rows.tsv"
MINILBL = "/extnvme/active/miniature_labels.tsv"
WPM = json.load(open("/mnt/cold-raid6/chess-audit/wp_model_d12.json"))

os.makedirs(OUT, exist_ok=True)
for st in ("puzzles", "static", "miniatures"):
    open(f"{OUT}/.done_{st}", "w").write("pre")
handles = {}
counts = {}

def fh(e, tag):
    k = (e, tag)
    if k not in handles:
        handles[k] = open(f"{OUT}/{e}.{tag}.bin", "ab")
        counts[k] = 0
    return handles[k]

def emit(board, cp, move, key):
    if not move or move not in board.legal_moves:
        return False
    e = routeB(board.fen())
    tag = "val" if (hash(key) % 20 == 0) else "train"
    fh(e, tag).write(struct.pack("<32shHHbB", pack_sfen(board),
                                 max(-30000, min(30000, int(cp))),
                                 pack_move(move), board.fullmove_number, 0, 0))
    counts[(e, tag)] = counts.get((e, tag), 0) + 1
    return True

def done(stage):
    return os.path.exists(f"{OUT}/.done_{stage}")

def mark(stage):
    open(f"{OUT}/.done_{stage}", "w").write("x")
    for h in handles.values():
        h.flush()
    print(f"[stage {stage}] done. counts so far:", flush=True)
    for k in sorted(counts):
        print(f"  {k[0]:18s} {k[1]:5s} {counts[k]:,}", flush=True)

# ---------- stage 1: puzzles ----------
if not done("puzzles"):
    sol = {}
    with open(CSV, encoding="utf-8", errors="replace") as f:
        rd = csv.reader(f)
        for row in rd:
            if len(row) >= 3:
                sol[row[0]] = row[2].split(" ")
    print(f"[puzzles] solutions: {len(sol):,}", flush=True)
    n = skip = 0
    for shard in sorted(glob.glob(f"{SHARDS}/w*.tsv")):
        for line in open(shard, encoding="utf-8", errors="replace"):
            p = line.rstrip("\n").split("\t")
            if len(p) < 5:
                continue
            fen, pid, depth, cp = p[0], p[1], int(p[2]), int(p[4])
            moves = sol.get(pid)
            if not moves or depth >= len(moves):
                skip += 1
                continue
            try:
                board = chess.Board(fen)
                mv = chess.Move.from_uci(moves[depth])
            except Exception:
                skip += 1
                continue
            if not emit(board, cp, mv, pid):
                skip += 1
                continue
            n += 1
            if n % 1000000 == 0:
                print(f"[puzzles] {n:,} packed", flush=True)
    print(f"[puzzles] packed {n:,} skipped {skip:,}", flush=True)
    mark("puzzles")

# ---------- stage 2: degm + syzygy ----------
def win_prob(cp, mat):
    m = min(max(mat, 17), 78) / 58.0
    a = -142.72052667 + m * (372.35176398 + m * (-340.71073572 + m * 415.23490212))
    b = 5.93832785 + m * (15.61267078 + m * (-30.57816876 + m * 69.63866711))
    return 1.0 / (1.0 + pow(10.0, -cp / (400.0 * b / (a + b) / 0.1817)))

def cp_for_win(mat, target=0.90):
    lo, hi = 0.0, 2000.0
    for _ in range(40):
        mid = (lo + hi) / 2
        if win_prob(mid, mat) < target:
            lo = mid
        else:
            hi = mid
    return int((lo + hi) / 2)

if not done("static"):
    n = 0
    for line in open(DEGM, encoding="utf-8", errors="replace"):
        p = line.rstrip("\n").split("|")
        if len(p) < 4:
            continue
        try:
            board = chess.Board(p[0])
            mv = next(iter(board.legal_moves))
        except Exception:
            continue
        emit(board, int(p[2]) / 100.0, mv, ("degm", n))
        n += 1
    print(f"[static] degm {n:,}", flush=True)
    n = 0
    for line in open(TBTSV):
        p = line.rstrip("\n").split("|")
        fen, wdl = p[0], int(p[2])
        try:
            board = chess.Board(fen)
            mv = next(iter(board.legal_moves))
        except Exception:
            continue
        mat = sum({"P":1,"N":3,"B":3,"R":5,"Q":9}[pc.symbol().upper()]
                  for pc in board.piece_map().values() if pc.symbol().upper() != "K")
        cp = cp_for_win(mat) if wdl > 0 else (-cp_for_win(mat) if wdl < 0 else 0)
        emit(board, cp, mv, ("tb", n))
        n += 1
    print(f"[static] syzygy {n:,}", flush=True)
    mark("static")

# ---------- stage 3: segments with played-move replay ----------
def replay_moves(gids_needed):
    games = {}
    for line in open(GAMES):
        f_ = line.rstrip("\n").split("\t")
        if f_[0] == "game_id":
            continue
        gid = int(f_[0])
        if gid in gids_needed:
            games[gid] = (int(f_[1]), int(f_[2]), f_[5])
    pgn_f = open(PGN, "rb")
    moves = {}
    for gid, (off, ln, res) in games.items():
        pgn_f.seek(off)
        blob = pgn_f.read(ln if ln > 0 else 65536)
        movetext = b""
        for lne in blob.split(b"\n"):
            if not lne.startswith(b"[") and lne.strip():
                movetext += lne + b" "
        if not movetext.strip():
            continue
        try:
            g = chess.pgn.read_game(io.StringIO(
                '[Result "%s"]\n\n' % res + movetext.decode("utf-8", "replace")))
        except Exception:
            continue
        if g is None:
            continue
        b = g.board()
        for ply, mv in enumerate(g.mainline_moves(), start=1):
            moves[(gid, ply)] = mv.uci()
        if len(moves) % 200000 == 0:
            print(f"[segments] replayed ~{len(moves):,} plies", flush=True)
    return moves

if not done("segments"):
    rows = []
    for fn in sorted(os.listdir(SEG)):
        if not (fn.startswith("pos_") and (fn.endswith(".clean.tsv") or fn.endswith(".draws.tsv"))):
            continue
        for line in open(f"{SEG}/{fn}", errors="ignore"):
            p = line.rstrip("\n").split("|")
            if len(p) < 8 or p[3] != "anti":
                continue
            rows.append((int(p[7]), int(p[6]), p[0], int(p[1]), p[5]))
    print(f"[segments] rows: {len(rows):,}", flush=True)
    moves = replay_moves({r[0] for r in rows})
    print(f"[segments] replay plies: {len(moves):,}", flush=True)
    n = skip = 0
    for gid, ply, fen, cp, seg_id in rows:
        mv = moves.get((gid, ply + 1))
        if mv is None:
            skip += 1
            continue
        try:
            board = chess.Board(fen)
            m = chess.Move.from_uci(mv)
        except Exception:
            skip += 1
            continue
        if emit(board, cp, m, seg_id):
            n += 1
    print(f"[segments] packed {n:,} skipped {skip:,}", flush=True)
    mark("segments")

# ---------- stage 4: miniatures ----------
if not done("miniatures") and os.path.exists(MINILBL):
    labels = {}
    for line in open(MINILBL):
        h, cp = line.rstrip("\n").split("\t")
        labels[h] = int(cp)
    print(f"[miniatures] labels: {len(labels):,}", flush=True)
    n = skip = 0
    for line in open(MINI):
        fen, uci, gid, ply = line.rstrip("\n").split("\t")
        cp = labels.get(fen)
        if cp is None:
            skip += 1
            continue
        try:
            board = chess.Board(fen)
            mv = chess.Move.from_uci(uci)
        except Exception:
            skip += 1
            continue
        if emit(board, cp, mv, ("mini", gid)):
            n += 1
    print(f"[miniatures] packed {n:,} skipped {skip:,}", flush=True)
    mark("miniatures")
elif not done("miniatures"):
    print("[miniatures] labels not ready; stage skipped (rerun after labeling)", flush=True)

for h in handles.values():
    h.close()
print("ASSEMBLE16 COMPLETE for available stages", flush=True)
