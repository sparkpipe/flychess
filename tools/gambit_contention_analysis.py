"""Contention analysis over all 94,395 matched gambit games (READ-ONLY).

Answers the operator's questions:
  1. Per-expert-category ply counts across each game's arc (move 1 -> the
     ply where winner wp first reaches 70% after the trough, within the
     <=50-ply scan data).
  2. How many games the WINNER is black (via firstpos lookup in the
     both-sides main dump).
  3. How many games are already in the main dataset (same lookup; hist
     games not found = not in main dump).

Inputs (RAID intermediates): games.json, games_hist.json, fens.txt,
fens_hist.txt, evals.txt, evals_hist.txt; both-sides main dump.
"""
import sys
import json
import chess
from collections import defaultdict

BASE = "/mnt/cold-raid6/rtx5090-archive/chess-lab/gambit-intermediates"
BOTH = "/mnt/cold-raid6/chess-audit/otb_complete_dump_both.txt"

matched = {}
for sfx, label in (("", "elo"), ("_hist", "hist")):
    for gm in json.load(open(f"{BASE}/games{sfx}.json")):
        matched[(sfx, gm["gid"])] = {"sfx": sfx, "trough_ply": gm["trough_ply"]}
print("matched games:", len(matched), file=sys.stderr)

# pass 1: fens+evals per matched game (both suffixes)
games = {}  # (sfx,gid) -> {ply: (fen, W)}
for sfx in ("", "_hist"):
    with open(f"{BASE}/fens{sfx}.txt") as ff, open(f"{BASE}/evals{sfx}.txt") as fe:
        for fline, eline in zip(ff, fe):
            fp = fline.rstrip("\n").split(" ", 3)
            ep = eline.split()
            if len(fp) < 4 or len(ep) < 3:
                continue
            key = (sfx, fp[0])
            if key not in matched:
                continue
            ply = int(fp[1])
            games.setdefault(key, {})[ply] = (fp[3], float(ep[2]))
print("games with trajectory:", len(games), file=sys.stderr)

# pass 2: winner color + main-dataset membership via firstpos lookup
need = {}
for key, plies in games.items():
    if 1 in plies:
        need[plies[1][0].split(" ")[0]] = key
found = {}
with open(BOTH) as f:
    for line in f:
        boardpart = line.split("|", 1)[0].split(" ")[0]
        k = need.get(boardpart)
        if k is not None and k not in found:
            parts = line.split("|")
            found[k] = parts[2]  # 'w' or 'b'
print("looked up:", len(need), "found:", len(found), file=sys.stderr)


def classify(fen):
    """v1.11 cascade category for one position."""
    b = chess.Board(fen)
    men = len(b.piece_map())
    if men <= 5:
        return "tb"
    cw = {"Q": 0, "R": 0, "B": 0, "N": 0}
    cb = {"Q": 0, "R": 0, "B": 0, "N": 0}
    for s, p in b.piece_map().items():
        u = p.symbol().upper()
        if u in cw:
            (cw if p.color else cb)[u] += 1
    resw = "".join(sorted(sum(([k] * max(0, cw[k] - cb[k]) for k in "QRBN"), [])))
    resb = "".join(sorted(sum(([k] * max(0, cb[k] - cw[k]) for k in "QRBN"), [])))
    kres = tuple(sorted((resw, resb)))
    sym = resw == "" and resb == ""
    RES = {("B", "N"): "NvB", ("N", "R"): "NvR", ("B", "R"): "BvR",
           ("NN", "R"): "Rv2m", ("BN", "R"): "Rv2m", ("BB", "R"): "Rv2m",
           ("Q", "RR"): "Qvmat", ("Q", "RN"): "Qvmat", ("Q", "RB"): "Qvmat"}
    if not sym and kres in RES:
        return RES[kres]
    if sym:
        wbc = [chess.square_file(s) + chess.square_rank(s) for s, p in
               b.piece_map().items() if p.piece_type == chess.BISHOP and p.color]
        bbc = [chess.square_file(s) + chess.square_rank(s) for s, p in
               b.piece_map().items() if p.piece_type == chess.BISHOP and not p.color]
        if wbc and bbc and (sum(x % 2 for x in wbc) * 2 > len(wbc)) != \
                (sum(x % 2 for x in bbc) * 2 > len(bbc)):
            return "oppB"
    if men <= 10:
        return "dvoretsky"
    lock = 0
    for f in range(8):
        wr = [chess.square_rank(s) for s in chess.SquareSet(
            b.pieces(chess.PAWN, chess.WHITE)) if chess.square_file(s) == f]
        br = [chess.square_rank(s) for s in chess.SquareSet(
            b.pieces(chess.PAWN, chess.BLACK)) if chess.square_file(s) == f]
        if wr and br and min(br) - max(wr) == 1 and 2 <= f <= 5:
            lock += 1
    return "bal_l%d" % (lock if lock < 3 else 3)


def one_game(item):
    key, plies = item
    m = matched[key]
    trough = m["trough_ply"]
    # arc end: first ply after trough where winner W >= 0.70 (else data end)
    end = max(plies)
    for p in sorted(plies):
        if p > trough and plies[p][1] >= 0.70:
            end = p
            break
    cats = defaultdict(int)
    for p in sorted(plies):
        if p > end:
            break
        cats[classify(plies[p][0])] += 1
    return key, end, dict(cats)


results = []
_items = list(games.items())
for _i, _it in enumerate(_items):
    results.append(one_game(_it))
    if _i % 10000 == 0:
        print("games %d/%d" % (_i, len(_items)), file=sys.stderr, flush=True)

cat_tot = defaultdict(int)
cat_games = defaultdict(int)
arc_lens = []
end_capped = 0
for key, end, cats in results:
    arc_lens.append(end)
    for c, n in cats.items():
        cat_tot[c] += n
        cat_games[c] += 1
    if end >= max(games[key]):
        end_capped += 1

black = sum(1 for v in found.values() if v == "b")
in_main = sum(1 for k in found)
hist_total = sum(1 for k in games if k[0] == "_hist")

total_plies = sum(cat_tot.values())
print("\n=== ARC (move 1 -> wp 70%%): %d games, %d plies total, "
      "median arc %d, capped-at-data-end %d ==="
      % (len(results), total_plies,
         sorted(arc_lens)[len(arc_lens) // 2], end_capped))
print("\n=== PLIES PER EXPERT CATEGORY ===")
print("%-12s %10s %8s %10s" % ("category", "plies", "%plies", "games"))
for c, n in sorted(cat_tot.items(), key=lambda kv: -kv[1]):
    print("%-12s %10s %7.1f%% %10s"
          % (c, format(n, ","), 100.0 * n / total_plies, format(cat_games[c], ",")))
print("\n=== WINNER COLOR (main-dataset lookup) ===")
print("found in main dump: %d / %d  (black winner: %d, white winner: %d)"
      % (in_main, len(games), black, in_main - black))
print("hist games: %d (not found in main dump unless pre-1930)" % hist_total)
print("NOT in main dump: %d" % (len(games) - in_main))
print("ANALYSIS-COMPLETE")
