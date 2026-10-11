import sys
sys.path.insert(0, "/srv/workspace/flychess/src/chess-lab/tools")
import router12 as R, chess

lines = open("/mnt/cold-raid6/chess-audit/wp_fit/segments/pos_55to70.clean.tsv", errors="replace").readlines()[:2000]
fens = [l.split("|")[0] for l in lines]
py = [R.route23(f) for f in fens]
eng = [l.strip() for l in open("/tmp/engine_routes.txt")]
mism = [(f, a, b) for f, a, b in zip(fens, py, eng) if a != b]
mg = [(f, a, b) for f, a, b in mism if b == "mg_unsafe_king"]
print("mg-boundary: %d of %d total" % (len(mg), len(mism)))
for f, a, b in mg[:4]:
    bb = chess.Board(f)
    us = bb.turn
    ksq = bb.king(us)
    kf, kr = chess.square_file(ksq), chess.square_rank(ksq)
    sh = 0
    own = bb.pieces(chess.PAWN, us)
    for df in (-1, 0, 1):
        for dr in ((1, 2) if us else (-1, -2)):
            s = (kf + df, kr + dr)
            if 0 <= s[0] < 8 and 0 <= s[1] < 8 and chess.square(s[0], s[1]) in own:
                sh += 1
    nsq = 0
    for df in range(-2, 3):
        for dr in range(-2, 3):
            sq = (kf + df, kr + dr)
            if not (0 <= sq[0] < 8 and 0 <= sq[1] < 8):
                continue
            s = chess.square(sq[0], sq[1])
            for att in bb.attackers(not us, s):
                if bb.piece_at(att).piece_type != chess.PAWN:
                    nsq += 1
                    break
    print("  sh=%d nsq=%d unsafe=%s fen=%s" % (sh, nsq, sh <= 1 or (sh <= 2 and nsq >= 2), f[:48]))
