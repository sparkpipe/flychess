"""AJ CORPUS FILTER + DEDUP — master-involved games only + drop games
already in games_all (by game fingerprint: players+date+result+plycount).

Output: /extnvme/newgames/aj_qualified.tsv (same schema as games_all.tsv:
gid offset 10,000,000, offset=len into the combined pgn, meta...).
Also emits a merged PGN of qualifying games for the miner.
Prune rule (operator 2026-10-05): amateur-vs-amateur OUT; master-beats-am,
am-beats-master, master-vs-master all IN. Master = either >= 2400.
"""
import re, io, os, glob

NEW = "/extnvme/newgames"
GAMES = "/mnt/cold-raid6/chess-audit/wp_fit/games_all.tsv"
MASTER = 2200

# existing fingerprints (Lumbras corpus)
have = set()
for line in open(GAMES):
    p = line.rstrip("\n").split("\t")
    if p[0] == "game_id":
        continue
    fp = (p[3].lower(), p[4].lower(), p[5], p[8], p[10])
    have.add(fp)
print(f"existing fingerprints: {len(have):,}")

out_tsv = open(f"{NEW}/aj_qualified.tsv", "w", encoding="utf-8")
out_tsv.write("game_id\toffset\tlength\twhite\tblack\tresult\twelo\tbelo\tdate\tevent\tplycount\treason\n")
out_pgn = open(f"{NEW}/aj_qualified.pgn", "w", encoding="utf-8")
gid = 10_000_000
kept = dup = pruned = 0
for path in sorted(glob.glob(f"{NEW}/x_*/*.pgn")):
    src = os.path.basename(path)
    fin = io.open(path, errors="replace")
    def flush(hdr, movetext):
        global gid, kept, dup, pruned
        if not hdr:
            return
        welo = belo = 0
        try:
            m = re.sub(r"\D", "", hdr.get("WhiteElo", ""))
            welo = int(m) if m else 0
            m = re.sub(r"\D", "", hdr.get("BlackElo", ""))
            belo = int(m) if m else 0
        except ValueError:
            pass
        if welo < MASTER and belo < MASTER:
            pruned += 1
            return
        w = hdr.get("White", "?")
        b = hdr.get("Black", "?")
        res = hdr.get("Result", "?")
        date = hdr.get("Date", "????.??.??")
        ply = hdr.get("PlyCount", "0")
        fp = (w.lower(), b.lower(), res, date, ply)
        if fp in have:
            dup += 1
            return
        have.add(fp)
        gid += 1
        rec_pgn = "".join(f'[{k} "{v}"]\n' for k, v in hdr.items()) + movetext
        off = out_pgn.tell()
        out_pgn.write(rec_pgn + "\n")
        kept += 1
        out_tsv.write("\t".join(str(x) for x in [
            gid, off, len(rec_pgn), w.replace("\t", " "), b.replace("\t", " "),
            res, welo, belo, date,
            hdr.get("Event", "?")[:60], ply, "aj-master"]) + "\n")
    hdr = {}
    movetext = ""
    for line in fin:
        if line.startswith("["):
            m = re.match(r'\[(\w+) "(.*)"\]\s*$', line)
            if m and not movetext:
                hdr[m.group(1)] = m.group(2)
            elif m and movetext:
                flush(hdr, movetext)
                hdr = {m.group(1): m.group(2)}
                movetext = ""
        else:
            movetext += line
    flush(hdr, movetext)
out_pgn.close()
out_tsv.close()
print(f"kept={kept:,} dup={dup:,} pruned-amateur={pruned:,}")
