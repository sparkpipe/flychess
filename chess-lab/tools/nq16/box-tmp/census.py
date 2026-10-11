import re, io

draws = decisive = 0
for fn in ("/extnvme/newgames/x_AJ-OTB-PGN-000/AJ-OTB-PGN-000.pgn",
           "/extnvme/newgames/x_AJ-OTB-PGN-001/AJ-OTB-PGN-001.pgn",
           "/extnvme/newgames/x_AJ-CORR-PGN-000/AJ-CORR-PGN-000.pgn",
           "/extnvme/newgames/x_AJ-CORR-PGN-001/AJ-CORR-PGN-001.pgn",
           "/extnvme/newgames/x_nic-magazine-pgn/nic-magazine-pgn.pgn"):
    n2 = q2 = 0
    welo = belo = 0
    res = "?"
    for line in io.open(fn, errors="replace"):
        if line.startswith("[WhiteElo"):
            m = re.sub(r"\D", "", line)
            welo = int(m) if m else 0
        elif line.startswith("[BlackElo"):
            m = re.sub(r"\D", "", line)
            belo = int(m) if m else 0
        elif line.startswith("[Result"):
            res = line.split('"')[1]
        elif line.strip() and not line.startswith("["):
            n2 += 1
            if welo >= 2400 or belo >= 2400:
                q2 += 1
                if res == "1/2-1/2":
                    draws += 1
                elif res in ("1-0", "0-1"):
                    decisive += 1
            welo = belo = 0
            res = "?"
    print(f"{fn.rsplit('/',1)[-1]}: games={n2:,} either>=2400={q2:,}")
print(f"TOTAL qualifying: draws={draws:,} decisive={decisive:,} sum={draws+decisive:,}")
