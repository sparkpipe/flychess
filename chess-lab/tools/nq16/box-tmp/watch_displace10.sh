#!/bin/bash
for i in $(seq 1 120); do
  sleep 180
  A=$(grep -c "Finished game" /mnt/cold-raid6/chess-audit/displace/fullfix_champ_s10.log 2>/dev/null)
  B=$(grep -c "Finished game" /mnt/cold-raid6/chess-audit/displace/mini_champ_s10.log 2>/dev/null)
  C=$(grep -c "Finished game" /mnt/cold-raid6/chess-audit/champ_repro/quarter_vs_sf8_s10.log 2>/dev/null)
  if [ "${A:-0}" -ge 6 ] && [ "${B:-0}" -ge 6 ] && [ "${C:-0}" -ge 6 ]; then break; fi
done
echo "=== games done: fullfix $A/6, mini $B/6, sf8leg $C/6 ==="
python3 - << "PYEOF"
import re, glob
for p in sorted(glob.glob("/mnt/cold-raid6/chess-audit/displace/*_s10.pgn")) + sorted(glob.glob("/mnt/cold-raid6/chess-audit/champ_repro/*.pgn")):
    txt = open(p).read()
    rows = []
    for g in txt.split("[Event")[1:]:
        wh = re.search(r"\[White .([^\"]+)", g).group(1)
        res = re.search(r"\[Result .([^\"]+)", g).group(1)
        plies = int(re.search(r"\[PlyCount .(\d+)", g).group(1))
        first = "fullfix" if "fullfix" in p else ("mini" if "mini" in p else "quarter_moe13")
        sc = (1.0 if res == "1-0" else 0.0 if res == "0-1" else 0.5) if wh == first else (0.0 if res == "1-0" else 1.0 if res == "0-1" else 0.5)
        rows.append((plies, sc))
    real = [r for r in rows if r[0] >= 10]
    print(p.split("/")[-1], ":", first, sum(r[1] for r in rows), "/", len(rows),
          " real-games:", sum(r[1] for r in real), "/", len(real),
          " forfeits:", len(rows) - len(real))
PYEOF
