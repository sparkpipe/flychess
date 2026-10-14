#!/usr/bin/env python3
"""AJ/NIC corpus raw stats: per-archive and deduped-across-all.
Amateur-vs-amateur filter: BOTH players lack an Elo tag -> dropped.
Games with no movetext -> counted separately (useless for evals).
Dedup: sha1 of the stripped movetext (same game anywhere = one game)."""
import glob
import hashlib
import json
import re
import sys
from multiprocessing import Pool

BASE = "/srv/workspace/chess-active/aj-corpus"


def scan(path):
    st = dict(games=0, movetext=0, both_elo=0, one_elo=0, no_elo=0,
              kept_nonamateur=0, plies_kept=0, elo_min=9999, elo_max=0)
    cur = {}
    mov = []
    seen_local = set()
    h = None

    def close_game():
        nonlocal h
        if cur is None:
            return
        st["games"] += 1
        text = " ".join(mov).strip()
        if not text or text in ("*", "1-0", "0-1", "1/2-1/2"):
            return
        st["movetext"] += 1
        we = cur.get("WhiteElo")
        be = cur.get("BlackElo")
        w, b = bool(we), bool(be)
        if w and b:
            st["both_elo"] += 1
        elif w or b:
            st["one_elo"] += 1
        else:
            st["no_elo"] += 1
        if not w and not b:
            return                       # amateur vs amateur -> dropped
        st["kept_nonamateur"] += 1
        try:
            pc = int(cur.get("PlyCount", "0") or 0)
        except ValueError:
            pc = 0
        pc = pc or 2 * len(re.findall(r"\d+\.", text))
        st["plies_kept"] += pc
        for e in (we, be):
            if e:
                try:
                    v = int(e)
                    st["elo_min"] = min(st["elo_min"], v)
                    st["elo_max"] = max(st["elo_max"], v)
                except ValueError:
                    pass
        h = hashlib.sha1(text.encode("utf-8", "replace")).hexdigest()
        seen_local.add((h, pc, bool(w), bool(b)))

    hdr = re.compile(r'\[(\w+)\s+"(.*)"\]')
    for line in open(path, encoding="utf-8", errors="replace"):
        if line.startswith("["):
            if line.startswith("[Event ") and mov is not None and cur:
                # finish previous game at the next header block
                pass
            m = hdr.match(line.strip())
            if m:
                if m.group(1) == "Event" and cur:
                    close_game()
                    cur = {}
                    mov = []
                    h = None
                cur[m.group(1)] = m.group(2)
        else:
            t = line.strip()
            if t:
                mov.append(t)
    close_game()
    st["elo_min"] = 0 if st["elo_min"] == 9999 else st["elo_min"]
    return path, st, seen_local


if __name__ == "__main__":
    files = sorted(glob.glob(f"{BASE}/*/*.pgn"))
    with Pool(min(6, len(files))) as p:
        results = p.map(scan, files)
    seen = set()
    plies_by_h = {}
    for p, s, loc in results:
        for h, pc, w, b in loc:
            if h not in plies_by_h:
                plies_by_h[h] = pc
            seen.add(h)
    uniq_plies = sum(plies_by_h.values())
    print(f"DEDUPED (movetext-hash): unique games {len(seen):,}  unique plies {uniq_plies:,}")
    for p, s, loc in results:
        print(f"{p.split('/')[-1]:28s} games {s['games']:>9,}  with-moves "
              f"{s['movetext']:>9,}  bothElo {s['both_elo']:>9,}  "
              f"oneElo {s['one_elo']:>8,}  noElo {s['no_elo']:>8,}  "
              f"KEPT {s['kept_nonamateur']:>9,}  plies {s['plies_kept']:>11,}")
    all_kept_plies = sum(s["plies_kept"] for _, s, _ in results)
    print(f"\nTOTAL kept plies (pre-dedup): {all_kept_plies:,}")
    print(f"~positions for d12: {all_kept_plies:,}")
