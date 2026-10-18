#!/usr/bin/env python3
"""Extract the smooth-rise training corpus from the AJ d10 labeling.

Selection = the operator-ruled criteria (2026-10-18, measured on 208K-game
trial as 6.74% of decisive games):
  - decisive result (1-0 / 0-1)
  - EMA(0.1) winner-pov eval rises from game start to wp >= 0.65
  - rise takes >= 10 plies, no raw |dcp| >= 120 swing en route
  - game ENDS still won (final smoothed wp >= 0.65)
  - PGN Termination header != 'time forfeit' (hash-matched to corpus)

Kept material = positions 0..t (t = first ply reaching the bar), inserted
into games.db positions with kind='aj_sr', ref='<hash>:<ply>'.

Two corpus passes: fast regex hash over raw movetext finds candidates;
python-chess replay verifies the canonical hash, applies the termination
check, and emits FENs. Insert is idempotent (fen-unique, INSERT OR IGNORE).

Usage: aj_extract_smoothrise.py --out-dir AJ_KEPT/OUT --corpus CORPUS_DIR
                                 --db games.db [--limit-games N]
"""
import argparse
import glob
import hashlib
import json
import math
import re
import sqlite3
import sys
import time

import chess
import chess.pgn

wp_of = lambda cp: 1.0 / (1.0 + math.exp(-cp / 361.0))

MOVE_NO = re.compile(r"\d+\.(\.\.)?|\{[^}]*\}|\$\d+|[?!]+|1-0|0-1|1/2-1/2|\*")
WS = re.compile(r"\s+")


def fast_hash(movetext):
    """Hash of the SAN join from raw movetext (move numbers/comments stripped)."""
    return hashlib.sha1(WS.sub(" ", MOVE_NO.sub(" ", movetext)).strip()
                        .encode("utf-8", "replace")).hexdigest()[:12]


def select_from_jsonl(out_dir):
    """hash -> end ply t for games meeting the smooth-rise criteria."""
    sel = {}
    dec = read = 0
    for path in sorted(glob.glob(f"{out_dir}/aj.*.jsonl")):
        for line in open(path):
            try:
                g = json.loads(line)
            except json.JSONDecodeError:
                continue
            read += 1
            r = g.get("r")
            if r not in ("1-0", "0-1"):
                continue
            dec += 1
            w = [v if r == "1-0" else -v for v in g["e"]]
            ema, e, sm = [], w[0], []
            for v in w:
                e = 0.9 * e + 0.1 * v
                sm.append(e)
            t = next((i for i, v in enumerate(sm) if wp_of(v) >= 0.65), None)
            if t is None or t < 10:
                continue
            if any(abs(b - c) >= 120 for b, c in zip(w[:t + 1], w[1:t + 1])):
                continue
            if wp_of(sm[-1]) < 0.65:
                continue
            sel[g["h"]] = t
    return sel, read, dec


def iter_corpus(corpus):
    """Yield (headers, movetext) per game without full PGN parsing."""
    for path in sorted(glob.glob(f"{corpus}/*/*.pgn")):
        hdrs, mov = [], []
        with open(path, encoding="utf-8", errors="replace") as f:
            for line in f:
                if line.startswith("["):
                    if mov:
                        yield hdrs, " ".join(mov)
                        hdrs, mov = [], []
                    hdrs.append(line)
                elif line.strip():
                    mov.append(line.strip())
            if mov:
                yield hdrs, " ".join(mov)


def header(hdr_lines, name):
    for l in hdr_lines:
        if l.startswith(f'[{name} "'):
            return l.split('"')[1] if '"' in l else ""
    return ""


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--corpus", required=True)
    ap.add_argument("--db", required=True)
    ap.add_argument("--limit-games", type=int, default=0,
                    help="stop after N selected games (smoke runs)")
    a = ap.parse_args()

    t0 = time.time()
    sel, read, dec = select_from_jsonl(a.out_dir)
    print(f"[{time.strftime('%H:%M:%S')}] jsonl: {read:,} games read, "
          f"{dec:,} decisive, {len(sel):,} meet smooth-rise criteria", flush=True)

    con = sqlite3.connect(a.db, timeout=60)
    con.execute("pragma journal_mode=wal")
    con.execute("pragma synchronous=NORMAL")

    fast_match = verified = forfeit = hash_mismatch = 0
    inserted = games_kept = 0
    for hdrs, mov in iter_corpus(a.corpus):
        h = fast_hash(mov)
        if h not in sel:
            continue
        fast_match += 1
        # rebuild via python-chess for the canonical hash + FENs
        import io
        g = chess.pgn.read_game(io.StringIO("".join(hdrs) + "\n" + mov + "\n"))
        if g is None:
            hash_mismatch += 1
            continue
        sans = []
        b = g.board()
        ok = True
        for mv in g.mainline_moves():
            try:
                sans.append(b.san(mv))
                b.push(mv)
            except Exception:
                ok = False
                break
        if not ok:
            hash_mismatch += 1
            continue
        ch = hashlib.sha1(" ".join(sans).encode()).hexdigest()[:12]
        if ch != h:
            hash_mismatch += 1
            continue
        verified += 1
        if header(hdrs, "Termination").lower().startswith("time"):
            forfeit += 1
            continue
        t = sel[h]
        b = g.board()
        fens = [b.fen()]
        for i, mv in enumerate(g.mainline_moves()):
            if i >= t:
                break
            b.push(mv)
            fens.append(b.fen())
        con.executemany(
            "insert or ignore into positions(fen, kind, ref) values (?,?,?)",
            [(f, "aj_sr", f"{h}:{i}") for i, f in enumerate(fens)])
        inserted += con.total_changes and 0 or 0
        games_kept += 1
        if games_kept % 500 == 0:
            con.commit()
            print(f"  {games_kept:,} games -> {con.total_changes:,} rows", flush=True)
        if a.limit_games and games_kept >= a.limit_games:
            break
    con.commit()
    n = con.execute("select count(*) from positions where kind='aj_sr'").fetchone()[0]
    print(f"[{time.strftime('%H:%M:%S')}] DONE in {time.time()-t0:.0f}s: "
          f"fast-matched {fast_match:,}, verified {verified:,} "
          f"(mismatch {hash_mismatch:,}), forfeit-dropped {forfeit:,}, "
          f"games kept {games_kept:,}, positions now {n:,}", flush=True)


if __name__ == "__main__":
    main()
