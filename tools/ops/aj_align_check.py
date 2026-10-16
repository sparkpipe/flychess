#!/usr/bin/env python3
"""Verify/repair an AJ out/*.jsonl against its shard's game order.

Each output line's "h" must equal the sha1-12 of the SAN join of the next
kept game in the shard (kept = Elo header present, parseable mainline,
>=3 plies — the worker's eval-free filters). The longest matching prefix
is the trustworthy part; anything after a mismatch/duplicate is the
product of a racing double-writer and is dropped by --fix.

Usage: aj_align_check.py <shard.pgn> <out.jsonl> [--fix]"""
import argparse
import hashlib
import json
import sys

import chess
import chess.pgn


def shard_hashes(path):
    hs = []
    with open(path, encoding="utf-8", errors="replace") as f:
        while True:
            g = chess.pgn.read_game(f)
            if g is None:
                return hs
            h = g.headers
            if "WhiteElo" not in h and "BlackElo" not in h:
                continue
            try:
                sans = []
                b = g.board()
                for mv in g.mainline_moves():
                    sans.append(b.san(mv))
                    b.push(mv)
            except Exception:
                continue
            if len(sans) < 2:
                continue
            hs.append(hashlib.sha1(" ".join(sans).encode()).hexdigest()[:12])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("shard")
    ap.add_argument("out")
    ap.add_argument("--fix", action="store_true")
    a = ap.parse_args()
    hs = shard_hashes(a.shard)
    lines = [l for l in open(a.out) if l.strip()]
    good = 0
    for i, l in enumerate(lines):
        want = hs[i] if i < len(hs) else None
        try:
            got = json.loads(l).get("h")
        except json.JSONDecodeError:
            got = None
        if want is None or got != want:
            break
        good = i + 1
    print(f"{a.out}: {len(lines)} lines, aligned prefix {good}, "
          f"shard kept-games {len(hs)}")
    if good < len(lines):
        if a.fix:
            with open(a.out, "w") as f:
                f.writelines(lines[:good])
            print(f"  TRUNCATED to aligned prefix ({len(lines) - good} "
                  f"lines dropped — racing-writer debris)")
            sys.exit(2)
        print("  MISMATCH beyond prefix — rerun with --fix to truncate")
        sys.exit(1)


if __name__ == "__main__":
    main()
