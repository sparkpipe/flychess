#!/usr/bin/env3
"""Watch fastchess and emit a live game stub PGN — only while fastchess
is actually alive and a game is in progress. If fastchess dies, the stub
disappears (the blinking dot stops)."""
import re
import subprocess
import time
import os

LOG = "/srv/workspace/chess-active/matches"
OUT = "/srv/workspace/chess-active/matches/_live_game.pgn"

while True:
    try:
        # only show live if fastchess is actually running
        r = subprocess.run(["pgrep", "-c", "fastchess"],
                           capture_output=True, text=True)
        fc_alive = int(r.stdout.strip() or "0") > 0
        if not fc_alive:
            if os.path.exists(OUT):
                os.remove(OUT)
            time.sleep(5)
            continue

        logs = [f for f in os.listdir(LOG) if f.endswith(".log")
                and not f.startswith("_")]
        if not logs:
            time.sleep(3)
            continue
        logs.sort(key=lambda f: os.path.getmtime(os.path.join(LOG, f)),
                   reverse=True)
        logname = logs[0]
        logpath = os.path.join(LOG, logname)
        text = open(logpath, errors="replace").read()
        starts = re.findall(
            r"Started game (\d+) of \d+ \((\S+) vs (\S+)\)", text)
        if not starts:
            time.sleep(3)
            continue
        n, white, black = starts[-1]
        pgnpath = logpath.replace(".log", ".pgn")
        pgn_games = 0
        if os.path.exists(pgnpath):
            pgn_games = open(pgnpath, errors="replace").read().count("[Event ")
        if int(n) > pgn_games:
            tag = logname.replace(".log", "")
            with open(OUT, "w") as f:
                f.write(f'[Event "{tag}"]\n')
                f.write(f'[White "{white}"]\n')
                f.write(f'[Black "{black}"]\n')
                f.write('[Result "*"]\n')
                f.write('[PlyCount "0"]\n\n')
                f.write(" *\n\n")
        elif os.path.exists(OUT):
            os.remove(OUT)
    except Exception:
        pass
    time.sleep(3)
