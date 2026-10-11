import re
txt = open("/mnt/cold-raid6/chess-audit/ladder10/moe13_sf8.pgn").read()
games = txt.split("[Event")[1:]
for i, g in enumerate(games):
    tc = re.search(r"\[TimeControl .([^\"]+)", g).group(1)
    plies = int(re.search(r"\[PlyCount .(\d+)", g).group(1))
    res = re.search(r"\[Result .([^\"]+)", g).group(1)
    wh = re.search(r"\[White .(\w+).", g).group(1)
    bl = re.search(r"\[Black .(\w+).", g).group(1)
    dur = re.search(r"\[GameDuration .([^\"]+)", g)
    print("game", i+1, ":", wh, "vs", bl, "|", res, "|", tc, "|", plies, "plies |", dur.group(1) if dur else "?")
    if plies >= 10:
        mv = re.search(r"\n\n(.*?)(?:\n\n|\Z)", g, re.S)
        print("--- MOVES ---")
        print(mv.group(1)[:2500] if mv else "n/a")
