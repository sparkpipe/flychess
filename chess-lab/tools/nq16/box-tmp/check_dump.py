lines = []
for i, line in enumerate(open("/mnt/cold-raid6/chess-audit/otb_complete_dump_both.txt")):
    lines.append(line.strip().split("|"))
    if i >= 40:
        break
game = []
prev_ply = 0
for p in lines:
    ply = int(p[3])
    if ply <= prev_ply and game:
        break
    game.append(p)
    prev_ply = ply
print(f"first game: {len(game)} positions, plies {game[0][3]} to {game[-1][3]}")
stms = ["w" if " w " in p[0] else "b" for p in game]
print(f"stm sequence: {stms}")
print(f"result: {game[0][6]}")
print(f"welo/belo: {game[0][4]}/{game[0][5]}")
# count games and postal (correspondence) proxies
