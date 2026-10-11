"""Train fullstack heads on round-2 (nets2) and round-3 (nets3) evals.
Same games, same fixed domain labels, per-netset eval extraction + calibration."""
import sys, glob, time
sys.path.insert(0, "/srv/workspace/flychess/src/chess-lab/tools")
import numpy as np, torch, torch.nn as nn
import chess.pgn
import rl_iter as RI

R = "/mnt/cold-raid6/chess-audit"
SP = f"{R}/selfplay_rl"

fens, zs = [], []
for gd in [f"{SP}/fleet_iter2", f"{SP}/fleet_iter3"]:
    for pgn_f in sorted(glob.glob(gd + "/*.pgn")):
        with open(pgn_f) as f:
            while True:
                g = chess.pgn.read_game(f)
                if g is None:
                    break
                res = g.headers.get("Result", "*")
                ow = 1.0 if res == "1-0" else 0.0 if res == "0-1" else 0.5
                b = g.board(); ply = 0
                for node in g.mainline():
                    if ply % 4 == 0:
                        fens.append(b.fen())
                        zs.append(ow if b.turn == chess.WHITE else 1.0 - ow)
                    b.push(node.move); ply += 1
print(f"{len(fens)} positions", flush=True)
hces, doms = RI.compute_hce_batch(fens)
Y = torch.tensor(np.array(zs, dtype=np.float32))
X_hce = torch.tensor(hces); X_dom = torch.tensor(doms)

# patch the extractor's net dir per round
orig = RI.R

def extract_with(nets_dir):
    import importlib
    src = open(f"{orig.replace('/mnt/cold-raid6/chess-audit','')}" if False else "/srv/workspace/flychess/src/chess-lab/tools/rl_iter.py").read()
    # extract_expert_evals reads EXPERTS via E.load_expert(name) — it loads from
    # a fixed path inside extract_stack_features; simplest: copy nets into a temp
    # layout? No — load_expert resolves via its own NETS dir. Instead re-point
    # by monkeypatching RI.R-equivalent path used in E.load_expert.
    import extract_stack_features as E
    return E, None

import extract_stack_features as E
import inspect
# find how load_expert builds its path
src = inspect.getsource(E.load_expert)
print(src[:400], flush=True)
