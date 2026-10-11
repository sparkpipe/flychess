"""Probe: does the reader respond to the witness at all? Compute
logits for one puzzle with real witness vs zeroed witness; compare.
Also verify reader.pt vs a freshly-initialized reader."""
import sys
import os
import json

os.environ.setdefault("ANORM", "1")
sys.path.insert(0, "/home/spec/chess-lab")
sys.path.insert(0, "/home/spec/chess-lab/tools")
import chess
import torch
import numpy as np
import fly_curriculum as fc
from mof_reader import Reader

OUTD = "/home/spec/chess-lab/singles/reader"
res = json.load(open(f"{OUTD}/lichess_eval.json"))
fin = json.load(open(f"{OUTD}/final.json"))
print("final.json:", json.dumps(fin))

import chess.engine  # noqa
p0 = res[0]
# rebuild the witness for puzzle 0 by rerunning its stored top5 data is
# not enough — instead read Wl from the eval script's saved... it isn't
# saved; so recompute: load closure + packs via the eval module
import mof_lichess_eval as ME
import flyfeat_cb
flyfeat_cb.feat_vec(chess.Board())
puzzles = [json.loads(l)
           for l in open("/home/spec/chess-lab/tbpools/lichess_endgame_eval.jsonl")][:1]
boards = [chess.Board(p["fen"]) for p in puzzles]
packs = [ME.pack_board(b) for b in boards]
Ls = [len(p[1]) for p, _ in packs]

z = np.load("/home/spec/chess-lab/singles/closure/deltas_2.npz")
state = json.load(open("/home/spec/chess-lab/singles/closure/models_2.json"))
base = ME.tp.build_model(0)
base_state = {k: p.detach().clone() for k, p in base.named_parameters()}
base_cat = torch.cat([base_state[k].flatten() for k, _ in base.named_parameters()])
n_fly = 138
Wl = np.zeros((n_fly, 1, 64), np.float16)
with torch.no_grad():
    for fi in range(n_fly):
        f = {"idx": z[f"m{fi:05d}_idx"], "val": z[f"m{fi:05d}_val"]}
        for k, p in base.named_parameters():
            p.copy_(base_state[k])
        flat = base_cat.clone()
        flat.index_add_(0, torch.from_numpy(f["idx"]).to(ME.fc.DEV),
                        torch.from_numpy(f["val"]).to(ME.fc.DEV))
        off = 0
        for _, p in base.named_parameters():
            n_ = p.numel()
            p.copy_(flat[off:off + n_].view_as(p))
            off += n_
        arrays, mvs = packs[0]
        L = len(mvs)
        Tt, _, _, _ = ME.fc.forward(base, arrays[0], arrays[1],
                                    arrays[2], arrays[3], arrays[4],
                                    arrays[5], arrays[6], arrays[7])
        Wl[fi, 0, :L] = Tt[0, :L].detach().cpu().numpy().astype(np.float16)
print("witness std:", float(Wl.std()))

model = Reader(n_fly).to(ME.fc.DEV)
model.load_state_dict(torch.load(f"{OUTD}/reader.pt",
                                 map_location=ME.fc.DEV))
model.eval()
L = Ls[0]
toks_real = np.zeros((1, 64, 17 + n_fly), np.float32)
toks_real[0, :L, :17] = packs[0][0][3][0, :L]
toks_real[0, :L, 17:] = Wl[:n_fly, 0, :L].T
toks_zero = toks_real.copy()
toks_zero[0, :L, 17:] = 0.0
xb = ME.Xl if False else None
# position features via pack
xb = packs[0][0][0]
mask = np.zeros((1, 64), bool)
mask[0, :L] = True
with torch.no_grad():
    lg_real = model(torch.from_numpy(toks_real).to(ME.fc.DEV),
                    torch.from_numpy(xb).to(ME.fc.DEV),
                    torch.from_numpy(mask).to(ME.fc.DEV))[0, :L].cpu().numpy()
    lg_zero = model(torch.from_numpy(toks_zero).to(ME.fc.DEV),
                    torch.from_numpy(xb).to(ME.fc.DEV),
                    torch.from_numpy(mask).to(ME.fc.DEV))[0, :L].cpu().numpy()
mus = [m.uci() for m in boards[0].legal_moves]
print("argmax real:", mus[int(np.argmax(lg_real))],
      " zero:", mus[int(np.argmax(lg_zero))])
print("logit diff max:", float(np.abs(lg_real - lg_zero).max()))
print("solution:", puzzles[0]["solution"])
