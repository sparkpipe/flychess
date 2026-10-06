"""Audit retrain: correct-domain full stack + 13-eval mini-stack.
One extraction pass over all fleet self-play games; two heads trained."""
import sys, glob, time
sys.path.insert(0, "/srv/workspace/flychess/src/chess-lab/tools")
import numpy as np, torch, torch.nn as nn
import chess.pgn
import rl_iter as RI

R = "/mnt/cold-raid6/chess-audit"
SP = f"{R}/selfplay_rl"

# 1. parse ALL fleet games (iter2 + iter3 dirs)
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
nfiles = len(glob.glob(SP + "/fleet_iter*/*.pgn"))
print(f"parsed {len(fens)} positions from {nfiles} files", flush=True)

# 2. features (FIXED domain labels now include exchanges)
t0 = time.time()
evals = RI.extract_expert_evals(fens)
print(f"evals {time.time()-t0:.0f}s", flush=True)
hces, doms = RI.compute_hce_batch(fens)
import collections
print("domains:", dict(collections.Counter(doms.tolist())), flush=True)

np.savez(f"{SP}/audit_features.npz", evals=evals, hces=hces, doms=doms,
         zs=np.array(zs, dtype=np.float32))

# 3. train both heads
def wp(cp):
    return 1.0 / (1.0 + torch.exp(-cp / 361.0))

def train(model, X_ev, X_hce, X_dom, Y, epochs, lr, name):
    g = torch.Generator().manual_seed(42)
    perm = torch.randperm(len(Y), generator=g)
    n_val = max(500, len(Y) // 10)
    val, tr = perm[:n_val], perm[n_val:]
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-5)
    best, best_sd, stale = 1e9, None, 0
    for ep in range(epochs):
        model.train()
        p = tr[torch.randperm(len(tr))]
        for i in range(0, len(p), 4096):
            idx = p[i:i + 4096]
            loss = ((wp(model(X_ev[idx], X_hce[idx], X_dom[idx])) - Y[idx]) ** 2).mean()
            opt.zero_grad(); loss.backward(); opt.step()
        model.eval()
        with torch.no_grad():
            vl = ((wp(model(X_ev[val], X_hce[val], X_dom[val])) - Y[val]) ** 2).mean().item()
        if vl < best - 1e-6:
            best, best_sd, stale = vl, {k: v.detach().clone() for k, v in model.state_dict().items()}, 0
        else:
            stale += 1
        if stale >= 40:
            print(f"  {name} early stop ep{ep} val={best:.6f}", flush=True)
            break
    model.load_state_dict(best_sd)
    print(f"  {name}: best val {best:.6f}", flush=True)
    return best

E_N, HCE_D = 13, 36
X_ev = torch.tensor(evals)
X_hce = torch.tensor(hces)
X_dom = torch.tensor(doms)
Y = torch.tensor(np.array(zs, dtype=np.float32))

class Full(nn.Module):
    def __init__(self, g0, w0, b0, m, s):
        super().__init__()
        self.gates = nn.Parameter(g0.clone())
        self.lin = nn.Linear(E_N + HCE_D + E_N, 1)
        with torch.no_grad():
            self.lin.weight.copy_(w0[None, :]); self.lin.bias.fill_(b0)
        self.register_buffer("mean", m.clone()); self.register_buffer("std", s.clone())
    def forward(self, ev, hce, dom):
        g = self.gates[dom]
        oh = torch.zeros(len(ev), E_N); oh.scatter_(1, dom[:, None], 1.0)
        return self.lin(torch.cat([(ev - self.mean) / self.std * g, hce, oh], 1)).squeeze(-1)

g0, w0, b0, mean, std = RI.parse_evh(f"{R}/head_evals_v1.evh")
full = Full(torch.from_numpy(g0), torch.from_numpy(w0), b0,
            torch.from_numpy(mean), torch.from_numpy(std))
train(full, X_ev, X_hce, X_dom, Y, 300, 2e-4, "fullstack_fixed")
g = full.gates.detach().numpy(); lw = full.lin.weight.detach().numpy().flatten()
RI.write_evh(f"{SP}/head_fullstack_fixed.evh", g, lw, float(full.lin.bias.item()),
             full.mean.numpy(), full.std.numpy())
torch.save(full.state_dict(), f"{SP}/head_fullstack_fixed.pt")

class Mini(nn.Module):
    def __init__(self, m, s):
        super().__init__()
        self.lin = nn.Linear(E_N, 1)
        nn.init.zeros_(self.lin.weight); nn.init.zeros_(self.lin.bias)
        with torch.no_grad():
            self.lin.weight[0, 0] = 1.0
        self.register_buffer("mean", m.clone()); self.register_buffer("std", s.clone())
    def forward(self, ev, hce, dom):
        return self.lin((ev - self.mean) / self.std).squeeze(-1)

mini = Mini(X_ev.mean(0), X_ev.std(0).clamp(min=1.0))
train(mini, X_ev, X_hce, X_dom, Y, 300, 1e-3, "ministack")
lw13 = mini.lin.weight.detach().numpy().flatten()
full_lw = np.zeros(E_N + HCE_D + E_N, dtype=np.float32); full_lw[:E_N] = lw13
RI.write_evh(f"{SP}/head_ministack.evh", np.ones((13, E_N), dtype=np.float32), full_lw,
             float(mini.lin.bias.item()), mini.mean.numpy(), mini.std.numpy())
torch.save(mini.state_dict(), f"{SP}/head_ministack.pt")
print("mini weights:", dict(zip(RI.EXPERTS, [round(float(w), 3) for w in lw13])), flush=True)
print("DONE", flush=True)
