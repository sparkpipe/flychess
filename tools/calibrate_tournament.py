"""Per-netset calibration via ENGINE-side eval extraction (deployment-faithful).

For each netset (nets, nets2, nets3) and each of 13 experts: load the expert's
.nnue into ALL 13 engine slots, stream the FEN sample, parse
'NNUE evaluation X (side to move, internal units)' = the exact value the
stacked head consumes in-search. Then train one Full-arch head per netset
(identical recipe) and export .evh per candidate.
"""
import sys, glob, re, time, subprocess, random
sys.path.insert(0, "/srv/workspace/flychess/src/chess-lab/tools")
import numpy as np, torch, torch.nn as nn
import chess.pgn
import rl_iter as RI

R = "/mnt/cold-raid6/chess-audit"
SP = f"{R}/selfplay_rl"
FORK = "/srv/workspace/flychess/src/Stockfish/src/stockfish"
EXPERTS = RI.EXPERTS
E_N, HCE_D = 13, 36

# ---- 1. FEN sample + labels (same source data, fixed domains) ----
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
print(f"using FULL dataset: {len(fens)} positions", flush=True)
hces, doms = RI.compute_hce_batch(fens)
np.savez(f"{SP}/calib_sample.npz", fens=np.array(fens), zs=np.array(zs),
         hces=hces, doms=doms)

EVAL_RE = re.compile(r"NNUE evaluation\s+(-?\d+) \(side to move, internal units\)")

def extract_netset(runs_dir, tag, out_npz):
    import extract_stack_features as E
    import data_loader
    t0 = time.time()
    fallbacks = [runs_dir, f"{R}/runs4", f"{R}/runs"]
    nets = []
    for e in EXPERTS:
        got = None
        for rd in fallbacks:
            import glob as _g
            if _g.glob(f"{rd}/{e}/lightning_logs/version_*/checkpoints/last.ckpt"):
                E.RUNS = rd; got = E.load_expert(e)
                if rd != runs_dir:
                    print(f"    {e}: fallback {rd}", flush=True)
                break
        nets.append(got)
    assert all(n is not None for n in nets), "expert checkpoint missing entirely"
    ev = np.zeros((len(fens), E_N), dtype=np.float32)
    B = 512
    for st in range(0, len(fens), B):
        chunk = fens[st:st+B]
        bs = data_loader.get_sparse_batch_from_fens(
            "Full_Threats+PP_3Wide+HalfKAv2_hm", chunk,
            [0]*len(chunk), [1]*len(chunk), [0]*len(chunk))
        us, them, wi, bi, o, sc, pc = bs.contents.get_tensors("cuda")
        for ei, mdl in enumerate(nets):
            import torch
            with torch.no_grad():
                vs = mdl.forward(us, them, wi, bi, pc)
            ev[st:st+len(chunk), ei] = [
                float(v) * mdl.quantization.nnue2score for v in vs]
        data_loader.destroy_sparse_batch(bs)
    print(f"  {tag}: extracted via {runs_dir} (+fallbacks) in {time.time()-t0:.0f}s", flush=True)
    np.savez(out_npz, evals=ev)
    return ev

def wp(cp):
    return 1.0 / (1.0 + torch.exp(-cp / 361.0))

class Full(nn.Module):
    def __init__(self, mean, std):
        super().__init__()
        self.gates = nn.Parameter(sim_gates_13())
        self.lin = nn.Linear(E_N + HCE_D + E_N, 1)
        nn.init.zeros_(self.lin.weight); nn.init.zeros_(self.lin.bias)
        with torch.no_grad():
            for e in range(E_N):
                self.lin.weight[0, e] = 1.0  # each expert passes at unity
        self.register_buffer("mean", mean.clone())
        self.register_buffer("std", std.clone())
    def forward(self, ev, hce, dom):
        g = self.gates[dom]
        oh = torch.zeros(len(ev), E_N); oh.scatter_(1, dom[:, None], 1.0)
        return self.lin(torch.cat([(ev - self.mean) / self.std * g, hce, oh], 1)).squeeze(-1)

FAMILY = {"balanced_l0": "balanced", "balanced_l1": "balanced", "balanced_l2": "balanced",
          "balanced_l3": "balanced", "nvb": "conf", "nvr": "conf", "bvr": "conf",
          "rv2m": "conf", "qvmat": "conf", "oppb": "conf",
          "dvoretsky": "end", "exchanges": "trans", "tactics": "tact"}
ADJ = {("balanced_l0","balanced_l1"),("balanced_l1","balanced_l2"),
       ("balanced_l2","balanced_l3"),("nvb","nvr"),("nvb","bvr"),("nvr","bvr"),
       ("nvr","rv2m"),("bvr","rv2m"),("rv2m","qvmat")}

def _sim(e1, e2):
    if e1 == e2: return 1.0
    s = 0.1
    if FAMILY.get(e1,"") == FAMILY.get(e2,""):
        s += 0.3
        if (e1, e2) in ADJ or (e2, e1) in ADJ: s += 0.2
    if FAMILY.get(e1,"") and FAMILY.get(e1,"") == FAMILY.get(e2,""): s += 0.2
    if e1 == "tactics" or e2 == "tactics": s = max(s, 0.3)
    return min(s, 1.0)

def sim_gates_13():
    g = torch.zeros(13, E_N)
    for d in range(12):  # router slots 0..11 = expert families
        for e in range(E_N):
            g[d, e] = _sim(EXPERTS[d], EXPERTS[e])
    g[12, :] = 0.3; g[12, 12] = 1.0  # tb slot
    return g

def train_and_export(runs_dir, tag):
    ev = extract_netset(runs_dir, tag, f"{SP}/calib_evals_{tag}.npz")
    X_ev = torch.tensor(ev); X_hce = torch.tensor(hces)
    X_dom = torch.tensor(doms); Y = torch.tensor(np.array(zs, dtype=np.float32))
    model = Full(X_ev.mean(0), X_ev.std(0).clamp(min=1.0))
    g = torch.Generator().manual_seed(42)
    perm = torch.randperm(len(Y), generator=g)
    n_val = max(500, len(Y) // 10)
    val, tr = perm[:n_val], perm[n_val:]
    opt = torch.optim.AdamW(model.parameters(), lr=5e-5, weight_decay=1e-5)
    best, best_sd, stale = 1e9, None, 0
    for ep in range(300):
        model.train()
        p = tr[torch.randperm(len(tr))]
        for i in range(0, len(p), 4096):
            k = p[i:i+4096]
            loss = ((wp(model(X_ev[k], X_hce[k], X_dom[k])) - Y[k]) ** 2).mean()
            opt.zero_grad(); loss.backward(); opt.step()
        model.eval()
        with torch.no_grad():
            vl = ((wp(model(X_ev[val], X_hce[val], X_dom[val])) - Y[val]) ** 2).mean().item()
        if vl < best - 1e-6:
            best, best_sd, stale = vl, {k2: v.detach().clone() for k2, v in model.state_dict().items()}, 0
        else:
            stale += 1
        if stale >= 40:
            print(f"  {tag}: early stop ep{ep}", flush=True); break
    model.load_state_dict(best_sd)
    print(f"  {tag}: best val {best:.6f}", flush=True)
    lw = model.lin.weight.detach().numpy().flatten()
    # DEBUG: output distribution on balanced training positions (informational)
    with torch.no_grad():
        out = model(X_ev[:20000], X_hce[:20000], X_dom[:20000]).numpy()
    p95 = float(np.percentile(np.abs(out), 95))
    print(f"  {tag} DEBUG: out mean={out.mean():.1f} std={out.std():.1f} p95|out|={p95:.0f}cp", flush=True)
    ok = 5 < p95 < 400
    print(f"  {tag} RANGE_GATE: {'PASS' if ok else 'FAIL'}", flush=True)
    RI.write_evh(f"{SP}/head_{tag}.evh", model.gates.detach().numpy(), lw,
                 float(model.lin.bias.item()), model.mean.numpy(), model.std.numpy())
    torch.save(model.state_dict(), f"{SP}/head_{tag}.pt")
    return best

def main():
    for runs_dir, tag in [(f"{R}/runs", "sQ"), (f"{R}/runs2", "sR2"), (f"{R}/runs3", "sR3")]:
        print(f"=== {tag} ({runs_dir}) ===", flush=True)
        train_and_export(runs_dir, tag)
    print("ALL HEADS CALIBRATED", flush=True)

if __name__ == "__main__":
    main()
