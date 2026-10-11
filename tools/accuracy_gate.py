"""ACCURACY GATE — fixed 992-position set, SF-d20 reference, cached features.

First run: builds the cache (reference evals + candidate features, ~15 min).
Every later run: loads checkpoint, one forward pass, appends to gates.jsonl.
Usage: accuracy_gate.py <checkpoint.pt> <label>
       accuracy_gate.py --build          (build the cache once)
"""
import sys, os, json, time
sys.path.insert(0, "/srv/workspace/flychess/src/nnue-pytorch")
sys.path.insert(0, "/srv/workspace/flychess/src/chess-lab/tools")
import numpy as np
import torch
import chess

R = "/mnt/cold-raid6/chess-audit"
SP = f"{R}/selfplay_rl"
CACHE = f"{SP}/gate_cache"
E_N, ACT_D, FT_D = 13, 1024, 86896
MAXF = 160
GATES = f"{SP}/gates.jsonl"
COLLAPSE_CORR = 0.90   # "becoming SF" warning threshold

def wp_np(cp):
    return 1.0 / (1.0 + np.exp(-np.array(cp, dtype=np.float64) / 361.0))

def build_cache():
    import accuracy_eval as AE
    os.makedirs(CACHE, exist_ok=True)
    fens = AE.sample_positions()
    print(f"{len(fens)} positions", flush=True)
    ref = AE.sf20_reference(fens)
    valid = [i for i, c in enumerate(ref) if c is not None]
    vfens = [fens[i] for i in valid]
    evals, acts, doms = AE.eval_candidates(vfens)
    ftlists = AE.rawft_all(vfens)
    hces = np.array([AE.hce_fn(chess.Board(f)) for f in vfens], dtype=np.float32)
    ft = np.full((len(valid), MAXF), -1, dtype=np.int64)
    for k, lst in enumerate(ftlists):
        c = min(len(lst), MAXF)
        ft[k, :c] = lst[:c]
    np.savez(f"{CACHE}/features.npz", fens=np.array(vfens), ref=np.array([ref[i] for i in valid]),
             evals=evals, acts=acts, doms=doms, hces=hces, ft=ft)
    print(f"cache built: {len(valid)} positions", flush=True)

class ClippedReLU(torch.nn.Module):
    def __init__(self, hi=63.0):
        super().__init__()
        self.hi = hi
    def forward(self, x):
        return torch.clamp(x, 0.0, self.hi)

class DenseV4(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.emb = torch.nn.Embedding(FT_D, ACT_D, sparse=True)
        self.lin_ev = torch.nn.Linear(E_N, 1, bias=False)
        self.tail = torch.nn.Sequential(
            torch.nn.Linear(E_N*ACT_D + ACT_D + 36 + E_N, 64), ClippedReLU(),
            torch.nn.Linear(64, 32), ClippedReLU(),
            torch.nn.Linear(32, 1),
        )
        self.register_buffer("ev_mean", torch.zeros(E_N))
        self.register_buffer("ev_std", torch.ones(E_N))
        self.register_buffer("a_mean", torch.zeros(E_N, ACT_D))
        self.register_buffer("a_std", torch.ones(E_N, ACT_D))
    def forward(self, ev, act, hce, oh, ftidx):
        x_ev = (ev - self.ev_mean) / self.ev_std
        x_act = (act.float() - self.a_mean) / self.a_std
        mask = (ftidx >= 0).float().unsqueeze(-1)
        raw = (self.emb(ftidx.clamp(min=0)) * mask).sum(dim=1)
        feats = torch.cat([x_act.view(len(ev), -1), raw, hce, oh], dim=1)
        return self.tail(feats).squeeze(-1) + self.lin_ev(x_ev).squeeze(-1)

def run_gate(ckpt_path, label):
    if not os.path.exists(f"{CACHE}/features.npz"):
        print("no cache — run --build first", flush=True)
        return None
    d = np.load(f"{CACHE}/features.npz")
    ref_wp = wp_np(d["ref"])
    evals, acts, doms = d["evals"], d["acts"], d["doms"]
    hces, ft = d["hces"], d["ft"]
    n = len(d["fens"])
    oh = np.zeros((n, E_N), dtype=np.float32)
    oh[np.arange(n), doms] = 1.0
    stm_sign = np.array([1.0 if chess.Board(f).turn else -1.0 for f in d["fens"]])

    sd = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    m = DenseV4(); m.load_state_dict(sd); m.eval()
    with torch.no_grad():
        pred = m(torch.tensor(evals), torch.from_numpy(acts), torch.tensor(hces),
                 torch.tensor(oh), torch.tensor(ft)).numpy()
    p = wp_np(pred * stm_sign)
    rmse = float(np.sqrt(((p - ref_wp) ** 2).mean()))
    corr = float(np.corrcoef(p, ref_wp)[0, 1])

    # baseline: routed expert (nQ) on the same set
    routed = evals[np.arange(n), doms] * stm_sign
    pr = wp_np(routed)
    nq_rmse = float(np.sqrt(((pr - ref_wp) ** 2).mean()))
    nq_corr = float(np.corrcoef(pr, ref_wp)[0, 1])

    entry = {"t": time.strftime("%m-%d %H:%M"), "label": label,
             "ckpt": ckpt_path, "wp_rmse": round(rmse, 4), "corr": round(corr, 4),
             "nq_rmse": round(nq_rmse, 4), "nq_corr": round(nq_corr, 4),
             "beats_nq_rmse": rmse < nq_rmse}
    if corr > COLLAPSE_CORR:
        entry["WARN"] = "SF-collapse risk: corr > 0.90"
    with open(GATES, "a") as f:
        f.write(json.dumps(entry) + "\n")
    print(f"GATE {label}: rmse {rmse:.4f} corr {corr:.4f} (nQ baseline {nq_rmse:.4f}/{nq_corr:.4f})",
          flush=True)
    return entry

def main():
    if "--build" in sys.argv:
        build_cache()
        return
    if len(sys.argv) < 3:
        print("usage: accuracy_gate.py <ckpt> <label> | --build")
        return
    run_gate(sys.argv[1], sys.argv[2])

if __name__ == "__main__":
    main()
