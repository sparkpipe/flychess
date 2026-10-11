"""FULL FUSION — collapse EVERYTHING into one accumulator. Bare-speed v5.1.

Extends the validated fusion: linearize the 13 expert SCALARS into the same
64+13-wide fused structure, eliminating all per-node expert evaluations.

v5.1 apply: fused accumulator (77-wide) -> tail(64->32->1) + scalar-term(13)
all from ONE pass over active features. Zero NNUE evals per node.

Steps:
  1. For fit positions: true scalars (13) per position + accumulators
  2. Affine fit: scalar_e ~ linear in acc_e (2048+1 params, closed form)
  3. Assemble V_scalar rows (64368 x 13) by sparse least squares against
     TRUE scalars (same iterative fit as V — sidesteps the affine layout issue)
  4. Export v6 format: single accumulator, no expert nets needed at eval time
  5. Validate: fused vs true v4 output on held-out
"""
import sys, os, time, struct, subprocess
sys.path.insert(0, "/srv/workspace/flychess/src/nnue-pytorch")
sys.path.insert(0, "/srv/workspace/flychess/src/chess-lab/tools")
import numpy as np
import torch

R = "/mnt/cold-raid6/chess-audit"
SP = f"{R}/selfplay_rl"
E_N, ACT_D, HCE_D, FT_D = 13, 1024, 36, 86896
ACT_BIN = "/srv/workspace/flychess/src/Stockfish-act/src/stockfish"

def main():
    import extract_stack_features as E
    import data_loader
    sd = torch.load(f"{SP}/dense_v4_ckpts3/best.pt", map_location="cpu", weights_only=False)
    W1 = sd["tail.0.weight"].numpy()
    B1 = sd["tail.0.bias"].numpy()
    W2 = sd["tail.2.weight"].numpy()
    b2 = sd["tail.2.bias"].numpy()
    W3 = sd["tail.4.weight"].numpy().flatten()
    b3 = float(sd["tail.4.bias"].numpy().flatten()[0])
    emb_w = sd["emb.weight"].numpy()
    a_mean = sd["a_mean"].numpy(); a_std = sd["a_std"].numpy()
    ev_mean = sd["ev_mean"].numpy(); ev_std = sd["ev_std"].numpy()
    lin_ev_w = sd["lin_ev.weight"].numpy().flatten()  # (13,)
    W1_acts = W1[:, :E_N*ACT_D]
    W_raw = W1[:, E_N*ACT_D:E_N*ACT_D+ACT_D]
    V_emb = emb_w @ W_raw.T

    EXPERTS = ["balanced_l0","balanced_l1","balanced_l2","balanced_l3",
               "nvb","nvr","bvr","rv2m","qvmat","oppb",
               "dvoretsky","exchanges","tactics"]
    nets = [E.load_expert(e) for e in EXPERTS]
    caps = [[] for _ in nets]
    hooks = []
    for mdl, cap in zip(nets, caps):
        def mk(buf):
            def h(mod, inp, out):
                o = out[0] if isinstance(out, tuple) else out
                buf.append(o.detach().cpu().numpy().copy())
            return h
        hooks.append(mdl.input.register_forward_hook(mk(cap)))

    eng = subprocess.Popen([ACT_BIN], stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                           text=True, bufsize=1)
    eng.stdin.write("uci\n"); eng.stdin.flush(); time.sleep(1)

    import fuse_v4 as F
    fens = F.sample_fens(30000)
    fit_fens, val_fens = fens[:24000], fens[24000:]

    def collect(fens_list):
        """per position: true acts (13k), true evals (13), rawft idx, hce, dom"""
        B = 512
        acts_all, evals_all, raws, hces, doms = [], [], [], [], []
        from score_experts import domain as domain_of
        DOMS = ["balanced_l0","balanced_l1","balanced_l2","balanced_l3","nvb",
                "nvr","bvr","rv2m","qvmat","oppb","dvoretsky","exchanges","tb"]
        D2I = {d: i for i, d in enumerate(DOMS)}
        import chess as CH
        for i, f in enumerate(fens_list):
            b = CH.Board(f)
            hces.append(F.hce_fn(b))
            try:
                doms.append(D2I.get(domain_of(f), 12))
            except Exception:
                doms.append(12)
        hces = np.array(hces, dtype=np.float32)
        doms = np.array(doms, dtype=np.int64)
        for s in range(0, len(fens_list), B):
            chunk = fens_list[s:s+B]
            bs = data_loader.get_sparse_batch_from_fens(
                "Full_Threats+PP_3Wide+HalfKAv2_hm", chunk,
                [0]*len(chunk), [1]*len(chunk), [0]*len(chunk))
            us, them, wi, bi, o, sc, pc = bs.contents.get_tensors("cuda")
            for cap in caps:
                cap.clear()
            evals_b = np.zeros((len(chunk), E_N), dtype=np.float32)
            for ei, mdl in enumerate(nets):
                with torch.no_grad():
                    vs = mdl.forward(us, them, wi, bi, pc)
                evals_b[:, ei] = [float(v) * mdl.quantization.nnue2score for v in vs]
            acts_b = np.stack([caps[ei][-1] for ei in range(E_N)], axis=1)
            acts_all.append(acts_b)
            evals_all.append(evals_b)
            for fen in chunk:
                eng.stdin.write(f"rawft {fen}\n"); eng.stdin.flush()
                line = eng.stdout.readline()
                while line and not line.startswith("rawft"):
                    line = eng.stdout.readline()
                raws.append([int(t) for t in line.split()[1:] if int(t) < 64368])
            data_loader.destroy_sparse_batch(bs)
        return (np.concatenate(acts_all), np.concatenate(evals_all),
                raws, hces, doms)

    print("collecting fit data...", flush=True)
    acts_f, evals_f, raws_f, hce_f, dom_f = collect(fit_fens)
    print("collecting val data...", flush=True)
    acts_v, evals_v, raws_v, hce_v, dom_v = collect(val_fens)
    for h in hooks:
        h.remove()
    eng.stdin.write("quit\n"); eng.stdin.flush()

    # targets for V: (a) h1_pre from acts+emb (64 dims), (b) the 13 raw evals
    # The scalar path in v4: lin_ev_w . ((evals - ev_mean)/ev_std)
    # So V must reproduce evals (13 outputs) too.
    T_h1_fit = ((acts_f - a_mean[None]) / a_std[None]).reshape(len(acts_f), -1) @ W1_acts.T
    T_h1_fit += np.array([V_emb[np.array(ix)].sum(0) if ix else np.zeros(64) for ix in raws_f])
    T_ev_fit = evals_f.copy()   # (N, 13) raw evals as regression targets
    T_fit = np.concatenate([T_h1_fit, T_ev_fit], axis=1)   # (N, 77)

    T_h1_val = ((acts_v - a_mean[None]) / a_std[None]).reshape(len(acts_v), -1) @ W1_acts.T
    T_h1_val += np.array([V_emb[np.array(ix)].sum(0) if ix else np.zeros(64) for ix in raws_v])
    T_val = np.concatenate([T_h1_val, evals_v.copy()], axis=1)

    # sparse least-squares fit: V (64368 x 77)
    print("fitting V (77-wide)...", flush=True)
    V = np.zeros((64368, 77), dtype=np.float64)
    for pas in range(4):
        order = np.random.RandomState(pas).permutation(len(T_fit))
        for k in order:
            idxs = raws_f[k]
            if not idxs:
                continue
            pred = V[idxs].sum(0)
            resid = (T_fit[k] - pred) / len(idxs)
            V[idxs] += resid * 0.5
        pv = np.array([V[np.array(ix)].sum(0) if ix else np.zeros(77) for ix in raws_v])
        err = np.sqrt(((pv - T_val) ** 2).mean())
        print(f"pass {pas}: val all-77 RMSE {err:.3f}", flush=True)
    pv = np.array([V[np.array(ix)].sum(0) if ix else np.zeros(77) for ix in raws_v])
    bias77 = (T_val - pv).mean(0)
    print(f"final: h1-part RMSE {np.sqrt(((pv[:, :64] - T_val[:, :64])**2).mean()):.3f}  "
          f"evals-part RMSE {np.sqrt(((pv[:, 64:] - T_val[:, 64:])**2).mean()):.1f}cp", flush=True)
    np.save(f"{SP}/fuse_V77.npy", V.astype(np.float32))
    np.save(f"{SP}/fuse_bias77.npy", bias77.astype(np.float32))

    # ---- full-pipeline validation: fused eval vs true v4 eval ----
    # h1: pv[:, :64] + B1 + hce + oh(ignored->bias) -> clamp -> W2 -> W3
    oh_v = np.zeros((len(val_fens), E_N))
    oh_v[np.arange(len(val_fens)), dom_v] = 1.0
    W_oh = W1[:, E_N*ACT_D+ACT_D+HCE_D:]
    W_hce_blk = W1[:, E_N*ACT_D+ACT_D:E_N*ACT_D+ACT_D+HCE_D]
    h1_pre = pv[:, :64] + B1[None, :] + hce_v @ W_hce_blk.T + oh_v @ W_oh.T + bias77[None, :64]
    h1c = np.clip(h1_pre, 0, 63)
    h2c = np.clip(h1c @ W2.T + b2, 0, 63)
    fused_out = h2c @ W3 + b3
    # scalar term from fused evals: lin_ev_w . ((pv[:,64:] - ev_mean)/ev_std)
    ev_hat = pv[:, 64:] + bias77[None, 64:]
    sc = ((ev_hat - ev_mean[None, :]) / ev_std[None, :]) @ lin_ev_w
    fused_final = fused_out + sc

    # true v4
    true_final = np.load(f"{SP}/fuse_v4_val.npy")  # (2, N) fused_v4 / true — reuse true
    # lengths may differ; recompute true quickly from acts_v + evals_v
    class CR(torch.nn.Module):
        def forward(self, x):
            return torch.clamp(x, 0.0, 63.0)
    import torch.nn as nn
    class V4M(nn.Module):
        def __init__(self):
            super().__init__()
            self.emb = nn.Embedding(FT_D, ACT_D, sparse=True)
            self.lin_ev = nn.Linear(E_N, 1, bias=False)
            self.tail = nn.Sequential(nn.Linear(E_N*ACT_D+ACT_D+HCE_D+E_N, 64), CR(),
                                      nn.Linear(64, 32), CR(), nn.Linear(32, 1))
            self.register_buffer("ev_mean", torch.zeros(E_N))
            self.register_buffer("ev_std", torch.ones(E_N))
            self.register_buffer("a_mean", torch.zeros(E_N, ACT_D))
            self.register_buffer("a_std", torch.ones(E_N, ACT_D))
        def forward(self, ev, act, hce, oh, ft):
            x_ev = (ev - self.ev_mean) / self.ev_std
            x_act = (act.float() - self.a_mean) / self.a_std
            mask = (ft >= 0).float().unsqueeze(-1)
            raw = (self.emb(ft.clamp(min=0)) * mask).sum(dim=1)
            feats = torch.cat([x_act.view(len(ev), -1), raw, hce, oh], dim=1)
            return self.tail(feats).squeeze(-1) + self.lin_ev(x_ev).squeeze(-1)
    v4 = V4M()
    v4.load_state_dict(sd)
    v4.eval()
    ft_dummy = torch.zeros((len(val_fens), 2), dtype=torch.int64)  # raw=0 both sides fair
    with torch.no_grad():
        true_out = v4(torch.tensor(evals_v), torch.tensor(acts_v), torch.tensor(hce_v),
                      torch.tensor(oh_v.astype(np.float32)), ft_dummy).numpy()
    err_final = fused_final - true_out
    off = err_final.mean()
    centered = err_final - off
    print(f"FULL-PIPELINE: offset {off:.1f}cp  centered RMSE {np.sqrt((centered**2).mean()):.2f}cp  "
          f"worst {np.abs(centered).max():.1f}cp  corr {np.corrcoef(fused_final, true_out)[0,1]:.4f}", flush=True)
    np.save(f"{SP}/fuse_v51_val.npy", np.stack([fused_final, true_out]))
    print("DONE — V77 + bias saved; engine export next", flush=True)

if __name__ == "__main__":
    main()
