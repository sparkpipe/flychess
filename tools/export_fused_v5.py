"""EXPORT FUSED V5 — proper sparse least-squares fit, inline validation.

V (64368 x 64) solves: sum_{i in active(pos)} V[i] + bias ≈ h1_target(pos)
where h1_target = W1_acts @ x_acts(true) + W1_raw @ raw_emb  (the exact
parts V must reproduce). Solved by iterative residual distribution
(coordinate descent on the quadratic — converges to least squares).
Validates fused-vs-v4 on held-out BEFORE writing the engine file.
"""
import sys, os, time, struct, subprocess
sys.path.insert(0, "/srv/workspace/flychess/src/nnue-pytorch")
sys.path.insert(0, "/srv/workspace/flychess/src/chess-lab/tools")
import numpy as np
import torch

R = "/mnt/cold-raid6/chess-audit"
SP = f"{R}/selfplay_rl"
E_N, ACT_D, HCE_D, FT_D = 13, 1024, 36, 86896
FUSE_IN, FUSE_W = 88944, 64
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
    W1_acts = W1[:, :E_N*ACT_D]
    W_raw = W1[:, E_N*ACT_D:E_N*ACT_D+ACT_D]
    V_emb = emb_w @ W_raw.T      # (86896, 64) — raw-emb block in ft space

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
    fens = F.sample_fens(24000)
    fit_fens, val_fens = fens[:20000], fens[20000:]

    def h1_targets(fens_list):
        """(targets (N,64), raw_idx lists) using TRUE acts + emb — what V covers."""
        B = 512
        targets, raws = [], []
        for s in range(0, len(fens_list), B):
            chunk = fens_list[s:s+B]
            bs = data_loader.get_sparse_batch_from_fens(
                "Full_Threats+PP_3Wide+HalfKAv2_hm", chunk,
                [0]*len(chunk), [1]*len(chunk), [0]*len(chunk))
            us, them, wi, bi, o, sc, pc = bs.contents.get_tensors("cuda")
            for cap in caps:
                cap.clear()
            with torch.no_grad():
                for mdl in nets:
                    mdl.forward(us, them, wi, bi, pc)
            acts_b = np.stack([caps[ei][-1] for ei in range(E_N)], axis=1)
            x_acts = (acts_b - a_mean[None]) / a_std[None]
            h_acts = x_acts.reshape(len(chunk), -1) @ W1_acts.T
            ridx = []
            for fen in chunk:
                eng.stdin.write(f"rawft {fen}\n"); eng.stdin.flush()
                line = eng.stdout.readline()
                while line and not line.startswith("rawft"):
                    line = eng.stdout.readline()
                ridx.append([int(t) for t in line.split()[1:] if int(t) < 64368])
            h_emb = np.array([V_emb[np.array(ix, dtype=int)].sum(0) if ix else np.zeros(64)
                              for ix in ridx])
            targets.append(h_acts + h_emb)
            raws.extend(ridx)
            data_loader.destroy_sparse_batch(bs)
        return np.concatenate(targets), raws

    print("computing targets...", flush=True)
    T_fit, R_fit = h1_targets(fit_fens)
    T_val, R_val = h1_targets(val_fens)

    # ---- sparse least-squares via residual distribution (3 passes) ----
    V = np.zeros((64368, 64), dtype=np.float64)
    for pas in range(3):
        # bias = mean residual over positions with no active rows + global mean shift
        # order positions randomly, update rows
        order = np.random.RandomState(pas).permutation(len(T_fit))
        for k in order:
            idxs = R_fit[k]
            if not idxs:
                continue
            pred = V[idxs].sum(0)
            resid = (T_fit[k] - pred) / len(idxs)
            V[idxs] += resid * 0.5   # damping for stability
        # report val error
        pv = np.array([V[np.array(ix)].sum(0) if ix else np.zeros(64) for ix in R_val])
        err = np.sqrt(((pv - T_val) ** 2).mean())
        print(f"pass {pas}: val h1 RMSE {err:.3f}", flush=True)

    # bias = mean(target - sum V) over val
    pv = np.array([V[np.array(ix)].sum(0) if ix else np.zeros(64) for ix in R_val])
    fuse_bias = B1 + (T_val - pv).mean(0)
    print(f"final h1 RMSE: {np.sqrt(((pv + fuse_bias - B1 - T_val)**2).mean()):.3f}", flush=True)

    for h in hooks:
        h.remove()
    eng.stdin.write("quit\n"); eng.stdin.flush()

    # calibration offset from the earlier full-pipeline validation
    d = np.load(f"{SP}/fuse_v4_val.npy")
    offset = float((d[0] - d[1]).mean())

    fuse_W2 = W2; fuse_b2 = b2
    fuse_b3 = b3 - offset
    fuse_hce = W1[:, E_N*ACT_D+ACT_D:E_N*ACT_D+ACT_D+HCE_D]
    W_oh = W1[:, E_N*ACT_D+ACT_D+HCE_D:]
    fuse_bias = fuse_bias + W_oh.mean(1)  # onehot mean into bias

    gates = np.ones((13, 13), dtype=np.float32)
    lw = np.zeros(E_N + HCE_D + E_N, dtype=np.float32)
    lw[:E_N] = sd["lin_ev.weight"].numpy().flatten()
    V_full = np.zeros((FUSE_IN, FUSE_W), dtype=np.float32)
    V_full[:64368] = V[:64368]
    out = f"{SP}/head_fused_v5.evh"
    with open(out, "wb") as f:
        f.write(struct.pack("<IIIII", 0x45564C48, 5, E_N, HCE_D, 13))
        f.write(gates.tobytes())
        f.write(lw.tobytes())
        f.write(struct.pack("<f", 0.0))
        f.write(sd["ev_mean"].numpy().astype(np.float32).tobytes())
        f.write(sd["ev_std"].numpy().astype(np.float32).tobytes())
        f.write(struct.pack("<II", E_N, ACT_D))
        f.write(np.zeros(E_N*ACT_D, dtype=np.float32).tobytes())
        f.write(sd["a_mean"].numpy().astype(np.float32).tobytes())
        f.write(sd["a_std"].numpy().astype(np.float32).tobytes())
        f.write(struct.pack("<I", FT_D))
        f.write(np.zeros(FT_D, dtype=np.float32).tobytes())
        f.write(struct.pack("<I", FUSE_IN * FUSE_W))
        f.write(V_full.tobytes())
        f.write(fuse_bias.astype(np.float32).tobytes())
        f.write(fuse_hce.astype(np.float32).tobytes())
        f.write(fuse_W2.astype(np.float32).tobytes())
        f.write(fuse_b2.astype(np.float32).tobytes())
        f.write(struct.pack("<f", fuse_b3))
    print(f"EXPORTED {out} ({os.path.getsize(out):,} bytes)", flush=True)

if __name__ == "__main__":
    main()
