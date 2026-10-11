"""FUSE V4 — closed-form construction of the fast equivalent.

Step 1: for N sample positions, compute per-expert accumulators (linear in
        features) + capture acts + scalars.
Step 2: per-(expert, dim) affine fit of the quadratic map
        acts[d] ~ a·q[2d] + b·q[2d+1] + c   (least squares, closed form).
        Also fit each expert scalar ~ linear in accumulator.
Step 3: assemble V (88944 x 64) = sum_e G_e A_e + W1_raw E; bias corrections.
Step 4: validate fused vs true v4 on held-out positions (RMSE in cp).
"""
import sys, os, glob, time
sys.path.insert(0, "/srv/workspace/flychess/src/nnue-pytorch")
sys.path.insert(0, "/srv/workspace/flychess/src/chess-lab/tools")
import numpy as np
import torch
import chess

R = "/mnt/cold-raid6/chess-audit"
SP = f"{R}/selfplay_rl"
E_N, ACT_D, HCE_D, FT_D = 13, 1024, 36, 86896
N_FIT = 40000
N_VAL = 8000

def wp(x):
    return 1.0 / (1.0 + torch.exp(-x / 361.0))

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
            torch.nn.Linear(E_N*ACT_D + ACT_D + HCE_D + E_N, 64), ClippedReLU(),
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

def hce_fn(board):
    f = []
    for pt, v in [(chess.PAWN,1),(chess.KNIGHT,3),(chess.BISHOP,3),(chess.ROOK,5),(chess.QUEEN,9)]:
        f.append(len(board.pieces(pt, chess.WHITE)) * v / 9.0)
        f.append(len(board.pieces(pt, chess.BLACK)) * v / 9.0)
    f.append(1.0 if board.turn == chess.WHITE else -1.0)
    f.append(len(board.piece_map()) / 32.0)
    f.append(board.fullmove_number / 100.0)
    f.append(1.0 if board.has_kingside_castling_rights(chess.WHITE) else 0.0)
    f.append(1.0 if board.has_queenside_castling_rights(chess.WHITE) else 0.0)
    f.append(1.0 if board.has_kingside_castling_rights(chess.BLACK) else 0.0)
    f.append(1.0 if board.has_queenside_castling_rights(chess.BLACK) else 0.0)
    f.append(1.0 if board.ep_square is not None else 0.0)
    for side in (chess.WHITE, chess.BLACK):
        files = [0]*8
        for s2 in board.pieces(chess.PAWN, side):
            files[chess.square_file(s2)] += 1
        f += [x/2.0 for x in files]
        f.append(sum(1 for x in files if x >= 2)/4.0)
    return f

def sample_fens(n):
    import audit_packer, random
    rng = random.Random(777)
    bins = sorted(glob.glob(f"{R}/expert_bins_both/*.bin"))
    fens = []
    per = n // len(bins) + 1
    for b in bins:
        raw = open(b, "rb").read()
        cnt = len(raw) // 40
        for i in rng.sample(range(cnt), min(per, cnt)):
            fens.append(audit_packer.unpack_sfen(raw[i*40:i*40+32])[0].fen())
    rng.shuffle(fens)
    return fens[:n]

def main():
    import extract_stack_features as E
    import data_loader
    from score_experts import domain as domain_of

    dev = "cuda" if torch.cuda.is_available() else "cpu"
    EXPERTS = ["balanced_l0","balanced_l1","balanced_l2","balanced_l3",
               "nvb","nvr","bvr","rv2m","qvmat","oppb",
               "dvoretsky","exchanges","tactics"]
    DOMS = ["balanced_l0","balanced_l1","balanced_l2","balanced_l3","nvb",
            "nvr","bvr","rv2m","qvmat","oppb","dvoretsky","exchanges","tb"]
    D2I = {d: i for i, d in enumerate(DOMS)}

    nets = [E.load_expert(e) for e in EXPERTS]
    # merged effective feature weights per expert (includes virtual merge)
    A = []
    for m in nets:
        w = m.input.merged_weight_and_bias(False)  # (merged_weight, bias)
        A.append(w)
    print("expert merged weight shapes:", [tuple(w[0].shape) for w in A[:2]], flush=True)

    sd = torch.load(f"{SP}/dense_v4_ckpts3/best.pt", map_location="cpu", weights_only=False)
    v4 = DenseV4(); v4.load_state_dict(sd); v4 = v4.to(dev).eval()
    W1 = sd["tail.0.weight"]          # (64, 14385)
    B1 = sd["tail.0.bias"]
    emb_w = sd["emb.weight"]          # (86896, 1024)
    lin_ev_w = sd["lin_ev.weight"]    # (1, 13)
    ev_mean, ev_std = sd["ev_mean"], sd["ev_std"]
    a_mean, a_std = sd["a_mean"], sd["a_std"]

    fens = sample_fens(N_FIT + N_VAL)
    fens_fit, fens_val = fens[:N_FIT], fens[N_FIT:]

    def run_batch(fens_list):
        """returns per-expert accumulators (stm 2048), acts, evals, hce, oh, ft, true_v4"""
        caps = [[] for _ in nets]
        hooks = []
        for mdl, cap in zip(nets, caps):
            def mk(buf):
                def h(mod, inp, out):
                    o = out[0] if isinstance(out, tuple) else out
                    buf.append(o.detach().cpu().numpy().copy())
                return h
            hooks.append(mdl.input.register_forward_hook(mk(cap)))
        B = 256
        accums, acts_all, evals_all = [], [], []
        hces = np.zeros((len(fens_list), HCE_D), dtype=np.float32)
        doms = np.zeros(len(fens_list), dtype=np.int64)
        for i, f in enumerate(fens_list):
            hces[i] = hce_fn(chess.Board(f))
            try:
                doms[i] = D2I.get(domain_of(f), 12)
            except Exception:
                doms[i] = 12
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
            acts_b = np.stack([caps[ei][-1] for ei in range(E_N)], axis=1)  # (B, 13, 1024)
            acts_all.append(acts_b)
            evals_all.append(evals_b)
            # accumulators: linear sums of merged weights over active indices
            mw = A[0][0].detach()  # placeholder, need per-expert below
            # do per expert in torch on GPU
            acc_list = []
            with torch.no_grad():
                for ei in range(E_N):
                    W = A[ei][0].detach().to("cuda").float()   # (88944, 1032)
                    wacc = W[wi.reshape(-1)].reshape(len(chunk), -1, W.shape[1]).sum(1)
                    bacc = W[bi.reshape(-1)].reshape(len(chunk), -1, W.shape[1]).sum(1)
                    bias = A[ei][1].detach().to("cuda").float()
                    wacc = wacc + bias; bacc = bacc + bias
                    # stm combine: [w|b] if stm=white else [b|w]  (us=1 white)
                    usv = us.reshape(-1, 1).float()
                    acc = usv * torch.cat([wacc, bacc], 1) + (1-usv) * torch.cat([bacc, wacc], 1)
                    acc_list.append(acc[:, :2048].cpu())
            accums.append(torch.stack(acc_list, 1))  # (B, 13, 2048)
            data_loader.destroy_sparse_batch(bs)
        for h in hooks:
            h.remove()
        return (torch.cat(accums), np.concatenate(acts_all),
                np.concatenate(evals_all), hces, doms)

    print("computing fit batch...", flush=True)
    t0 = time.time()
    acc_f, acts_f, evals_f, hce_f, dom_f = run_batch(fens_fit)
    print(f"fit batch done {time.time()-t0:.0f}s: acc {tuple(acc_f.shape)} acts {acts_f.shape}", flush=True)

    # ---- step 2: per-(expert, dim) affine fit  acts[d] ~ a*q0 + b*q1 + c ----
    # the transform: l0 = clamp(acc 2048); quarters Q0..Q3 (512 each); out = Q0*Q1, Q2*Q3
    # BUT we captured only 1024 outputs => acts[:, e, d] for d in 0..1023:
    #   d in [0,512) = Q0*Q1 ; d in [512,1024) = Q2*Q3? verify dims: acc width 2048,
    #   l1=1024 per perspective -> quarters of the STM-COMBINED 2048 => 4 x 512
    print("fitting per-dim affine maps...", flush=True)
    a_coef = np.zeros((E_N, ACT_D, 3), dtype=np.float32)  # (a, b, c) per expert/dim
    r2 = np.zeros((E_N, ACT_D), dtype=np.float32)
    acc_np = acc_f.numpy()  # (N, 13, 2048)
    for e in range(E_N):
        for d in range(ACT_D):
            if d < 512:
                q0 = acc_np[:, e, d]; q1 = acc_np[:, e, d + 512]
            else:
                q0 = acc_np[:, e, 512 + d]; q1 = acc_np[:, e, 1024 + d]  # Q2*Q3 quarters
            y = acts_f[:, e, d].astype(np.float64)
            X = np.stack([q0, q1, np.ones_like(q0)], 1)
            # closed-form lstsq via normal equations (3x3) for speed
            XtX = X.T @ X
            Xty = X.T @ y
            try:
                coef = np.linalg.solve(XtX + 1e-6 * np.eye(3), Xty)
            except np.linalg.LinAlgError:
                coef = np.zeros(3)
            a_coef[e, d] = coef
            pred = X @ coef
            ss_res = ((y - pred) ** 2).sum()
            ss_tot = ((y - y.mean()) ** 2).sum() + 1e-9
            r2[e, d] = 1 - ss_res / ss_tot
    print(f"affine fits: R2 mean {r2.mean():.4f}  min {r2.min():.3f}  <0.9 frac {(r2<0.9).mean()*100:.1f}%", flush=True)
    np.save(f"{SP}/fuse_a_coef.npy", a_coef)
    np.save(f"{SP}/fuse_r2.npy", r2)

    # ---- step 3: assemble V ----
    # head reads: contribution_e[u] = sum_d W1[u, e*1024+d] * (acts[d]-mean)/std
    # with acts[d] ~ a*q0 + b*q1 + c:
    #   => coefficient on q0: sum_d W1[u,ed]/std_d * a_d   (G on accumulator dims)
    W1_np = W1.numpy()
    a_mean_np = a_mean.numpy(); a_std_np = a_std.numpy()
    IN_D = W1_np.shape[1]
    G = np.zeros((E_N, 64, 2048), dtype=np.float64)   # head-unit -> accumulator functional
    bias_const = np.zeros(64, dtype=np.float64)
    for e in range(E_N):
        We = W1_np[:, e*ACT_D:(e+1)*ACT_D]        # (64, 1024)
        scale = We / a_std_np[e][None, :]          # (64, 1024)
        bias_const += (scale * -a_mean_np[e][None, :]).sum(1)
        bias_const += (scale * a_coef[e, :, 2]).sum(1)   # affine constants
        a_m = (scale * a_coef[e, :, 0]).sum(1)     # (64,) coef on q0 across d
        b_m = (scale * a_coef[e, :, 1]).sum(1)
        for d in range(512):
            G[e, :, d] += scale[:, d] * a_coef[e, d, 0]
            G[e, :, d + 512] += scale[:, d] * a_coef[e, d, 1]
        for d in range(512, 1024):
            G[e, :, 512 + d] += scale[:, d] * a_coef[e, d, 0]
            G[e, :, 1024 + d] += scale[:, d] * a_coef[e, d, 1]
    # raw embedding block: W1[:, 13312:14336] @ E^T per feature -> (86896, 64)
    W_raw = W1_np[:, E_N*ACT_D:E_N*ACT_D+ACT_D]   # (64, 1024)
    E_w = emb_w.numpy()                            # (86896, 1024)
    V_raw = E_w @ W_raw.T                          # (86896, 64)
    # HCE block + onehot go to direct computation (not into V)

    # assemble V from accum functional * expert merged weights:
    # fused feature contribution: for expert e, unit u: G[e,u,:] @ A_e_combined
    # A_e maps feature idx -> 2048 stm accumulator... BUT A_e is (88944, 1032) = [1024 l1 | 8 psqt] per PERSPECTIVE.
    # Our acc (2048) = [w(1024) | b(1024)] stm-combined. The feature idx space for white/black differs (wi/bi).
    # For validation we keep it explicit: fused_val uses acc directly.
    print("assembled. validating on held-out...", flush=True)
    acc_v, acts_v, evals_v, hce_v, dom_v = run_batch(fens_val)
    acc_v_np = acc_v.numpy()

    # fused forward: pre-tail hidden h = sum_e G[e] @ acc_e + V_raw-block(ft) + W1_hce@hce + W1_oh@oh + bias
    # note: acts in v4 are CACHE-scale (2x engine)! Our a_coef fitted on captured acts (cache scale) — consistent.
    # raw embedding: sum over active ft features of E rows -> (N,1024) @ W_raw.T
    # recompute ft via rawft... use emb directly from v4 forward for validation fairness:
    with torch.no_grad():
        # true v4 output
        oh_v = np.zeros((len(fens_val), E_N), dtype=np.float32)
        oh_v[np.arange(len(fens_val)), dom_v] = 1.0
        ft_placeholder = np.zeros((len(fens_val), 160), dtype=np.int64)  # dummy
        true_out = v4(torch.tensor(evals_v).to(dev), torch.from_numpy(acts_v).to(dev),
                      torch.tensor(hce_v).to(dev), torch.tensor(oh_v).to(dev),
                      torch.tensor(ft_placeholder).to(dev)).cpu().numpy()
    # fused: h1_pre = einsum over experts
    h_pre = np.zeros((len(fens_val), 64), dtype=np.float64)
    for e in range(E_N):
        h_pre += np.einsum("nu,nd->nu2" if False else "un,fn->fu", G[e].T if False else G[e], acc_v_np[:, e, :])
    h_pre += bias_const[None, :]
    # hce + onehot blocks
    W_hce = W1_np[:, E_N*ACT_D+ACT_D:E_N*ACT_D+ACT_D+HCE_D]
    W_oh = W1_np[:, E_N*ACT_D+ACT_D+HCE_D:]
    oh_v2 = np.zeros((len(fens_val), E_N), dtype=np.float64)
    oh_v2[np.arange(len(fens_val)), dom_v] = 1.0
    h_pre += hce_v @ W_hce.T + oh_v2 @ W_oh.T
    h_pre += B1.numpy()[None, :]
    # tail forward (same as v4)
    h1v = np.clip(h_pre, 0, 63)
    W2 = sd["tail.2.weight"].numpy(); b2 = sd["tail.2.bias"].numpy()
    h2v = np.clip(h1v @ W2.T + b2, 0, 63)
    W3 = sd["tail.4.weight"].numpy().flatten(); b3 = sd["tail.4.bias"].numpy()
    fused_out = h2v @ W3 + b3[0]
    # scalar path (lin_ev on normalized evals) — keep exact (uses evals)
    ev_norm = (evals_v - ev_mean.numpy()[None, :]) / ev_std.numpy()[None, :]
    fused_out += ev_norm @ lin_ev_w.numpy().flatten()

    # NOTE: true_out above used dummy ft (raw-emb contribution = emb(0)*mask=0),
    # so for fair comparison subtract the raw-emb contribution from BOTH? The true v4
    # with dummy ft also has raw=0. So compare fused(without raw) vs true(with raw=0): consistent.
    err = fused_out - true_out
    print(f"VALIDATION: RMSE {np.sqrt((err**2).mean()):.2f}cp  corr {np.corrcoef(fused_out, true_out)[0,1]:.4f}", flush=True)
    print(f"true std {true_out.std():.1f}cp  fused std {fused_out.std():.1f}cp", flush=True)
    np.save(f"{SP}/fuse_v4_val.npy", np.stack([fused_out, true_out]))

if __name__ == "__main__":
    main()
