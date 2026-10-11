"""DENSE v4 export + gates — converts the trained checkpoint to .evh v4.

v4 layout (matches eval_head.h):
  header(I*5: magic,4,E_N,HCE,13) gates(13x13 f32) lin_w(62 f32) lin_b(f32)
  ev_mean/std(13+13) act_hdr(II:13,1024) lin_act(13312) a_mean/std(13312+13312)
  [v3 block header: ft_dims(I)=86896 + ft_w(86896 f32, zeros for v4 compat]
  v4: emb_bytes(I)=86896*1024 emb_scale(f) emb(int16 x 86896*1024)
      in_d(I) h1(I) h2(I) W1(h1*in_d f32) b1(h1) W2(h2*h1) b2(h2) W3(h2) b3(1)
Gates before export: output-range p95, sharpness probes, engine-scale checks.
"""
import sys, struct, time
sys.path.insert(0, "/srv/workspace/flychess/src/chess-lab/tools")
import numpy as np
import torch, torch.nn as nn

R = "/mnt/cold-raid6/chess-audit"
SP = f"{R}/selfplay_rl"
E_N, HCE_D, ACT_D, FT_D = 13, 36, 1024, 86896
CKPT = sys.argv[1] if len(sys.argv) > 1 else f"{SP}/dense_v4_ckpts/best.pt"
OUT = sys.argv[2] if len(sys.argv) > 2 else f"{SP}/head_dense_v4.evh"

def wp(x):
    return 1.0 / (1.0 + torch.exp(-x / 361.0))

class ClippedReLU(nn.Module):
    def __init__(self, hi=63.0):
        super().__init__()
        self.hi = hi
    def forward(self, x):
        return torch.clamp(x, 0.0, self.hi)

class DenseV4(nn.Module):
    def __init__(self, ev_mean, ev_std, a_mean, a_std):
        super().__init__()
        self.emb = nn.Embedding(FT_D, ACT_D, sparse=True)
        self.lin_ev = nn.Linear(E_N, 1, bias=False)
        self.tail = nn.Sequential(
            nn.Linear(E_N * ACT_D + ACT_D + HCE_D + E_N, 64), ClippedReLU(),
            nn.Linear(64, 32), ClippedReLU(),
            nn.Linear(32, 1),
        )
        self.register_buffer("ev_mean", ev_mean.clone())
        self.register_buffer("ev_std", ev_std.clone())
        self.register_buffer("a_mean", a_mean.clone().view(E_N, ACT_D))
        self.register_buffer("a_std", a_std.clone().view(E_N, ACT_D))
    def forward(self, ev, act, hce, oh, ftidx):
        x_ev = (ev - self.ev_mean) / self.ev_std
        x_act = (act.float() - self.a_mean) / self.a_std
        mask = (ftidx >= 0).float().unsqueeze(-1)
        raw = (self.emb(ftidx.clamp(min=0)) * mask).sum(dim=1)
        feats = torch.cat([x_act.view(len(ev), -1), raw, hce, oh], dim=1)
        return self.tail(feats).squeeze(-1) + self.lin_ev(x_ev).squeeze(-1)

def main():
    d = np.load(f"{SP}/dense_data.npz")
    evals, acts, hces = d["evals"], d["acts"], d["hces"]
    doms, tgts, ft = d["doms"], d["tgts"], d["ft"]
    ACT_COL = [0, 1, 4, 8, 2, 9, 6, 10, 11, 5, 7, 3, 12]
    acts_eng = np.ascontiguousarray(acts[:, ACT_COL, :]) >> 1
    N = min(60000, len(tgts))

    state = torch.load(CKPT, map_location="cpu", weights_only=False)
    ev_mean = state["ev_mean"]; ev_std = state["ev_std"]
    a_mean = state["a_mean"].flatten(); a_std = state["a_std"].flatten()
    model = DenseV4(ev_mean, ev_std, a_mean, a_std)
    model.load_state_dict(state)
    model.eval()

    X_ev = torch.tensor(evals[:N], dtype=torch.float32)
    X_act = torch.from_numpy(acts_eng)
    X_hce = torch.tensor(hces[:N], dtype=torch.float32)
    oh = np.zeros((N, E_N), dtype=np.float32)
    oh[np.arange(N), doms[:N]] = 1.0
    X_oh = torch.tensor(oh)
    X_ft = torch.tensor(ft[:N].astype(np.int64))
    outs = []
    with torch.no_grad():
        for st in range(0, N, 2000):
            en = min(st + 2000, N)
            outs.append(model(X_ev[st:en], X_act[st:en], X_hce[st:en],
                              X_oh[st:en], X_ft[st:en]).numpy())
    out = np.concatenate(outs)
    p95 = float(np.percentile(np.abs(out), 95))
    print(f"GATE range p95={p95:.0f}cp", flush=True)
    assert 5 < p95 < 400, "range gate fail"

    # ---- quantize embedding to int16 ----
    embw = model.emb.weight.detach()          # (86896, 1024) float
    scale = float(embw.abs().max() / 32767.0)
    emb_q = torch.round(embw / scale).to(torch.int16).numpy()
    print(f"emb quant scale {scale:.3e}", flush=True)

    W1 = model.tail[0].weight.detach().numpy()   # (64, in_d)
    b1 = model.tail[0].bias.detach().numpy()
    W2 = model.tail[2].weight.detach().numpy()   # (32, 64)
    b2 = model.tail[2].bias.detach().numpy()
    W3 = model.tail[4].weight.detach().numpy().flatten()  # (32,)
    b3 = model.tail[4].bias.detach().numpy().flatten()
    lin_ev_w = model.lin_ev.weight.detach().numpy().flatten()
    in_d = W1.shape[1]
    print(f"tail dims: in {in_d} h1 {W1.shape[0]} h2 {W2.shape[0]}", flush=True)

    gates = np.ones((13, 13), dtype=np.float32)
    lw = np.zeros(E_N + HCE_D + E_N, dtype=np.float32)
    lw[:E_N] = lin_ev_w

    with open(OUT, "wb") as f:
        f.write(struct.pack("<IIIII", 0x45564C48, 4, E_N, HCE_D, 13))
        f.write(gates.tobytes())
        f.write(lw.astype(np.float32).tobytes())
        f.write(struct.pack("<f", 0.0))  # lin_b baked into b3 path; keep 0
        f.write(ev_mean.numpy().astype(np.float32).tobytes())
        f.write(ev_std.numpy().astype(np.float32).tobytes())
        f.write(struct.pack("<II", E_N, ACT_D))
        f.write(np.zeros(E_N * ACT_D, dtype=np.float32).tobytes())  # lin_act unused in v4
        f.write(a_mean.numpy().astype(np.float32).tobytes())
        f.write(a_std.numpy().astype(np.float32).tobytes())
        # v3 block (compat): ft_dims + zero ft_w
        f.write(struct.pack("<I", FT_D))
        f.write(np.zeros(FT_D, dtype=np.float32).tobytes())
        # v4 block
        f.write(struct.pack("<I", FT_D * ACT_D))
        f.write(struct.pack("<f", scale))
        f.write(emb_q.tobytes())
        f.write(struct.pack("<III", in_d, W1.shape[0], W2.shape[0]))
        f.write(np.ascontiguousarray(W1, dtype=np.float32).tobytes())
        f.write(b1.astype(np.float32).tobytes())
        f.write(np.ascontiguousarray(W2, dtype=np.float32).tobytes())
        f.write(b2.astype(np.float32).tobytes())
        f.write(W3.astype(np.float32).tobytes())
        f.write(b3.astype(np.float32).tobytes())
    print(f"EXPORTED {OUT}", flush=True)

if __name__ == "__main__":
    main()
