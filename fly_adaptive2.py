"""ADAPTIVE NUMERICS PHASE 2: per-neuron trainable leak (multi-timescale
fly imagination). Mirrors fly_imagine.py protocol EXACTLY (same inputs
1024, same W_sens -> frozen connectome -> planes/reach heads, same
curriculum I1/I2/I3, same tiered gates) with ONE change inside propagate:

    a = a + lam_i * ((WT @ a) - a)          lam_i in [0.01, 0.5] trainable
    a = a / (mean|a| + 1e-6) * 2.0          per-step mean-abs norm (no rails)

lam_i = 0.01 + 0.49 * sigmoid(lam_logit_i); lam_logit init so lam starts
at ~0.25 (sigma(-0.0408) = 0.4898 -> 0.01 + 0.49*0.4898 = 0.250). NOTE:
spec's literal "-1.1" was derived as logit(0.25) WITHOUT the affine bound
mapping (0.01+0.49*sigmoid(-1.1) = 0.132); we honor the stated intent
"lam starts ~0.25".

ADAPT=leak  per-neuron trainable lam (the phase-2 arm)
ADAPT=fixed fixed scalar leak cur.LEAK + normalization (ablation control)

Gates per lesson: PASS planes>=0.999 & reach_nz>=0.98; PASS-MARGINAL
planes>=0.999 & reach_nz>=0.975; 24000-step cap (PASS-MARGINAL at cap if
best>=0.975, LOW-PROCEED if best>=0.95); EXHAUSTED hard-stop (sys.exit(1))
if best reach_nz<0.95. Stall (8 non-improving evals) is tracked and logged
but does NOT terminate, matching the reference protocol's semantics.
Diagnostics every 500 steps: lam min/mean/max + spread + at-rail fraction
(|a|>=19.9, must stay <0.05); per-class reach accuracy (own/enemy/empty/
off) every 1000 steps. Checkpoint fly_adapt2_leak.pt, log adapt2_leak.out.
"""
import os, sys, json, random
import numpy as np
import torch

_LAB = os.path.expanduser("~") + "/chess-lab"
sys.path.insert(0, _LAB)
import fly_curriculum as cur
import fly_imagine as fi

DEV = cur.DEV
STATE = os.environ.get("ADAPT_STATE", _LAB + "/fly_adapt2_leak.pt")
ADAPT = os.environ.get("ADAPT", "leak")
IORDER = fi.IORDER
CAPSTEP = 24000
LAM_MIN, LAM_SPAN = 0.01, 0.49          # lam = LAM_MIN + LAM_SPAN*sigmoid
LAM_LOGIT0 = float(np.log((0.25 - LAM_MIN) / (LAM_MIN + LAM_SPAN - 0.25)))
                                        # ~ -0.0408 -> lam0 = 0.25


class AdaptFly(torch.nn.Module):
    """fi.ImagineFly mirror + per-neuron leak lam (trainable)."""

    def __init__(self, readout_n=2048):
        super().__init__()
        z = np.load(cur.BRAIN)
        m = torch.sparse_csr_tensor(
            torch.from_numpy(z["indptr"].astype(np.int64)),
            torch.from_numpy(z["indices"].astype(np.int64)),
            torch.from_numpy(z["data"].astype(np.float32)),
            size=tuple(z["shape"]), device=DEV).t().to_sparse_csr()
        self.WT = m
        self.N = m.shape[0]
        sens = np.load(cur.SENS).astype(np.int64)
        self.register_buffer("inj_idx", torch.from_numpy(sens).to(DEV))
        self.W_sens = torch.nn.Linear(fi.INDIM, len(sens))
        rng = np.random.default_rng(20260918)      # SAME head as reference
        free = np.setdiff1d(np.arange(self.N), sens)
        self.register_buffer(
            "head_idx", torch.from_numpy(
                rng.permutation(free)[:readout_n].astype(np.int64)).to(DEV))
        self.planes_head = torch.nn.Sequential(
            torch.nn.Linear(readout_n, 512), torch.nn.GELU(),
            torch.nn.Linear(512, (fi.PLANES + 1) * 64))
        self.reach_head = torch.nn.Sequential(
            torch.nn.Linear(readout_n, 512), torch.nn.GELU(),
            torch.nn.Linear(512, 64))
        self.lam_logit = torch.nn.Parameter(
            torch.full((self.N, 1), LAM_LOGIT0, device=DEV))

    def lam_value(self):
        return LAM_MIN + LAM_SPAN * torch.sigmoid(self.lam_logit)

    def propagate(self, x, return_a=False):
        s = torch.clamp(self.W_sens(x), -6, 6)
        a = torch.zeros(self.N, x.shape[0], device=DEV)
        a[self.inj_idx] = s.T
        if ADAPT == "leak":
            lam = self.lam_value()                       # (N,1) broadcast
            for _ in range(cur.PROP_STEPS):
                a = a + lam * (self.WT @ a - a)
                a = a / (a.abs().mean() + 1e-6) * 2.0    # fixed norm, no rails
        else:                                            # ADAPT=fixed control
            for _ in range(cur.PROP_STEPS):
                a = (1 - cur.LEAK) * a + cur.LEAK * (self.WT @ a)
                a = a / (a.abs().mean() + 1e-6) * 2.0
        return a if return_a else a

    def forward(self, x):
        a = self.propagate(x)
        h = a[self.head_idx].T
        return self.planes_head(h), self.reach_head(h)


def diag_lam_rail(model, lesson, seed=9100):
    """lam distribution + at-rail fraction on one fresh sample."""
    rng = random.Random(seed + cur._phash(lesson) % 99991)
    tr = fi.sample_transition(rng)
    if tr is None:
        return None
    b, mv, qs = tr
    x, _, _ = fi.transition_tensors(b, mv, qs)
    with torch.no_grad():
        lam = model.lam_value().flatten()
        a = model.propagate(torch.from_numpy(x[None]).to(DEV), return_a=True)
        rail = float((a.abs() >= 19.9).float().mean())
        l = lam.cpu().numpy()
        return {"lam_min": round(float(l.min()), 4),
                "lam_mean": round(float(l.mean()), 4),
                "lam_max": round(float(l.max()), 4),
                "lam_p10": round(float(np.percentile(l, 10)), 4),
                "lam_p90": round(float(np.percentile(l, 90)), 4),
                "lam_frac_lo": round(float((l < 0.10).mean()), 4),
                "lam_frac_hi": round(float((l > 0.40).mean()), 4),
                "at_rail": round(rail, 5)}


def perclass_reach_acc(model, lesson, n=96, seed=8100):
    """Reach accuracy split by target class: own(-1)/enemy(+1)/empty(.3)/off(0)."""
    rng = random.Random(seed + cur._phash(lesson) % 99991)
    cap = None if lesson == "I1" else (True if lesson == "I2" else None)
    tot = {k: [0.0, 0] for k in ("own", "enemy", "empty", "off")}
    for _ in range(n):
        tr = fi.sample_transition(rng, capture=cap)
        if tr is None:
            continue
        b, mv, qs = tr
        x, p1, r1 = fi.transition_tensors(b, mv, qs)
        with torch.no_grad():
            _, pred_r = model(torch.from_numpy(x[None]).to(DEV))
        pr = pred_r[0].cpu().numpy()
        ok = np.abs(pr - r1) < 0.25
        for name, mask in (("own", r1 < -0.5), ("enemy", r1 > 0.5),
                           ("empty", (r1 > 0.05) & (r1 < 0.5)),
                           ("off", np.abs(r1) <= 1e-6)):
            tot[name][0] += float(ok[mask].sum())
            tot[name][1] += int(mask.sum())
    out = {}
    for k, (c, t) in tot.items():
        out[k] = round(c / t, 4) if t else None
    return out


def main():
    torch.manual_seed(0)
    model = AdaptFly().to(DEV)
    if os.path.exists(STATE):
        model.load_state_dict(torch.load(STATE, weights_only=True))
        print("resumed", STATE, flush=True)
    else:
        print("FRESH adaptive fly ADAPT=%s lam0~%.3f (logit %.4f)"
              % (ADAPT, 0.25, LAM_LOGIT0), flush=True)
    opt = torch.optim.Adam(model.parameters(), lr=3e-4)
    for lesson in IORDER:
        print(f"=== ADAPT2 MILESTONE {lesson} (ADAPT={ADAPT}) ===", flush=True)
        rng = random.Random(8000 + (0 if lesson == "I1" else
                                    (1 if lesson == "I2" else 2)))
        best = -1.0
        stall = 0
        for step in range(1, CAPSTEP + 1):
            fi.train_step(model, opt, rng, lesson)   # identical protocol
            if step % 50:
                continue
            pacc, rfull, rnz = fi.eval_battery(model, lesson)
            rec = {"adapt2_milestone": lesson, "step": step,
                   "planes": round(pacc, 4), "reach_full": round(rfull, 4),
                   "reach_nz": round(rnz, 4)}
            if step % 500 == 0:
                d = diag_lam_rail(model, lesson)
                if d:
                    rec.update(d)
            if step % 1000 == 0:
                rec["reach_perclass"] = perclass_reach_acc(model, lesson)
            print(json.dumps(rec), flush=True)
            if pacc >= 0.999 and rnz >= 0.98:
                print(f"ADAPT2 {lesson} PASS at step {step} "
                      f"(planes={pacc:.3f} reach_nz={rnz:.3f})", flush=True)
                torch.save(model.state_dict(), STATE)
                break
            if pacc >= 0.999 and rnz >= 0.975:
                print(f"ADAPT2 {lesson} PASS-MARGINAL at step {step} "
                      f"(planes={pacc:.3f} reach_nz={rnz:.3f}) "
                      f"- accepted at floor, watched", flush=True)
                torch.save(model.state_dict(), STATE)
                break
            if step >= 100 and rnz <= best + 0.001:
                stall += 1
            else:
                stall = 0
            best = max(best, rnz)
            if stall == 8:
                print(f"ADAPT2 {lesson} stall-watch at step {step} "
                      f"(best {best:.3f}) - protocol continues to cap",
                      flush=True)
            if step == CAPSTEP:
                if best >= 0.975:
                    print(f"ADAPT2 {lesson} PASS-MARGINAL at cap "
                          f"(best {best:.3f}) - accepted at floor", flush=True)
                    torch.save(model.state_dict(), STATE)
                    break
                if best < 0.95:
                    print(f"ADAPT2 {lesson} EXHAUSTED (best {best:.3f})",
                          flush=True)
                    torch.save(model.state_dict(), STATE)
                    sys.exit(1)
                print(f"ADAPT2 {lesson} LOW-PROCEED at cap (best {best:.3f})",
                      flush=True)
                torch.save(model.state_dict(), STATE)
    torch.save(model.state_dict(), STATE)
    with torch.no_grad():
        l = model.lam_value().flatten().cpu().numpy()
    print("FINAL lam distribution: min=%.4f mean=%.4f max=%.4f "
          "p10=%.4f p90=%.4f frac<0.1=%.3f frac>0.4=%.3f"
          % (l.min(), l.mean(), l.max(), np.percentile(l, 10),
             np.percentile(l, 90), float((l < 0.1).mean()),
             float((l > 0.4).mean())), flush=True)
    d2 = fi.depth2_reach_acc(model, random.Random(99))
    print(f"ADAPT2 DEPTH-2 free-running reach nz acc = {d2:.3f}", flush=True)
    print("ADAPT2 ALL MILESTONES PROCESSED", flush=True)


if __name__ == "__main__":
    main()
