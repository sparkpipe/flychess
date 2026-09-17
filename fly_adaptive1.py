"""PHASE 1 ADAPTIVE NUMERICS — fly_imagine protocol + adaptive dynamics.
Protocol IDENTICAL to fly_imagine.py (lessons I1/I2/I3, eval every 50 steps
via fi.eval_battery, tiered gates planes>=0.999 + reach_nz {0.98 full /
0.975 marginal / 0.95 low}, 24000-step cap, EXHAUSTED hard-stop exit(1) if
best reach_nz < 0.95 at cap, stable seeds torch.manual_seed(0) +
random.Random(8000+lesson_idx) train / seed+cur._phash eval). Inputs are
UNTOUCHED: fi.sample_transition / fi.transition_tensors (1024-dim) / fi.loss_fn.

ADAPT=ln    per propagation step: a = LN(a) * gamma + beta with trainable
            per-neuron gamma/beta. gamma init 2.5066 = 2.0/sqrt(2/pi) so a
            post-LN standard-normal activation has mean|a| ~= 2.0 (the
            ANORM_T=2.0 fixed-normalization baseline target); beta init 0.
ADAPT=homeo fixed mean-abs normalization a /= (mean|a| + 1e-6) * 2.0 then
            a *= g with trainable per-neuron gain g (init 1.0) — the
            homeostatic per-neuron range equalizer.

Health probe every 500 steps: fraction of |a| >= 19.9 per propagation step
(target < 0.05 at step 6; unnormalized LEAK=0.5 baseline rails ~94-96%).
Per-class reach accuracy (own=-1 / enemy=+1 / empty=0.3 / off=0 bins)
every 1000 steps — occupied bins are the Phase 1 hypothesis target.
"""
import os, sys, json, random
import numpy as np
import torch
import chess

_LAB = os.path.expanduser("~") + "/chess-lab"
sys.path.insert(0, _LAB)
import fly_imagine as fi
import fly_curriculum as cur

DEV = cur.DEV
ADAPT = os.environ.get("ADAPT", "ln")
assert ADAPT in ("ln", "homeo"), "ADAPT must be ln or homeo, got %r" % ADAPT
STATE = _LAB + "/fly_adapt1_%s.pt" % ADAPT
CAP_STEPS = 24000
RAIL = 19.9
TGT = 2.0                                   # mean|a| target (= ANORM_T)
GAMMA_INIT = TGT / float(np.sqrt(2.0 / np.pi))   # -> post-LN mean|a| ~ 2.0
EPS_LN = 1e-5


class AdaptiveFly(fi.ImagineFly):
    """fi.ImagineFly with the dynamics swap inside propagate + probes.
    Same W_sens(1024 -> N_sens), frozen WT, same heads/readout neurons
    (same head_idx seed 20260918), same init RNG stream as the baseline."""

    def __init__(self, readout_n=2048):
        super().__init__(readout_n=readout_n)
        self.gamma = torch.nn.Parameter(
            torch.full((self.N,), float(GAMMA_INIT), device=DEV))
        self.beta = torch.nn.Parameter(torch.zeros(self.N, device=DEV))
        self.gain = torch.nn.Parameter(torch.ones(self.N, device=DEV))
        self.probe = False            # rail stats only computed when probing
        self.last_rails = None        # per-step frac |a| >= RAIL
        self.last_meanabs = None      # final-step mean|a|

    def propagate(self, x):
        s = torch.clamp(self.W_sens(x), -6, 6)
        a = torch.zeros(self.N, x.shape[0], device=DEV)
        a[self.inj_idx] = s.T
        rails = []
        for _ in range(cur.PROP_STEPS):
            a = (1 - cur.LEAK) * a + cur.LEAK * (self.WT @ a)
            if ADAPT == "ln":
                mu = a.mean(dim=0, keepdim=True)
                var = a.var(dim=0, unbiased=False, keepdim=True)
                a = (a - mu) / torch.sqrt(var + EPS_LN)
                a = self.gamma[:, None] * a + self.beta[:, None]
            else:                      # homeo
                a = a / (a.abs().mean() + 1e-6) * TGT
                a = a * self.gain[:, None]
            if self.probe:
                rails.append(float((a.abs() >= RAIL).float().mean()))
        if self.probe:
            self.last_rails = rails
            self.last_meanabs = float(a.abs().mean())
        return a


def _probe_batch(n=16, seed=5501):
    rng = random.Random(seed)
    xs = []
    while len(xs) < n:
        tr = fi.sample_transition(rng)
        if tr is None:
            continue
        b, mv, qs = tr
        x, _, _ = fi.transition_tensors(b, mv, qs)
        xs.append(x)
    return torch.from_numpy(np.stack(xs)).to(DEV)


def calibrate_gamma(model, iters=3):
    """One-time deterministic init fix for ADAPT=ln: the post-LN activation
    distribution is heavy-tailed (not Gaussian), so the nominal
    GAMMA_INIT = 2.0/sqrt(2/pi) lands mean|a| ~ 0.84, not ~2.0. Solve the
    scalar init scale by fixed-point iteration on a FIXED probe batch so
    post-scale mean|a| ~ TGT at step 6 (baseline operating point). gamma
    stays a fully trainable per-neuron vector afterwards."""
    xt = _probe_batch()
    g = float(model.gamma.detach().abs().mean()) or GAMMA_INIT
    with torch.no_grad():
        for _ in range(iters):
            s = torch.clamp(model.W_sens(xt), -6, 6)
            a = torch.zeros(model.N, xt.shape[0], device=DEV)
            a[model.inj_idx] = s.T
            for _ in range(cur.PROP_STEPS):
                a = (1 - cur.LEAK) * a + cur.LEAK * (model.WT @ a)
                mu = a.mean(dim=0, keepdim=True)
                var = a.var(dim=0, unbiased=False, keepdim=True)
                a = (a - mu) / torch.sqrt(var + EPS_LN) * g
            g = g * TGT / max(float(a.abs().mean()), 1e-6)
        model.gamma.fill_(g)
    return g


def rail_probe(model, n=16, seed=5501):
    """At-rail fraction per propagation step on a fresh fixed batch."""
    xt = _probe_batch(n, seed)
    was = model.probe
    model.probe = True
    with torch.no_grad():
        model(xt)
    model.probe = was
    return model.last_rails, model.last_meanabs


def perclass_reach(model, lesson, n=192, seed=6200):
    """Reach accuracy split by label class: own -1 / enemy +1 / empty 0.3 /
    off 0 (same |pred-label|<0.25 criterion as fi.eval_battery)."""
    rng = random.Random(seed + cur._phash(lesson) % 99991)
    ok_b = {"own": [], "enemy": [], "empty": [], "off": []}
    n_b = {"own": 0, "enemy": 0, "empty": 0, "off": 0}
    for _ in range(n):
        cap = None if lesson == "I1" else (True if lesson == "I2" else None)
        tr = fi.sample_transition(rng, capture=cap)
        if tr is None:
            continue
        b, mv, qs = tr
        x, p1, r1 = fi.transition_tensors(b, mv, qs)
        with torch.no_grad():
            _, pred_r = model(torch.from_numpy(x[None]).to(DEV))
        pr = pred_r[0].cpu().numpy()
        ok = np.abs(pr - r1) < 0.25
        for name, v in (("own", -1.0), ("enemy", 1.0),
                        ("empty", 0.3), ("off", 0.0)):
            m = np.abs(r1 - v) < 1e-6
            if m.any():
                ok_b[name].append(float(ok[m].mean()))
                n_b[name] += int(m.sum())
    out = {}
    for name in ("own", "enemy", "empty", "off"):
        if ok_b[name]:
            out[name] = round(float(np.mean(ok_b[name])), 4)
    out["counts"] = n_b
    return out


def main():
    torch.manual_seed(0)
    model = AdaptiveFly().to(DEV)
    if os.path.exists(STATE):
        model.load_state_dict(torch.load(STATE, weights_only=True))
        print("resumed", STATE, flush=True)
    else:
        gcal = calibrate_gamma(model) if (ADAPT == "ln" and
                                          os.environ.get("ACAL", "1") == "1"
                                          ) else None
        print("FRESH adaptive fly mode=%s N=%d gamma_init=%.4f "
              "gamma_calibrated=%s params+%.1fM"
              % (ADAPT, model.N, GAMMA_INIT,
                 "n/a" if gcal is None else "%.4f" % gcal,
                 3 * model.N / 1e6), flush=True)
    opt = torch.optim.Adam(model.parameters(),
                           lr=float(os.environ.get("ILR", "3e-4")))
    for lesson in fi.IORDER:
        print("=== I MILESTONE %s (adapt=%s) ===" % (lesson, ADAPT),
              flush=True)
        rng = random.Random(8000 + (0 if lesson == "I1" else
                                    (1 if lesson == "I2" else 2)))
        best = -1.0
        stall = 0
        for step in range(1, CAP_STEPS + 1):
            fi.train_step(model, opt, rng, lesson)
            if step % 50:
                continue
            pacc, rfull, rnz = fi.eval_battery(model, lesson)
            print(json.dumps({"adapt": ADAPT, "i_milestone": lesson,
                              "step": step, "planes": round(pacc, 4),
                              "reach_full": round(rfull, 4),
                              "reach_nz": round(rnz, 4)}), flush=True)
            if step % 500 == 0:
                rb, ma = rail_probe(model)
                print(json.dumps({"adapt": ADAPT, "step": step,
                                  "rail_by_step": [round(v, 4) for v in rb],
                                  "mean_abs": round(ma, 3)}), flush=True)
            if step % 1000 == 0:
                pc = perclass_reach(model, lesson)
                print(json.dumps({"adapt": ADAPT, "step": step,
                                  "reach_class_acc": pc}), flush=True)
            if pacc >= 0.999 and rnz >= 0.98:
                print("I MILESTONE %s PASS at step %d (planes=%.3f "
                      "reach_nz=%.3f)" % (lesson, step, pacc, rnz), flush=True)
            elif pacc >= 0.999 and rnz >= 0.975:
                print("I MILESTONE %s PASS-MARGINAL at step %d - accepted "
                      "at floor, watched" % (lesson, step), flush=True)
                torch.save(model.state_dict(), STATE)
                break
            elif pacc >= 0.999 and rnz >= 0.95:
                print("I MILESTONE %s PASS-LOW at step %d - accepted low, "
                      "watched" % (lesson, step), flush=True)
                torch.save(model.state_dict(), STATE)
                break
            if step >= 100 and rnz <= best + 0.001:
                stall += 1
            else:
                stall = 0
                best = max(best, rnz)
            if stall >= 8:
                print("I %s STOP-watch at step %d (best %.3f)"
                      % (lesson, step, best), flush=True)
                stall = 0
                best = max(best, rnz)
            if step == CAP_STEPS:
                torch.save(model.state_dict(), STATE)
                if best >= 0.975:
                    print("I MILESTONE %s PASS-MARGINAL at cap (best %.3f)"
                          % (lesson, best), flush=True)
                    break
                if best >= 0.95:
                    print("I MILESTONE %s PASS-LOW at cap (best %.3f) - "
                          "proceed" % (lesson, best), flush=True)
                    break
                print("I %s EXHAUSTED (best %.3f)" % (lesson, best),
                      flush=True)
                sys.exit(1)
    torch.save(model.state_dict(), STATE)
    d2 = fi.depth2_reach_acc(model, random.Random(99))
    print("I DEPTH-2 free-running reach nz acc = %.3f" % d2, flush=True)
    print("I ALL MILESTONES DONE (adapt=%s)" % ADAPT, flush=True)


if __name__ == "__main__":
    main()
