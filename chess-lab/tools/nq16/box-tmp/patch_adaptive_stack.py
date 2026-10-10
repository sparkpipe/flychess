import sys

# ============ fly_curriculum.py ==========================================
p = "/home/spec/chess-lab/fly_curriculum.py"
s = open(p).read()
n0 = len(s)

def rep(old, new, count=1):
    global s
    assert s.count(old) == count, (s.count(old), old[:80])
    s = s.replace(old, new)

# LAST_ACC registry
rep("""def eval_piece(model, piece, n=64, seed=5000, stage=1):""",
    """LAST_ACC = {}                         # (stage, piece-name) -> last pair


def eval_piece(model, piece, n=64, seed=5000, stage=1):""")

# record last battery accuracy in eval_piece
rep("""                    failures.append((b.fen(), sq, lt, it, round(gl - gi, 3)))
    return ok / max(tot, 1), failures""",
    """                    failures.append((b.fen(), sq, lt, it, round(gl - gi, 3)))
    LAST_ACC[(stage, pname(piece))] = ok / max(tot, 1)
    return ok / max(tot, 1), failures""")

# settling-based adaptive depth: retino branch (byte-exact anchor)
rep("""            for _ in range(PROP_STEPS):
                a = (1 - LEAK) * a + LEAK * (self.WT @ a)
                if os.environ.get("ANORM", "0") == "1":
                    a = a / (a.abs().mean() + 1e-6)                         * float(os.environ.get("ANORM_T", "2.0"))
                else:
                    a = torch.clamp(a, -CAP, CAP)
            return a""",
    """            max_steps = int(os.environ.get("MPROP", str(PROP_STEPS)))
            eps = float(os.environ.get("MEPS", "0.02"))
            _prev = None
            for _ in range(max_steps):
                a = (1 - LEAK) * a + LEAK * (self.WT @ a)
                if os.environ.get("ANORM", "0") == "1":
                    a = a / (a.abs().mean() + 1e-6)                         * float(os.environ.get("ANORM_T", "2.0"))
                else:
                    a = torch.clamp(a, -CAP, CAP)
                if _prev is not None and float(
                        (a - _prev).abs().mean()) < eps:
                    break
                _prev = a
            return a""")

# settling-based adaptive depth: non-retino branch (byte-exact anchor)
rep("""        for _ in range(PROP_STEPS):
            a = (1 - LEAK) * a + LEAK * (self.WT @ a)
            if os.environ.get("ANORM", "0") == "1":
                a = a / (a.abs().mean() + 1e-6)                     * float(os.environ.get("ANORM_T", "2.0"))
            else:
                a = torch.clamp(a, -CAP, CAP)
        return a""",
    """        max_steps = int(os.environ.get("MPROP", str(PROP_STEPS)))
        eps = float(os.environ.get("MEPS", "0.02"))
        _prev = None
        for _ in range(max_steps):
            a = (1 - LEAK) * a + LEAK * (self.WT @ a)
            if os.environ.get("ANORM", "0") == "1":
                a = a / (a.abs().mean() + 1e-6)                     * float(os.environ.get("ANORM_T", "2.0"))
            else:
                a = torch.clamp(a, -CAP, CAP)
            if _prev is not None and float(
                    (a - _prev).abs().mean()) < eps:
                break
            _prev = a
        return a""")

# LP-weighted replay in the three milestone mixes
rep("""            if prior and rng.random() < 0.35:
                st2, pc2 = rng.choice(prior)""",
    """            if prior and rng.random() < 0.35:
                _w = [max(0.02, 0.98 - LAST_ACC.get((_s, _p), 0.9))
                      for _s, _p in prior]
                st2, pc2 = rng.choices(prior, weights=_w)[0]""")
rep("""            if prior and rng.random() < 0.35:
                st3, pc3 = rng.choice(prior)""",
    """            if prior and rng.random() < 0.35:
                _w = [max(0.02, 0.98 - LAST_ACC.get((_s, _p), 0.9))
                      for _s, _p in prior]
                st3, pc3 = rng.choices(prior, weights=_w)[0]""")
rep("""            if prior and rng.random() < 0.35:
                st4, pc4 = rng.choice(prior)""",
    """            if prior and rng.random() < 0.35:
                _w = [max(0.02, 0.98 - LAST_ACC.get((_s, _p), 0.9))
                      for _s, _p in prior]
                st4, pc4 = rng.choices(prior, weights=_w)[0]""")

open(p, "w").write(s)
print("curriculum OK, bytes", n0, "->", len(s))

# ============ fly_vision.py ==============================================
q = "/home/spec/chess-lab/fly_vision.py"
t = open(q).read()
m0 = len(t)

def repv(old, new, count=1):
    global t
    assert t.count(old) == count, (t.count(old), old[:80])
    t = t.replace(old, new)

# settling depth
repv("""        for _ in range(cur.PROP_STEPS):
            a = torch.clamp((1 - cur.LEAK) * a + cur.LEAK * (self.WT @ a),
                            -cur.CAP, cur.CAP)
        return a""",
    """        max_steps = int(os.environ.get("VPROP", str(cur.PROP_STEPS)))
        eps = float(os.environ.get("VEPS", "0.02"))
        _prev = None
        for _ in range(max_steps):
            a = torch.clamp((1 - cur.LEAK) * a + cur.LEAK * (self.WT @ a),
                            -cur.CAP, cur.CAP)
            if _prev is not None and float(
                    (a - _prev).abs().mean()) < eps:
                break
            _prev = a
        return a""")

# occupied-bin weighting (25x for +-1 reach codes) in the vision loss
repv("""    w = np.where(np.abs(tg) > 1e-6, 3.0, 1.0).astype(np.float32)""",
    """    w = np.where(np.abs(tg) == 1.0, 25.0,
                 np.where(np.abs(tg) > 1e-6, 3.0, 1.0)).astype(np.float32)""")

# per-piece accuracy registry + LP-weighted replay
repv("""            if passed and rng.random() < 0.35:
                rp = rng.choice(passed)""",
    """            _w = [max(0.02, 0.98 - VLAST_ACC.get(p, 0.9)) for p in passed]
            rp = rng.choices(passed, weights=_w)[0]""")
repv("""def train_step(model, opt, rng, piece, hard=None):""",
    """VLAST_ACC = {}                        # piece -> last full binacc


def train_step(model, opt, rng, piece, hard=None):""")
repv("""            full, nz = eval_battery(model, piece)""",
    """            full, nz = eval_battery(model, piece)
            VLAST_ACC[piece] = full""")

open(q, "w").write(t)
print("vision OK, bytes", m0, "->", len(t))

# ============ fly_imagine.py =============================================
r = "/home/spec/chess-lab/fly_imagine.py"
u = open(r).read()
k0 = len(u)

def repi(old, new, count=1):
    global u
    assert u.count(old) == count, (u.count(old), old[:80])
    u = u.replace(old, new)

# settling depth in the imagination propagate
repi("""        norm = os.environ.get("ANORM", "0") == "1"
        tgt = float(os.environ.get("ANORM_T", "2.0"))
        for _ in range(cur.PROP_STEPS):
            a = (1 - cur.LEAK) * a + cur.LEAK * (self.WT @ a)
            if norm:
                a = a / (a.abs().mean() + 1e-6) * tgt
            else:
                a = torch.clamp(a, -cur.CAP, cur.CAP)
        return a""",
    """        norm = os.environ.get("ANORM", "0") == "1"
        tgt = float(os.environ.get("ANORM_T", "2.0"))
        max_steps = int(os.environ.get("IPROP", str(cur.PROP_STEPS)))
        eps = float(os.environ.get("IEPS", "0.02"))
        _prev = None
        for _ in range(max_steps):
            a = (1 - cur.LEAK) * a + cur.LEAK * (self.WT @ a)
            if norm:
                a = a / (a.abs().mean() + 1e-6) * tgt
            else:
                a = torch.clamp(a, -cur.CAP, cur.CAP)
            if _prev is not None and float(
                    (a - _prev).abs().mean()) < eps:
                break
            _prev = a
        return a""")

# occupied-bin weighting (25x for signed cells) in the imagination loss
repi("""    w = torch.where(torch.abs(tgt_r) > 1e-6, 3.0, 1.0)""",
    """    w = torch.where(torch.abs(tgt_r) == 1.0, 25.0,
        torch.where(torch.abs(tgt_r) > 1e-6, 3.0, 1.0))""")

# replay of previous imagination lessons (30%), LP-weighted by gate distance
repi("""        for step in range(1, 12001):
            train_step(model, opt, rng, lesson)""",
    """        prev = IORDER[:IORDER.index(lesson)]
        I_LAST_ACC = {}
        for step in range(1, 12001):
            train_step(model, opt, rng, lesson)
            if prev and step % 200 == 0:
                for p_ in prev:
                    _, _, rnz_p = fi.eval_battery(model, p_, n=64)
                    I_LAST_ACC[p_] = rnz_p
            if prev and rng.random() < 0.3:
                _wl = [max(0.02, 0.98 - I_LAST_ACC.get(p, 0.9))
                       for p in prev]
                train_step(model, opt, rng, rng.choices(prev, weights=_wl)[0])""")

open(r, "w").write(u)
print("imagine OK, bytes", k0, "->", len(u))
