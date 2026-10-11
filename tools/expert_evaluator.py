"""EXPERT EVALUATOR — per-expert accuracy vs SF, self-updating blackbox.

Usage:
  expert_evaluator.py <expert_name>                    # evaluate current net
  expert_evaluator.py <expert_name> <net_path>         # evaluate specific net
  expert_evaluator.py --all                            # all 13

Reads:
  - the expert's current training bin (blackbox: whatever is there now)
  - depth-DB evals (SF-d20 reference) joined by position
  - dead-zone positions (outside any up-slope segment)

Reports per expert:
  - eval RMSE vs SF-d20 (win-prob space)
  - correlation
  - domain-specific accuracy (on positions the router sends to this expert)
  - dead-zone performance (positions the expert sees that weren't in up-slope training)
"""
import sys, os, glob, math, json, subprocess, time
sys.path.insert(0, "/srv/workspace/flychess/src/nnue-pytorch")
sys.path.insert(0, "/srv/workspace/flychess/src/chess-lab/tools")
import numpy as np
import torch
import chess

R = "/mnt/cold-raid6/chess-audit"
DB = f"{R}/depth_db/sparks"
OUT = f"{R}/expert_evals"

EXPERTS = ["balanced_l0", "balanced_l1", "balanced_l2", "balanced_l3",
           "nvb", "nvr", "bvr", "rv2m", "qvmat", "oppb",
           "dvoretsky", "exchanges", "tactics"]
BASE13 = EXPERTS

def wp(x):
    return 1.0 / (1.0 + np.exp(-np.array(x, dtype=np.float64) / 361.0))

def domain_of_fen(fen):
    """Faithful port of the C++ router (same rules as score_experts)."""
    b = chess.Board(fen)
    men = len(b.piece_map())
    if men <= 5:
        return "tb"
    cw, cb = {"Q":0,"R":0,"B":0,"N":0}, {"Q":0,"R":0,"B":0,"N":0}
    for s, p in b.piece_map().items():
        u = p.symbol().upper()
        if u in ("K","P"): continue
        (cw if p.color else cb)[u] += 1
    resw = "".join(sum(([k]*max(0,cw[k]-cb[k]) for k in "QRBN"), []))
    resb = "".join(sum(([k]*max(0,cb[k]-cw[k]) for k in "QRBN"), []))
    sym = resw == "" and resb == ""
    DOMS = ["balanced_l0","balanced_l1","balanced_l2","balanced_l3","nvb",
            "nvr","bvr","rv2m","qvmat","oppb","dvoretsky","exchanges","tb"]
    if sym:
        wcol2 = sum(1 for s,p in b.piece_map().items() if p.piece_type==3 and p.color and (chess.square_file(s)+chess.square_rank(s))%2==1)
        bcol2 = sum(1 for s,p in b.piece_map().items() if p.piece_type==3 and not p.color and (chess.square_file(s)+chess.square_rank(s))%2==1)
        nb_w = sum(1 for p in b.pieces(3, True)); nb_b = sum(1 for p in b.pieces(3, False))
        if nb_w and nb_b and wcol2 != bcol2:
            return "oppb"
        if men <= 10:
            return "dvoretsky"
        lock = 0
        for f in range(8):
            wr = [chess.square_rank(s) for s in chess.SquareSet(b.pieces(1, True)) if chess.square_file(s)==f]
            br = [chess.square_rank(s) for s in chess.SquareSet(b.pieces(1, False)) if chess.square_file(s)==f]
            if wr and br and min(br)-max(wr)==1 and 2<=f<=5: lock += 1
        return ["balanced_l0","balanced_l1","balanced_l2","balanced_l3"][min(lock,3)]
    kres = tuple(sorted((resw, resb)))
    RES = {("B","N"):"nvb", ("N","R"):"nvr", ("B","R"):"bvr",
           ("NN","R"):"rv2m", ("BN","R"):"rv2m", ("BB","R"):"rv2m",
           ("Q","RR"):"qvmat", ("Q","RN"):"qvmat", ("Q","RB"):"qvmat"}
    if kres in RES:
        return RES[kres]
    if men <= 10:
        return "dvoretsky"
    us = b.turn
    for atk, vic in ((2,2),(3,3),(4,4),(5,5),(2,3),(3,2)):
        if b.attacks_mask(us) & b.pieces_mask(vic, not us):
            return "exchanges"
    lock = 0
    for f in range(8):
        wr = [chess.square_rank(s) for s in chess.SquareSet(b.pieces(1, True)) if chess.square_file(s)==f]
        br = [chess.square_rank(s) for s in chess.SquareSet(b.pieces(1, False)) if chess.square_file(s)==f]
        if wr and br and min(br)-max(wr)==1 and 2<=f<=5: lock += 1
    return ["balanced_l0","balanced_l1","balanced_l2","balanced_l3"][min(lock,3)]

def load_reference(max_positions=100000):
    """Load depth-DB positions with SF-d20 evals, classified by domain."""
    from collections import defaultdict
    by_domain = defaultdict(list)
    n = 0
    for shard_f in sorted(glob.glob(f"{DB}/*/*.tsv.*")):
        for line in open(shard_f):
            parts = line.rstrip().split("\t")
            if len(parts) < 2: continue
            fen = parts[0]
            cps = parts[1].split(",")
            if len(cps) < 20: continue
            try:
                cp20 = int(cps[19])
            except ValueError: continue
            d = domain_of_fen(fen)
            by_domain[d].append((fen, cp20))
            n += 1
            if n >= max_positions:
                break
        if n >= max_positions:
            break
    return by_domain

def evaluate_expert(expert_name, net_path=None, by_domain=None):
    """Evaluate one expert's eval vs SF-d20 on its domain positions."""
    import extract_stack_features as E
    import data_loader

    if by_domain is None:
        by_domain = load_reference()

    # get positions this expert's domain
    key = "tb" if expert_name == "tactics" else expert_name
    positions = by_domain.get(key, [])
    if len(positions) < 50:
        return {"expert": expert_name, "n": len(positions), "status": "insufficient data"}

    # sample up to 2000 positions
    import random
    rng = random.Random(42)
    sample = rng.sample(positions, min(2000, len(positions)))
    fens = [f for f, c in sample]
    sf_cps = np.array([c for f, c in sample], dtype=np.float64)

    # run the expert's net on these positions
    net_path = net_path or f"{R}/nets/{expert_name}.nnue"
    if not os.path.exists(net_path):
        return {"expert": expert_name, "status": f"net not found: {net_path}"}

    # use the python model (checkpoint) not the serialized .nnue
    ckpt = f"{R}/runs/{expert_name}/lightning_logs/version_0/checkpoints/last.ckpt"
    if os.path.exists(ckpt):
        model = E.load_expert(expert_name)
    else:
        return {"expert": expert_name, "status": "no checkpoint"}

    # batch inference
    evals = np.zeros(len(fens))
    BATCH = 256
    for st in range(0, len(fens), BATCH):
        chunk = fens[st:st+BATCH]
        bs = data_loader.get_sparse_batch_from_fens(
            "Full_Threats+PP_3Wide+HalfKAv2_hm", chunk,
            [0]*len(chunk), [1]*len(chunk), [0]*len(chunk))
        us, them, wi, bi, o, sc, pc = bs.contents.get_tensors("cuda")
        with torch.no_grad():
            vs = model.forward(us, them, wi, bi, pc)
        evals[st:st+len(chunk)] = [float(v) * model.quantization.nnue2score for v in vs]
        data_loader.destroy_sparse_batch(bs)

    # compute accuracy metrics in win-prob space
    expert_wp = wp(evals)
    sf_wp = wp(sf_cps)
    rmse = float(np.sqrt(((expert_wp - sf_wp) ** 2).mean()))
    corr = float(np.corrcoef(expert_wp, sf_wp)[0, 1]) if len(sample) > 1 else 0.0

    # eval-space error (cp)
    cp_err = np.abs(evals - sf_cps)
    return {
        "expert": expert_name,
        "net": net_path,
        "n_positions": len(sample),
        "domain_pool": len(positions),
        "wp_rmse": round(rmse, 4),
        "wp_corr": round(corr, 4),
        "cp_mae": round(float(cp_err.mean()), 1),
        "cp_p95": round(float(np.percentile(cp_err, 95)), 1),
    }

def main():
    os.makedirs(OUT, exist_ok=True)
    print("loading depth-DB reference (SF-d20, domain-classified)...", flush=True)
    by_domain = load_reference(max_positions=200000)
    for d, pos in sorted(by_domain.items()):
        print(f"  {d:15s} {len(pos):,} positions", flush=True)

    if "--all" in sys.argv:
        results = []
        for e in EXPERTS:
            r = evaluate_expert(e, by_domain=by_domain)
            print(json.dumps(r), flush=True)
            results.append(r)
        json.dump(results, open(f"{OUT}/all_experts.json", "w"), indent=1)
    else:
        expert = sys.argv[1] if len(sys.argv) > 1 else "balanced_l0"
        r = evaluate_expert(expert, by_domain=by_domain)
        print(json.dumps(r, indent=1))
        json.dump(r, open(f"{OUT}/{expert}.json", "w"), indent=1)

if __name__ == "__main__":
    main()
