"""Evaluate r2 and r3 expert checkpoints vs SF-d20 on their domain positions."""
import sys, os, glob, json, random
sys.path.insert(0, "tools")
sys.path.insert(0, "/srv/workspace/flychess/src/nnue-pytorch")
import numpy as np
import torch
import extract_stack_features as E
import data_loader
from expert_evaluator import load_reference
from model.nnue import NNUEModel
from model.config import NNUELightningConfig

R = "/mnt/cold-raid6/chess-audit"
print("loading reference...", flush=True)
by_domain = load_reference(max_positions=200000)

def eval_ckpt(round_name, expert, fens):
    import extract_stack_features as E
    E.RUNS = f"{R}/{round_name}"
    net = E.load_expert(expert)
    net = net.to("cuda").eval()
    evals = np.zeros(len(fens))
    BATCH = 256
    for st in range(0, len(fens), BATCH):
        chunk = fens[st:st+BATCH]
        bs = data_loader.get_sparse_batch_from_fens(
            "Full_Threats+PP_3Wide+HalfKAv2_hm", chunk,
            [0]*len(chunk), [1]*len(chunk), [0]*len(chunk))
        us, them, wi, bi, o, sc, pc = bs.contents.get_tensors("cuda")
        with torch.no_grad():
            vs = net.forward(us, them, wi, bi, pc)
        evals[st:st+len(chunk)] = [float(v) * net.quantization.nnue2score for v in vs]
        data_loader.destroy_sparse_batch(bs)
    return evals

rng = random.Random(42)
targets = [
    ("balanced_l0", f"{R}/runs2/balanced_l0/lightning_logs/version_0/checkpoints/last.ckpt",
                    f"{R}/runs3/balanced_l0/lightning_logs/version_0/checkpoints/last.ckpt"),
    ("balanced_l1", f"{R}/runs2/balanced_l1/lightning_logs/version_0/checkpoints/last.ckpt",
                    f"{R}/runs3/balanced_l1/lightning_logs/version_0/checkpoints/last.ckpt"),
    ("balanced_l2", f"{R}/runs2/balanced_l2/lightning_logs/version_0/checkpoints/last.ckpt",
                    f"{R}/runs3/balanced_l2/lightning_logs/version_0/checkpoints/last.ckpt"),
    ("nvb",         f"{R}/runs2/nvb/lightning_logs/version_0/checkpoints/last.ckpt",
                    f"{R}/runs3/nvb/lightning_logs/version_0/checkpoints/last.ckpt"),
    ("oppb",        f"{R}/runs2/oppb/lightning_logs/version_0/checkpoints/last.ckpt",
                    f"{R}/runs3/oppb/lightning_logs/version_0/checkpoints/last.ckpt"),
    ("exchanges",   f"{R}/runs2/exchanges/lightning_logs/version_0/checkpoints/last.ckpt",
                    f"{R}/runs3/exchanges/lightning_logs/version_0/checkpoints/last.ckpt"),
]

results = []
for expert, r2ck, r3ck in targets:
    positions = by_domain.get(expert, [])
    if len(positions) < 50:
        continue
    sample = rng.sample(positions, min(1500, len(positions)))
    fens = [f for f, c in sample]
    sf_cps = np.array([c for f, c in sample], dtype=np.float64)

    for rname, ck in [("r2", r2ck), ("r3", r3ck)]:
        if not os.path.exists(ck):
            continue
        try:
            evals = eval_ckpt("runs2" if rname == "r2" else "runs3", expert, fens)
            cp_err = np.abs(evals - sf_cps)
            corr = float(np.corrcoef(evals, sf_cps)[0, 1])
            r = {"expert": expert, "round": rname, "n": len(fens),
                 "cp_mae": round(float(cp_err.mean()), 1),
                 "cp_p95": round(float(np.percentile(cp_err, 95)), 1),
                 "cp_corr": round(corr, 4)}
        except Exception as e:
            r = {"expert": expert, "round": rname, "status": str(e)[:80]}
        results.append(r)
        print(json.dumps(r), flush=True)

json.dump(results, open(f"{R}/expert_evals/r2_r3_comparison.json", "w"), indent=1)
print("DONE", flush=True)
