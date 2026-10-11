"""Evaluate ranking functions on the MoF routing test data.

Signals per (question, fly): margin (internal), knn (external cosine to
member positions), agree (how many flies made the same pick), modal
(picked the majority answer). Rankings are evaluated by: top-1 fly
correct rate, any-correct within top-k, mean reciprocal rank of a
correct fly, and the answer-level baseline (take the majority answer).
"""
import json
import random

D = json.load(open("/home/spec/chess-lab/singles/mof_test.json"))


def evaluate(key, rev=True, topk=(1, 3, 5, 10)):
    """rank flies by key (desc if rev); top-1 = top fly's correct."""
    t1 = t5 = 0
    rrs = []
    topk_hits = {k: 0 for k in topk}
    for q in D:
        flies = sorted(q["flies"], key=lambda f: -f[key] if rev
                       else f[key])
        corrects = [i for i, f in enumerate(flies)
                    if f["correct"]]
        if corrects:
            t1 += 1 if corrects[0] == 0 else 0
            rrs.append(1.0 / (corrects[0] + 1))
            for k in topk:
                topk_hits[k] += int(corrects[0] < k)
        else:
            rrs.append(0.0)
    n = len(D)
    return {"top1": round(t1 / n, 3),
            **{f"top{k}": round(topk_hits[k] / n, 3) for k in topk},
            "mrr": round(sum(rrs) / n, 3)}


out = {}
out["random"] = evaluate("margin", rev=False)   # arbitrary order proxy
out["margin"] = evaluate("margin")
out["knn"] = evaluate("knn")
out["agree"] = evaluate("agree")
out["agree_then_margin"] = None
# combo: sort by agree desc, margin desc tiebreak
t1 = rrs = 0
rrs = []
t5 = 0
for q in D:
    flies = sorted(q["flies"], key=lambda f: (-f["agree"], -f["margin"]))
    corrects = [i for i, f in enumerate(flies) if f["correct"]]
    if corrects:
        t1 += int(corrects[0] == 0)
        t5 += int(corrects[0] < 5)
        rrs.append(1.0 / (corrects[0] + 1))
    else:
        rrs.append(0.0)
out["agree_then_margin"] = {"top1": round(t1 / len(D), 3),
                            "top5": round(t5 / len(D), 3),
                            "mrr": round(sum(rrs) / len(D), 3)}
# answer-level baselines
modal_correct = sum(1 for q in D if q["modal_correct"])
home_correct = sum(1 for q in D
                   if q["flies"][0]["correct"])  # placeholder replaced below
n_correct_stats = [q["n_correct"] for q in D]
out["majority_answer_correct"] = round(modal_correct / len(D), 3)
out["mean_flies_knowing"] = round(sum(n_correct_stats) / len(D), 2)
out["min_flies_knowing"] = min(n_correct_stats)
out["oracle_coverage"] = round(sum(1 for c in n_correct_stats
                                   if c >= 1) / len(D), 3)
# home-fly rank under knn
hr = []
for q in D:
    flies = sorted(q["flies"], key=lambda f: -f["knn"])
    h = q.get("home_fly")
    if h is not None:
        hr.append([i for i, f in enumerate(flies)
                   if f["fly"] == h][0])
out["home_fly_knn_rank_mean"] = round(sum(hr) / len(hr), 2)
print(json.dumps(out, indent=1))
json.dump(out, open("/home/spec/chess-lab/singles/mof_rankings.json",
                    "w"), indent=1)
