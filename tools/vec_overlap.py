"""Weight-overlap analysis across the single-solution delta vectors.

Operator directive: group questions by weight overlap — high overlap =
similar problems, small changes could solve both; low overlap = separate
capacity in the substrate.

Usage: python3 tools/vec_overlap.py [K]     (top-K support, default 1000)
Reads ten_q/vec_{i}.pt + ten_q/report.json; prints the pairwise matrix.
"""
import sys
import os
import json

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import torch

OUT = os.environ.get("TENQ_OUT", "/home/spec/chess-lab/ten_q")


def flat(d):
    return torch.cat([v.flatten() for v in d.values()])


def topk_idx(d, k):
    f = flat(d)
    return torch.topk(f.abs(), k).indices


def main():
    k = int(sys.argv[1]) if len(sys.argv) > 1 else 1000
    rep = json.load(open(f"{OUT}/report.json"))
    n = len(rep)
    vecs, sups = {}, {}
    for i in range(n):
        d = torch.load(f"{OUT}/vec_{i}.pt", map_location="cpu")
        vecs[i] = flat(d)
        sups[i] = set(topk_idx(d, k).tolist())

    print(f"pairwise: full-cosine / top{k}-support-overlap "
          f"(|A&B|/{k}) / top{k}-restricted cosine")
    hdr = "     " + "".join(f"{j:>13}" for j in range(n))
    print(hdr)
    best, worst = None, None
    for i in range(n):
        row = f"{i:>3}  "
        for j in range(n):
            if i == j:
                row += "            -"
                continue
            a, b = vecs[i], vecs[j]
            cos = float(torch.dot(a, b) /
                        (a.norm() * b.norm() + 1e-12))
            ov = len(sups[i] & sups[j])
            # cosine restricted to the union of the two supports
            u = sorted(sups[i] | sups[j])
            au, bu = a[u], b[u]
            cosu = float(torch.dot(au, bu) /
                         (au.norm() * bu.norm() + 1e-12))
            row += f"  {cos:+.2f}/{ov:4d}/{cosu:+.2f}"
            m = (cos, ov, cosu, i, j)
            if best is None or cosu > best[2]:
                best = m
            if worst is None or cosu < worst[2]:
                worst = m
        print(row)
    fp = lambda r: r["first_pass_step"]
    print(json.dumps({
        "most_similar": {"pair": [best[3], best[4]],
                         "full_cos": round(best[0], 4),
                         "support_overlap": best[1],
                         "support_cos": round(best[2], 4)},
        "most_orthogonal": {"pair": [worst[3], worst[4]],
                            "full_cos": round(worst[0], 4),
                            "support_overlap": worst[1],
                            "support_cos": round(worst[2], 4)},
        "first_pass": {i: fp(r) for i, r in enumerate(rep)}},
        indent=1))


if __name__ == "__main__":
    main()
