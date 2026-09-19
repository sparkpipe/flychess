"""Overall progress metric for the S6 run: parses every gate table in
v3_s6.out and reports the distance-to-goal trend (the anti-'2-forward-
2-back' metric). Usage: python3 tools/progress_metric.py [logfile]."""
import re
import sys

GOAL = 0.98


def main(path="/home/spec/chess-lab/v3_s6.out"):
    lines = open(path, errors="replace").read().splitlines()
    steps, tables, seen = [], [], {}
    for l in lines:
        m = re.search(r'"step": (\d+)', l)
        f = re.search(r"^S6 FAM (.+)$", l)
        if m:
            steps.append(int(m.group(1)))
        if f:
            cells = dict(kv.split("=") for kv in f.group(1).split())
            seen.setdefault(steps[-1] if steps else 0,
                            {k: float(v) for k, v in cells.items()})
    series = sorted(seen.items())
    if len(series) < 2:
        print("insufficient gate history")
        return

    def deficit(tb):
        return sum(max(0.0, GOAL - v) for v in tb.values())
    d0, d1 = deficit(series[0][1]), deficit(series[-1][1])
    fwd = back = 0
    for k in sorted(series[0][1]):
        vals = [tb[k] for _, tb in series if k in tb]
        fwd += sum(1 for a, b in zip(vals, vals[1:]) if b > a + 0.005)
        back += sum(1 for a, b in zip(vals, vals[1:]) if b < a - 0.005)
    burned = d0 - d1
    verdict = ("FORWARD" if burned > 0.1 else
               "TREADING WATER" if abs(burned) <= 0.1 else "REGRESSING")
    print(f"PROGRESS gates={len(series)} steps={series[0][0]}.."
          f"{series[-1][0]} deficit {d0:.2f}->{d1:.2f} "
          f"(burned {burned:+.2f}) up/down {fwd}/{back} -> {verdict}")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else
         "/home/spec/chess-lab/v3_s6.out")
