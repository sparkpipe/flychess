"""TC CURRICULUM DRIVER — every integer TC 1..60, 1920 games per checkpoint.

Per TC step (all logged to selfplay_rl/curriculum/driver.log):
  1. push current RL head to 15 sparks (CPU-only, concurrency 16 games/node)
  2. fleet self-play: 960 Chess960 openings x both colors at st=<tc>s
  3. collect PGNs, train head (rl_iter.py --from-dir, bootstraps from prev)
  4. Elo checkpoint on the 5090: h2h vs previous head @1s + e2500/e2850 @1s
     (every 10th TC adds e2850 @10s)
  5. append results to curriculum/results.jsonl, advance state

State: curriculum/state.json {next_tc}; head chain lives in rl_iter's state.json.
Resume: just rerun this script — it picks up at next_tc.
"""
import json, os, re, subprocess, sys, time

R = "/mnt/cold-raid6/chess-audit"
SP = f"{R}/selfplay_rl"
CU = f"{SP}/curriculum"
FORK = "/srv/workspace/flychess/src/Stockfish/src/stockfish"
FC = "/srv/workspace/flychess/src/arena/fastchess-linux-x86-64/fastchess"
TOOLS = "/srv/workspace/flychess/src/chess-lab/tools"
NODES = ["spark0", "spark1", "spark2", "spark3", "spark4", "spark5",
         "spark6", "spark7", "spark8", "spark9",
         "sparka", "sparkb", "sparkc", "sparkd", "sparke"]

RUN_SHARD = """#!/bin/bash
MT=${1:-1}; PGN=${2:-games_cur.pgn}
cd ~/flychess
NETS=$HOME/flychess/nets
B=$HOME/flychess/Stockfish/src/stockfish
OPTS="option.EvalFile=$NETS/balanced_l0.nnue option.EvalFile2=$NETS/balanced_l1.nnue option.EvalFile3=$NETS/balanced_l2.nnue option.EvalFile4=$NETS/balanced_l3.nnue option.EvalFile5=$NETS/nvb.nnue option.EvalFile6=$NETS/nvr.nnue option.EvalFile7=$NETS/bvr.nnue option.EvalFile8=$NETS/rv2m.nnue option.EvalFile9=$NETS/qvmat.nnue option.EvalFile10=$NETS/oppb.nnue option.EvalFile11=$NETS/dvoretsky.nnue option.EvalFile12=$NETS/exchanges.nnue option.EvalFile13=$NETS/tactics.nnue option.StackHead=$HOME/flychess/head.evh"
exec ./fastchess -engine name=sp_a cmd=$B $OPTS \\
  -engine name=sp_b cmd=$B $OPTS \\
  -each proto=uci st=$MT timemargin=200 -rounds 64 -repeat \\
  -openings file=shard.epd format=epd order=sequential \\
  -pgnout file=$PGN -concurrency 10
"""

ELO_OPTS = None  # built at startup


def log(msg):
    line = f"[{time.strftime('%m-%d %H:%M:%S')}] {msg}"
    print(line, flush=True)
    with open(f"{CU}/driver.log", "a") as f:
        f.write(line + "\n")


def sh(cmd, timeout=None):
    return subprocess.run(cmd, shell=True, capture_output=True, text=True, timeout=timeout)


def push_runner_and_shards():
    open(f"{CU}/run_shard.sh", "w").write(RUN_SHARD)
    for n in NODES:
        sh(f"scp -q {CU}/run_shard.sh {n}:~/flychess/run_shard.sh", timeout=60)


def push_head():
    for n in NODES:
        sh(f"scp -q {SP}/head_rl_current.evh {n}:~/flychess/head.evh", timeout=120)


def fleet_generate(tc):
    pgn = f"games_tc{tc}.pgn"
    for n in NODES:
        try:
            sh(f"ssh -f -n {n} 'pkill -x fastchess; cd ~/flychess "
               f"&& rm -f {pgn} && nohup ./run_shard.sh {tc} {pgn} > fc_tc{tc}.log 2>&1 &'",
               timeout=90)
        except subprocess.TimeoutExpired:
            log(f"  tc{tc}: LAUNCH TIMEOUT on {n} (continuing)")
    # expected: ~128 games/node, ~105s/game at tc=1, 16 concurrent
    est = max(600, tc * 105 * 128 / 16 * 1.3 + 600)
    deadline = time.time() + est * 2.5 + 1800
    while time.time() < deadline:
        time.sleep(max(60, tc * 30))
        running = 0
        for n in NODES:
            try:
                c = sh(f"ssh -o ConnectTimeout=10 {n} 'pgrep -x fastchess | wc -l'",
                       timeout=30).stdout.strip()
                if c not in ("0", ""):
                    running += 1
            except subprocess.TimeoutExpired:
                running += 1  # unreachable node counts as busy; collect() decides later
        if running == 0:
            return True
        log(f"  tc{tc}: {running} nodes still generating")
    log(f"  tc{tc}: TIMEOUT waiting for fleet")
    return False


def fleet_collect(tc):
    gd = f"{CU}/tc{tc}"
    os.makedirs(gd, exist_ok=True)
    total = 0
    for n in NODES:
        sh(f"scp -q {n}:~/flychess/games_tc{tc}.pgn {gd}/{n}.pgn", timeout=300)
        c = sh(f"grep -c Result {gd}/{n}.pgn || true").stdout.strip()
        total += int(c) if c.isdigit() else 0
    return total


def train(tc, gd):
    out = sh(f"cd {TOOLS}/.. && python3 tools/rl_iter.py {tc} --from-dir {gd}",
             timeout=7200).stdout
    m = re.search(r"weight change L2: ([0-9.]+)", out)
    change = float(m.group(1)) if m else -1.0
    it = json.load(open(f"{SP}/state.json"))["iteration"]
    return change, f"{SP}/head_rl_iter{it}.evh", it


def pgn_score(path, first):
    txt = open(path).read() if os.path.exists(path) else ""
    games = re.findall(r'\[White "(\w+)"\]\n\[Black "(\w+)"\]\n\[Result "([^"]+)"\]', txt)
    pts = 0.0
    for wh, bl, res in games:
        if wh == first:
            pts += 1.0 if res == "1-0" else 0.0 if res == "0-1" else 0.5
        else:
            pts += 0.0 if res == "1-0" else 1.0 if res == "0-1" else 0.5
    return pts, len(games)


def elo_checkpoint(tc, cur_head, prev_head):
    opts = " ".join(
        [f"option.EvalFile={R}/nets/balanced_l0.nnue"] +
        [f"option.EvalFile{i + 2}={R}/nets/{e}.nnue"
         for i, e in enumerate(["balanced_l1", "balanced_l2", "balanced_l3", "nvb", "nvr",
                                "bvr", "rv2m", "qvmat", "oppb", "dvoretsky",
                                "exchanges", "tactics"])])
    ed = f"{CU}/tc{tc}"
    matches = [
        ("h2h", f"-engine name=cur cmd={FORK} {opts} option.StackHead={cur_head} "
                f"-engine name=prev cmd={FORK} {opts} option.StackHead={prev_head}", "cur", 1),
        ("e2500", f"-engine name=cur cmd={FORK} {opts} option.StackHead={cur_head} "
                  f"-engine name=e2500 cmd=/usr/games/stockfish option.UCI_LimitStrength=true option.UCI_Elo=2500", "cur", 1),
        ("e2850", f"-engine name=cur cmd={FORK} {opts} option.StackHead={cur_head} "
                  f"-engine name=e2850 cmd=/usr/games/stockfish option.UCI_LimitStrength=true option.UCI_Elo=2850", "cur", 1),
    ]
    if tc % 10 == 0:
        matches.append(("e2850_s10", f"-engine name=cur cmd={FORK} {opts} option.StackHead={cur_head} "
                       f"-engine name=e2850 cmd=/usr/games/stockfish option.UCI_LimitStrength=true option.UCI_Elo=2850", "cur", 10))
    procs = []
    for name, spec, first, stc in matches:
        cmd = (f"{FC} {spec} -each proto=uci st={stc} timemargin=200 -rounds 3 -repeat "
               f"-openings file={R}/3open.epd format=epd "
               f"-pgnout file={ed}/elo_{name}.pgn > {ed}/elo_{name}.log 2>&1")
        procs.append((name, first, subprocess.Popen(cmd, shell=True)))
    res = {}
    for name, first, p in procs:
        p.wait()
        res[name] = pgn_score(f"{ed}/elo_{name}.pgn", first)
    return res


def main():
    os.makedirs(CU, exist_ok=True)
    state_f = f"{CU}/state.json"
    state = json.load(open(state_f)) if os.path.exists(state_f) else {"next_tc": 2}
    push_runner_and_shards()

    for tc in range(state["next_tc"], 61):
        log(f"=== TC {tc}s checkpoint start ===")
        prev_it = json.load(open(f"{SP}/state.json"))["iteration"]
        prev_head = f"{SP}/head_rl_iter{prev_it}.evh"

        push_head()
        fleet_generate(tc)
        total = fleet_collect(tc)
        log(f"  tc{tc}: collected {total} games")
        if total < 1500:
            log(f"  tc{tc}: TOO FEW GAMES ({total}), aborting driver for review")
            return

        change, cur_head, it = train(tc, f"{CU}/tc{tc}")
        log(f"  tc{tc}: iter{it} trained, weight change {change:.4f}")

        res = elo_checkpoint(tc, cur_head, prev_head)
        log(f"  tc{tc}: elo " + " ".join(f"{k}={v[0]}/{v[1]}" for k, v in res.items()))

        with open(f"{CU}/results.jsonl", "a") as f:
            f.write(json.dumps({"tc": tc, "games": total, "iter": it,
                                "change": round(change, 4), "elo": res,
                                "head": cur_head, "prev": prev_head}) + "\n")
        state["next_tc"] = tc + 1
        json.dump(state, open(state_f, "w"))
        log(f"=== TC {tc}s checkpoint done -> next {tc + 1} ===")
    log("CURRICULUM COMPLETE through tc=60")


if __name__ == "__main__":
    main()
