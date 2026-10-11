# dbnet — general position-labeling infrastructure

Not specific to any engine or expert set: label any corpus of FENs with
any engine at any depth into a shared SQLite DB, using a fleet of workers
(sparks, a Mac, a 5090 box — anything with the engine binary). The only
assumption is the schema below.

## Schema (games.db)

    positions(seq PK, fen UNIQUE, kind, ref)     -- the corpus
    labels(fen, engine, depth PK, cp, by_node, labeled_at)
    deepest(fen PK, engine, depth, cp, ...)      -- convenience: max depth per fen

Labels are stored **verbatim stm-pov** (side-to-move), mate encoded as
10000+min(|m|,900) with sign. Display layers decode |v|>9000 as mate.

## The sweep service: `dbnet_d20.py` + `d20_worker.py` (port 8125)

One global cursor — the seq of the next position WITHOUT the needed label.

    GET  /next?n=K  -> {"items": [{"i": seq, "fen": fen}, ...], "done": bool}
    POST /result    {"rows": [{"i","fen","depth","cp"}, ...]}  -> stored

The cursor increments toward the end, wraps to 0, skips labeled rows
(LEFT JOIN), and reports `done:true` on a full empty circle. One mutex
serializes cursor + writes. No leases, no lockfiles, no reissue: work a
dead worker never POSTed is re-handed after the wrap, and duplicates are
idempotent (insert-or-replace). Measured: GET 9ms, POST 14ms, 0.44ms of
DB time per position; 256 ARM cores sustained ~15k positions/min with the
pipeline 30x ahead of the fleet. See EVALUATION.md for how this design
compares with two earlier ones.

Run: `python3 dbnet_d20.py`. Worker: `d20_worker.py <node> <cores>
<api-url> <sf-binary>` — one process driving N single-thread SF children,
all depths per search from the info stream.

## On-demand evals: `on_demand_eval2.py` (localhost:8124)

For interactive UIs: a pool of engine processes behind a free-list queue.
Each request owns a process exclusively and runs `ucinewgame` per eval —
deterministic, no stream interleaving. Returns white-pov cp. Serves the
live match UI. Note: apple-silicon SF builds with `-flto=full` segfault on
quick-mate positions — build with `-fno-lto` (repro:
`experiments/pd-depthexp/sf_crash_battery.py`).

## Fleet deploy + telemetry

- `deploy_d20v2.sh <sparkN>` — one worker, 16 SF children, restart wrapper
  (rc=0 = sweep complete, exits; crash = restart in 10s).
- `pdstatus.sh` — per-node progress, aggregate rate, backlog, per-node CPU.
- Staged launch law: 1 -> verify -> 2 -> verify -> rest -> verify, running
  telemetry at EVERY stage.

## Legacy (retired, kept for history)

`dbnet_service*.py`, `dbnet_worker*.py`, `shard_*` — the lease/JSONL-shard
era, superseded by the cursor protocol. See EVALUATION.md for the post-
mortem. `depth_worker.py` (fen-shard input, d1..N trajectories) remains
useful when the corpus is a file rather than the positions table.
