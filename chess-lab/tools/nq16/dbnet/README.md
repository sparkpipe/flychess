# dbnet — labeling + eval services (the operator protocol)

## d20 corpus sweep: `dbnet_d20.py` + `d20_worker.py` (port 8125)

One global cursor: the seq of the next position WITHOUT a d20 label.
`GET /next?n=K` returns `{"items": [{"i": seq, "fen": fen}]}`; the cursor
increments toward the end, wraps to 0, and always skips labeled rows
(LEFT JOIN on labels). A full circle with nothing unlabeled -> `done: true`.
`POST /result {"rows": [{"i","fen","depth","cp"},...]}` spends the
sub-millisecond PK writes (labels insert-or-replace + deepest upsert).
One mutex serializes cursor + writes. No leases, no lockfiles, no reissue.
Work held by a dead worker is re-handed after the wrap; duplicates are
idempotent. Measured: GET 9ms, POST 14ms (20 rows), 256 ARM cores sustained
~15k positions/min (full d20 trajectories, depths 1-20 per search).

## Labeling backbone (legacy, port 8123): `dbnet_service2.py`

Workers POST /submit -> per-node append-only JSONL shard -> one serial
ingest thread batch-inserts into games.db. Idempotent. Needs
`idx_alloc_fen` on allocations(fen, state) or lease closure is a table
scan per fen (this wedged the ingestor once; see git history).

## Trajectory labeler: `depth_worker.py`

One `go depth N` per position; every depth 1..N score parsed from the info
stream (mate -> 10000+min(|m|,900) sign-adjusted, stm-pov verbatim - the
label LAW). Deterministic (ucinewgame per position).

## On-demand UI evals: `on_demand_eval2.py` (localhost:8124)

Pool of SF17 processes behind a free-list queue; each request owns its
process exclusively (no stream interleaving) and runs ucinewgame for
determinism. Note: apple-silicon SF builds with -flto=full SEGFAULT on
quick-mate positions; build with the Makefile patched to -fno-lto
(see experiments/pd-depthexp/sf_crash_battery.py).

## Fleet deploy + telemetry

`deploy_d20v2.sh <sparkN>` — one worker, 16 SF children, restart wrapper
(rc=0 = sweep complete, exits; crash = restart in 10s).
`pdstatus.sh` — per-node progress, aggregate rate, ingest backlog, CPU.
Staged launch law: 1 -> verify -> 2 -> verify -> rest -> verify, with
telemetry run by the assistant at EVERY stage.
