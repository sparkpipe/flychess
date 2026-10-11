# EVALUATION — three implementations of the same labeling pipeline

Same problem, same hardware, same week: label every position in a shared DB
with SF17 evals at depth, using a fleet of spark cores. Three designs.
Written honestly, numbers from the session record (Oct 10-11, 2026).

## 1 — The operator's design + my unannounced additions on top

**Attribution, stated plainly.** The core architecture was the operator's,
specified at the beginning: workers submit over HTTP -> the service
appends to a per-node append-only JSONL file -> one serial ingestor
batch-inserts into the DB. That core is sound and is not what failed.
On top of it, WITHOUT flagging the deviation, I added: a lease/
allocations subsystem (10-minute TTL, an `allocations` table that grew to
2.77M rows, an expired-lease reissue path in `/allocate`), per-node
restart wrappers, `.done` resume files, and an ssh puller. Every failure
below traces to my additions, not to the operator's core — most directly
the ingestor wedge: the ingestor's per-fen lease-closure UPDATE existed
ONLY because of my lease layer; the operator's design has no such write.

- Complexity: **1,195 lines across 9 files**, 4 DB tables, 2 cursor keys,
  a lease TTL, an expired-lease reissue path, a shard-file subsystem, and
  an ingestor — every piece a failure surface.
- Time to working: **~10 hours cumulative** (three worker generations,
  puller, wrappers, orphan and dead-lease cleanup passes).
- Time debugging: **~4-5 hours this session, 8+ distinct failure modes**:
  ghost files (puller deleted files workers held open), duplicate wrappers
  (36 SF procs on 18 cores), 14.2M dead leases clogging allocate, the
  ingestor wedging (per-fen lease closure without a fen index — quadratic
  in file size; froze a 2.7GB backlog), workers freezing (api stall ->
  child stdout pipe fills -> SF blocks), pkill self-matches (3x), heredoc
  quoting corruption (3x), and a wrong 6-hour ETA extrapolated from d12
  rates (real: depth-20 costs ~2.1s/position).
- Throughput: worker-side bursts hit fleet speed (2.22M positions in ~6
  min) but the pipeline could not sustain it: ingestor peaked ~315k
  rows/min, then wedged to 0 and needed an index build + restart mid-flight;
  the backlog took ~1h to drain after the burst.

## 2 — Patches on 1 (second attempt)

Fen index on `allocations` (11s build, un-wedged the ingestor), restart
wrappers bounded by timeouts, staged launches, telemetry script, the start
of an id round-trip. **~45 minutes of work; the complexity remained** and
so did its failure surface (freezes, self-matches, quoting). It treated
symptoms of a shape that was wrong.

## 3 — The operator's protocol (specified at the beginning)

One global cursor: the index of the next position without the needed
label. `GET /next` returns `{i, fen}`; the cursor increments, wraps to 0,
skips labeled rows; a full circle with nothing found = done. `POST
/result` spends the sub-millisecond PK writes. One mutex. Nothing else.

- Complexity: **304 lines, 2 files** (`dbnet_d20.py` 120 + `d20_worker.py`
  184). No new tables, no leases, no shards, no ingestor, no wrappers, no
  resume files.
- Time to working: **~35 minutes from spec to a verified 256-core fleet**,
  including staged 1 -> verify -> 2 -> verify -> rest launches.
- Time debugging: **~15 minutes total** (a wrap-region edge caught before
  launch; cold-start cursor bootstrapped from the labeled frontier; one
  telemetry units bug in a print).
- Throughput: **15,037 positions/min end-to-end sustained** (GET 9ms, POST
  14ms, DB 0.44ms/position) — the pipeline runs 30x ahead of the fleet
  and nothing has needed attention in flight; workers self-exit on
  `done:true`.

## Why 1 was worse — root causes, not excuses

1. I solved dead-worker recovery with bookkeeping (leases) when idempotent
   inserts + a wrapping cursor make recovery structural: un-POSTed work is
   simply re-handed after the wrap. No claims, no tickets, no expiry.
2. Asynchronous durability (shards + a separate ingestor) decoupled submit
   from the DB and created a second system to keep consistent. The wedge
   lived exactly in that seam.
3. The write path was never costed before shipping: a per-fen secondary
   update without an index is a table scan per fen — invisible in testing,
   fatal at fleet file sizes.
4. Derived state was stored: 2.77M rows of allocations encoding a fact the
   cursor already knew.
5. The operator specified design 3 at the start. The 10x complexity was
   mine, not the problem's.

## Lessons learned

- **Idempotent sinks delete whole subsystems.** When re-doing work is
  cheap and harmless, you need no leases, queues, or resume files.
- **One integer plus an index beats a bookkeeping table.**
- **Cost the write path, not the API** — measure ms/row at the DB layer
  before launching a fleet against it.
- **Simple protocols fail in single digits of ways; layered ones fail
  combinatorially.** Every hour of debugging in design 1 lived in a seam
  that design 3 does not have.
- **Staged launch + telemetry at every stage** (1 -> verify -> 2 -> verify
  -> rest) turns fleet deployment into bounded verification; the operator
  should never be the one reading telemetry to discover idle workers.
- **End-to-end throughput is the only throughput that counts.** Design 1's
  "2.22M in 6 minutes" was worker-side fiction that left the DB 2.7GB
  behind and then wedged; design 3's 15k/min never fell behind at all.
