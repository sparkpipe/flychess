# DESIGN LAW — simplest version that has a chance of working

Operator ruling, 2026-10-11: "I do not mind if you improve on my spec…
ADDING new complexity is NOT an improvement. Making it more simple is the
improvement. Start with the simplest version that has a chance of working,
then iterate with incremental complexity. NEVER start with a battleship
airrier design if an inflatable raft will do the job."

## The laws

1. **The spec is the complexity ceiling, not the floor.** A specified
   design's size is the budget. Deliverables come back shorter or equal;
   anything longer requires a ruling BEFORE it is written.
2. **Raft first, measured.** Build and run the simplest version that has
   a chance of working, with numbers, before any improvement exists.
   Improvements are judged against that measured baseline.
3. **Evidence-gated complexity.** A new layer may be added only for a
   failure that has actually been observed, and the addition must name
   that failure. Speculative failure modes do not qualify — and an
   observed failure that idempotency already absorbs DISQUALIFIES the
   layer (e.g., a dead worker whose positions the cursor wrap re-hands).
4. **Subtract-first fixes.** When something breaks, the first candidate
   fix is removing code. Patching within a wrong shape is not a fix.
5. **Every PR answers "what did you NOT build?"** Name the simpler
   version rejected and why it cannot work. No answer = design not found.

## Why this is the safe direction, not the risky one

Durability in this system comes from idempotent writes + retry — one-line
properties. Bookkeeping for failure recovery (leases, queues, resume
files) is not safety; it is a second system whose seams become the
failure surface. The cursor protocol is kill-any-process-anytime safe
with zero bookkeeping: see EVALUATION.md for the measured cost of
believing otherwise (1,195 LOC and ~5 hours of debugging vs 304 LOC and
15 minutes, at higher end-to-end throughput for the simple version).

## The objective function (operator, 2026-10-11): score = value / LOC^3

Optimize for solution value divided by the cube of code size. Doubling
code requires 8x delivered value to break even; 10x code requires 1000x.
Measured example from this pipeline: lease/shard (1,195 LOC, ~0 sustained
value once wedged) vs cursor protocol (304 LOC, 15,037 pos/min) — the
simple version scores ~60x higher even granting the complex one equal
throughput, and infinitely higher against its real zero.

Counting rules:
- LOC = net lines of code we own (services, workers, glue), normal
  formatting, standard idioms — not vendored engines/trainers.
- Obfuscation/golfing counts as a VALUE regression, not a size win; the
  metric punishes moving parts, not readable lines.
- Every PR reports LOC, LOC^3, the measured value, and the score — before
  review. Deleting code at equal value is the highest-leverage move
  available (halving size = 8x score).

## Milestone-PR law (operator, 2026-10-11: "you never make PR to evaluate before I pester you")

PRs are opened at every working milestone, BEFORE being asked — they are
the evaluation checkpoint, not an afterthought. Tiers:

- under ~50 LOC: implement, PR immediately after.
- ~50-300 LOC: implement, PR with the template numbers filled.
- over ~300 LOC or any new subsystem: open the PR with the DESIGN first
  (raft spec + LOC^3 math + the simpler version not built), get the
  ruling, then write the code into that PR.

The PR template (.github/PULL_REQUEST_TEMPLATE.md) makes the simplicity
case structural: value, LOC, LOC^3, score, and the rejected simpler
version are filled in before review, so complexity can only reach the
operator as an explicit, numbered deviation they can reject in one click.
