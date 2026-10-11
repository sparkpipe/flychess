#!/usr/bin/env python3
"""Append the operator-corrected expert design to DESIGN-MOE.md."""
s = open("/srv/workspace/flychess/src/chess-lab/DESIGN-MOE.md").read()
add = """

---

## EXPERT DESIGN v2.0 — Operator Rulings 2026-10-02

### The Dvoretsky expert

**Definition**: a dvoretsky position is a position whose material configuration
matches the Dvoretsky Endgame Manual (DEGM). NOT "any position ≤N pieces."
The book's territory (measured from DEGM_Ch1-15 pools, 1601 root positions):

- 49% have rooks (31% have 2+ rooks); only 11% have a queen
- 46% have no Q and no R at all (pure minor/pawn endings)
- Men range 3-22; no-Q positions reach 18 men

**Detector (deterministic, verified against the book)**:
```
is_dvoretsky(pos):
    npp = total non-pawn, non-king pieces (both sides)
    Q   = total queens
    return npp <= 8 and Q <= 2
```
- Recall on DEGM book: 1601/1601 = 100%
- False-positive on OTB balanced-middlegame corpus: 14.2%
- Match on existing OTB dvoretsky bin: 100%

The ~14% overlap with quiet middlegames is expected (those are positions
Dvoretsky's methods DO apply to); the book is the ground truth for training
but the detector is deliberately slightly wider (recall-first).

**Routing precedence**: TB check (men<=5) fires first, then dvoretsky
detector, THEN the remaining domain cascade.

**Training data**: DEGM pools (24,579 positions with ground-truth labels)
as VERIFIER + all OTB positions passing the detector + verified endgame
puzzles. The book positions are ground truth labels (cat=win/draw, best
move, DTZ), not bulk corpus.

### The Exchanges expert = the Transition State

**Concept**: captures change the nature of the position (pawn captures
excepted). The exchanges expert IS the transition between stable domain
experts. It does not own a static territory — it owns the move sequence
during which material is changing.

**Protocol**:
1. Position is in expertA (stable domain, no recent captures)
2. First capture (non-pawn) occurs → router switches to exchanges expert
3. Trades continue (recaptures, counter-trades, sacrifices) → exchanges stays active
4. After 3 consecutive plies with NO material change → router switches to
   expertB (the stable domain that the post-trade material configuration
   routes to)

**Training corpus rules (the 100% router match)**:
- expertA: trains from its stable positions THROUGH the first capture,
  extending to (but not including) the first expertB position. ExpertA
  learns what to trade and when — the trade decision is expertA's knowledge.
- exchanges: trains from the first capture position to the 3rd stable
  expertB ply (inclusive). Every transition from every domain pair feeds
  this corpus. The exchanges expert has seen all exchanges from all positions.
- expertB: trains FROM the first ply of its 3-ply stability window — the
  SAME position expertA's corpus ends at (the boundary ply belongs to both).
  ExpertB starts its training at the first position it will actually route to.

**Realtime routing**: the router fires the exchanges expert on the first
non-pawn capture and switches to expertB after 3 stable plies (no lookahead;
the 3-ply rule is the online detector). In training (offline), the full
transition is known; the corpus boundaries are set so that the training
partition matches what the realtime router will do.

**Sampling**: winner's side, every other ply, against this transition
background. All positions are side-to-move normalized to the winner.

### The TB expert

**Routing**: men <= 5, first check, absolute (unchanged).
**Training data**: Syzygy 5M positions (COMPLETE) + ≤5-piece puzzle positions.
The current slot-13 net (tactics-corpus trained) is to be replaced.

### Data curation law

- Every training position must be of high quality
- The DUMP (tactics_dump, round3_train, raw tactics bins) is QUARANTINED —
  toxic material, no slope filter, excluded from all training
- New mining: positive eval slopes (4 bands) in games with 2400+ opposition.
  The qualifying positions are those where a player GAINS eval against a
  2400+ opponent (how to beat strong players, not how strong players beat weak).
  Postal/correspondence games included (postal 2000 ≈ OTB 2400 quality; filter
  out amateur-vs-amateur). Winner's side, every other ply.

### Win-probability calibration (open)

The wp(cp) = sigmoid(cp/361) mapping needs empirical verification against
actual game outcomes. Testable: bin positions by wp bucket, compare to
observed win rate in the 126M-line OTB dump.
"""
open("/srv/workspace/flychess/src/chess-lab/DESIGN-MOE.md", "w").write(s + add)
print("DESIGN-MOE.md updated")
