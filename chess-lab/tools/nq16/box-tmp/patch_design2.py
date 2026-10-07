p = "/srv/workspace/flychess/src/chess-lab/DESIGN-MOE.md"
t = open(p).read()

old1 = """**Training corpus rules (the 100% router match):**
- expertA: trains from its stable positions THROUGH the captures, extending to
  (but not including) the first expertB position. The trade decision is
  expertA knowledge — it learns what to trade and when.
- expertB: trains FROM the first ply of its 3-ply stability window — the SAME
  position expertA corpus ends at (the boundary ply belongs to both).
  ExpertB starts its training at the first position it will actually route to."""
new1 = """**Training corpus rules (operator ruling 2026-10-03, 2-ply overlap):**
Let c = the first position of the last material change (post-capture).
- ply <= c:      expertA ONLY (stable positions + all capture positions)
- c+1, c+2:     BOTH expertA and expertB (the 2 overlapped positions)
- ply >= c+3:   expertB ONLY (the 3rd stable ply settles the handoff)
- expertA = domain of the last pre-change stable position; expertB = domain
  of the post-trade material. The trade decision stays expertA knowledge."""
assert old1 in t, "anchor1"
t = t.replace(old1, new1)

old3 = """**Training data**: DEGM pools (24,579 positions with ground-truth labels)
as VERIFIER + all OTB positions passing the detector + verified endgame
puzzles. The book positions are ground truth labels (cat=win/draw, best
move, DTZ), not bulk corpus."""
new3 = """**Training data**: DEGM book extraction = 24,579 unique labeled positions;
both-side expansion ~49k (the "48k"); current both-side dvoretsky bin holds
56,163 records (book both-side + verified endgame puzzles). The 1,601
degm_pools jsonl lines are chapter ROOT positions only. The 953k
expert_bins/dvoretsky.bin is the OLD OTB-detector bin (dump-era, quarantined
family — NOT book data). Book positions are ground truth labels (cat=win/draw,
best move, DTZ), not bulk corpus; OTB detector positions + verified endgame
puzzles are the bulk."""
assert old3 in t, "anchor3"
t = t.replace(old3, new3)

# append anti-band ruling after the sampling line of the v2 section
anchor = "**Sampling**: signal side, every other ply, stm-normalized; all"
addition = """**Anti-engine bands (ruling 2026-10-03)**: win->55, 55->40, 45->30.
win->55 starts at the FIRST time wp > 70, includes ALL plies above 70 and
the decline, closes at wp <= 55 — the entire sequence is suspect and trains
anti. (Replaces the separate 70->55 band.)

"""
assert anchor in t, "anchor4"
t = t.replace(anchor, addition + anchor)
open(p, "w").write(t)
print("DESIGN PATCHED (3 edits)")
