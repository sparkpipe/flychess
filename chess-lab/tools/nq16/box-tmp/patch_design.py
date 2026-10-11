import sys
p = "/srv/workspace/flychess/src/chess-lab/DESIGN-MOE.md"
t = open(p).read()
start = t.index("### The Exchanges expert = the Transition State")
end = t.index("### The TB expert")
new = """### The Transition State — NO exchanges expert (operator ruling 2026-10-03)

There is NO exchanges expert. The transition between stable domain experts is
handled by the ROUTER, not by a dedicated net. 12 experts total:
balanced_l0..l3, nvb, nvr, bvr, rv2m, qvmat, oppb, dvoretsky, tb.
(The old slot-13 tactics net is replaced by TB; exchanges is deleted.)

1. Position routes to expertA (stable domain)
2. Captures occur (material changing) — expertA STAYS ACTIVE through them
3. After 3 consecutive plies with no material change — the router hands the
   position to expertB (whatever the post-trade configuration routes to)

**Training corpus rules (the 100% router match):**
- expertA: trains from its stable positions THROUGH the captures, extending to
  (but not including) the first expertB position. The trade decision is
  expertA knowledge — it learns what to trade and when.
- expertB: trains FROM the first ply of its 3-ply stability window — the SAME
  position expertA corpus ends at (the boundary ply belongs to both).
  ExpertB starts its training at the first position it will actually route to.

**Realtime routing**: the 3-ply rule is the online detector (no lookahead).
Precedence: TB (men<=5, absolute) first; during the unstable window expertA
stays active; once stable, dvoretsky detector (npp<=8, Q<=2), then the domain
cascade by current material.

**Sampling**: every ply, side-to-move normalized per corpus side; all
positions carry game provenance (game_id, ply, segment band, signal type).

"""
t = t[:start] + new + t[end:]
open(p, "w").write(t)
print("PATCHED")
