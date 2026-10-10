# nQ.16B — Expert Taxonomy, Router Order, and Data Sources

Status: OPERATOR-APPROVED composition (2026-10-05). 16 experts.
Router = pure function of position; first match wins; NO catch-all domain —
every asymmetric material vector resolves to a named pattern or piece_down,
anything unmatched is a router bug and must hardfail in the audit.

## Router order (check top to bottom)

| # | Expert | Domain (plain language) | Routing condition |
|---|--------|------------------------|-------------------|
| 1 | tb | Tablebase ground truth — syzygy-proven | total pieces ≤ 5 (kings included) |
| 2 | mvr | The exchange: a rook against one minor, anything else equal | residue R v {N or B} |
| 3 | rv2m | A rook against exactly two minors | residue R v {NN, NB, BB} |
| 4 | qvmat | A queen against two or three pieces | residue Q v {2 or 3 of R/N/B} |
| 5 | nvb | Knight vs bishop family (incl. 2N v 2B) | residue N v B, or 2N v 2B |
| 6 | piece_down | Down a piece or more, no named pattern | any other residue: one side has fewer pieces |
| 7 | oppb | Opposite-colored bishops (exactly 1B v 1B, opposite colors) | equal material, opposite bishop-color majorities |
| 8 | dv_Q | Sparse endgames with queens (≤8 non-pawns, ≤2Q) | npp≤8 & Q≥1 |
| 9 | dv_R | Sparse endgames with rooks, no queens | npp≤8 & Q=0 & R≥1 |
| 10 | dv_rest | Sparse endgames, minors and pawns only | npp≤8 & Q=0 & R=0 |
| 11 | op_pawnimb | Opening pawn imbalances (gambits): equal pieces, pawn count differs | fullmove < 15 & pawns unequal |
| 12 | op_even_l0 | Even opening, no locked center files | fm<15, pawns equal, 0 locks |
| 13 | op_even_l1 | Even opening, one locked center file | fm<15, pawns equal, 1 lock |
| 14 | op_even_l2+ | Even opening, two or more locked center files | fm<15, pawns equal, ≥2 locks |
| 15 | mg_unsafe | Middlegame with exposed kings: opposite castling, or own shield ≤ 1 | developed, unsafe king |
| 16 | mg_safe | All other middlegames (castled symmetric, one castled, none castled) | fallback |

Stability rulings baked in: no side-to-move keys anywhere (the old pd_up/pd_down
and mg my/other splits alternated experts every ply); routing keys are material
residue, piece counts, castling state, center locks, move number only.

## Data sources (all routed by the same router)

| Source | Volume | Label |
|--------|--------|-------|
| Game segments (decisive, climber side) | ~2.5M (full corpus) | fleet d12 cp |
| Game segments (draws, climber side) | ~8.9M | fleet d12 cp |
| Puzzles (ALL, uncapped) | 6,100,952 | SF d12 cp |
| DEGM book (degm.tsv) | 56,163 | book cp (÷100) |
| Syzygy tablebase (tb.tsv) | 12,547 | wdl → cp via wp-model inversion |
| Miniatures (winner-side, move 1 to finish, NO grading) | in progress (~2.8M+) | SF d12 cp (second pass) |

Record format: 40-byte nodchip (position, score clamp ±30000, move, ply, 0, 0).
Train/val split 95/5 by source id hash.
Output: /extnvme/active/train16/<expert>.{train,val}.bin

## Audit gate (before training)

Independent subagent re-derives the routing for sampled positions from every
bin and verifies category membership against this document; any mismatch or
unclassified position fails the build.
