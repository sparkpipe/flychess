# flychess — the fly-brain chess project

Drosophila connectome (MaleCNS 188,778 neurons) learns chess.

## Architecture
- Features (color-blind, STM-relative) -> sensory/CX injection -> 2-step
  propagation through the real connectome -> 4096 move-slot readout
- FlyLeela engine: PUCT search over fly policy/value, menu-comparison value,
  opponent-trap layer, stance controller (nurture/chaos/desperation)
- Swarm mode: per-fly sensory dropout = diverse simulated flies

## Training: staged, gated curriculum (nothing advances below 100%)
1. piece movement instincts (open board)
2. two pieces: blocking / capture
3. three pieces: protection
4. king safety, pins, forks, discovered attacks
5. basic mates (KQvK, KRvK) -> 100%
6. K+P endings -> 100%
7. Dvoretsky, one chapter at a time

## Tools
- checklist.py — tournament-player gate (SF resistance, 100% bars)
- ladder_match.sh — Elo ladder vs calibrated anchors + SF UCI_Elo rungs
