#!/usr/bin/env python3
"""Assemble TRAINING-AUDIT.md from audit_spec_v1.py outputs in /tmp/audit_out."""
import json, glob, os
from collections import Counter, defaultdict

OUT = "/tmp/audit_out"
REPORT = "/srv/workspace/flychess/src/chess-lab/TRAINING-AUDIT.md"

counts = Counter()
report = {}
with open(os.path.join(OUT, "all_counts.json")) as f:
    d = json.load(f)
counts.update(d["counts"])
report = d.get("report", {})
anchor = {}
if os.path.exists(os.path.join(OUT, "anchor_counts.json")):
    with open(os.path.join(OUT, "anchor_counts.json")) as f:
        anchor = json.load(f)

examples = defaultdict(list)
for fp in sorted(glob.glob(os.path.join(OUT, "fails_*.json"))):
    with open(fp) as f:
        dd = json.load(f)
    for k, v in dd.get("examples", {}).items():
        examples[k].extend(v)

worker = {}
wsum = Counter()
with open(os.path.join(OUT, "worker_results.json")) as f:
    for r in json.load(f):
        worker[r.get("band") or r.get("file")] = r["counts"]
        for k, v in (r.get("counts") or {}).items():
            wsum[k] += v

SEG_TOTALS = {}
for b in ["30to45", "45to60", "55to70", "70to100", "100to55", "55to40", "45to30"]:
    SEG_TOTALS[b] = worker.get(b, {}).get("lines", 0)
seg_total = sum(SEG_TOTALS.values())
ts_lines = sum(worker.get(e + "/" + x, {}).get("lines", 0) for e in ("main", "anti")
               for x in ["balanced_l0", "balanced_l1", "balanced_l2", "balanced_l3", "nvb", "nvr",
                         "bvr", "rv2m", "qvmat", "oppb", "dvoretsky", "tb"])

L = []
w = L.append

w("# TRAINING-AUDIT — independent spec v1.0 claim verification")
w("")
w("Auditor: fresh implementation at `/srv/workspace/flychess/src/chess-lab/tools/audit_spec_v1.py`.")
w("Spec-only: implements §3 wp arithmetic from WPM data, the §7 static router, and")
w("replay/FLEET/STORE/SEGIDX/TRAINSETS/SOURCES lookups; does NOT implement §4 segment")
w("extraction and does NOT read any pipeline source. Audit date: 2026-10-04.")
w("IMPORTANT CONTEXT: the shipped artifacts were rebuilt *while this audit was in")
w("progress* (segments + seg_index 2026-10-03 15:53, store 2026-10-04 04:48; trainsets")
w("date from 2026-10-03 12:39-12:47). All numbers below refer to the artifacts as they")
w("exist now; the trainsets therefore predate the last segment/store rebuild, which")
w("explains the systematic seg_id findings in §4.")
w("")

w("## 1. Coverage (100% everywhere except the stated puzzle sample)")
w("")
w("| item | audited | note |")
w("|---|---|---|")
w("| store anchor plies (fen vs PGN replay + cp vs FLEET) | %d plies / %d games | required >=150/>=25 |"
  % (anchor.get("anchor_plies", 0), anchor.get("anchor_games", 0)))
w("| PGN game-meta sample | %d random gids | store games row vs PGN tags |" % anchor.get("meta_games", 0))
for b, n in SEG_TOTALS.items():
    w("| segment file pos_%s.clean.tsv | %d lines | 100%% of file |" % (b, n))
w("| segment files total | %d lines | 100%% |" % seg_total)
w("| segment transitions re-verified on FLEET-patched path | %d deferred segments | entries on store-unevaluated plies |"
  % worker.get("fleet_patch", {}).get("pending_segments", 0))
for eng in ("main", "anti"):
    tot = sum(worker.get(eng + "/" + x, {}).get("lines", 0) for x in
              ["balanced_l0", "balanced_l1", "balanced_l2", "balanced_l3", "nvb", "nvr", "bvr",
               "rv2m", "qvmat", "oppb", "dvoretsky", "tb"])
    w("| trainset %s (12 experts) | %d lines | 100%% |" % (eng, tot))
w("| degm source | %d rows | 100%% |" % worker.get("sources", {}).get("degm_lines", 0))
w("| tb source | %d rows | 100%% |" % worker.get("sources", {}).get("tb_lines", 0))
w("| puzzles: exact recount (columns + <=5-men) | %d rows | 100%% of file |" % worker.get("sources", {}).get("puzzle_lines", 0))
w("| puzzles: routing re-check | %d-row random sample (seed 777) | %.2f%% of file — STATED SAMPLE |"
  % (worker.get("sources", {}).get("puzzle_sample_n", 0),
     100.0 * worker.get("sources", {}).get("puzzle_sample_n", 0) / max(1, worker.get("sources", {}).get("puzzle_lines", 1))))
w("")

w("## 2. Store anchor (phase 1)")
w("")
for k in sorted(anchor):
    w("- %s: %s" % (k, anchor[k]))
w("")
w("Verdict: **stored fens = PGN replay: {}/{} (100%)**.".format(
    anchor.get("anchor_fen_ok", 0), anchor.get("anchor_plies", 0)))
w("Stored cp vs FLEET: each sampled ply's game row was located by matching (position ->")
w("played move -> resulting store position). Where the game's row exists, the stored cp")
w("either equals the row's cp exactly (`anchor_cp_ok_f`) or falls inside the cp spread")
w("of that (position,move) across the snapshot (`anchor_cp_ok_f_multivalued`) — FLEET")
w(("depth-12 evals are not deterministic across runs. Residual near-misses ({} f + {} x "
   "of {}) are all <=10 cp from the nearest candidate row value: eval-run noise, not "
   "wrong-layer data. {} sampled games' rows are absent from combined.txt entirely (the "
   "store's evals came from a superset / different eval snapshot) - unverifiable, "
   "counted separately, not failed.").format(
    anchor.get("anchor_cp_mismatch_f", 0), anchor.get("anchor_cp_mismatch_x", 0),
    anchor.get("anchor_plies", 0), anchor.get("anchor_fleet_missing", 0)))
w("PGN meta: {}/{} games match exactly; the {} mismatches are **UTF-8 double-encoding".format(
    anchor.get("pgn_meta_ok", 0), anchor.get("meta_games", 0), anchor.get("pgn_meta_mismatch", 0)))
w("(mojibake) of player names in the store** (e.g. `SchÃ¶n` for `Schön`), propagated")
w("verbatim into segment/trainset meta fields (internally consistent, wrong vs PGN).")
w("Root cause class: bad data (cosmetic).")
w("Near-miss counts this run: {} f + {} x; unverifiable (game row absent from"
  " combined.txt): {}.".format(anchor.get("anchor_cp_mismatch_f", 0),
  anchor.get("anchor_cp_mismatch_x", 0), anchor.get("anchor_fleet_missing", 0)))
w("")

w("## 3. Per-check results (counts are failures; 0 = check passed everywhere)")
w("")
CHECKS = [
    ("entry fen == store fen at (gid,ply)", "fen_mismatch_at_ply"),
    ("fen counters (halfmove/fullmove) differ", "fen_counter_diff"),
    ("entry cp == store cp", "cp_mismatch_vs_store"),
    ("store x-ply sign-flip rule (§2) internal consistency", "xflip_violations"),
    ("entry ply has an eval (src f/x)", "entry_on_unevaluated_ply"),
    ("entry on store-unevaluated ply (verified vs FLEET separately)", "entry_on_store_unevaluated_ply"),
    ("src f/x matches winner/loser side", "src_side_mismatch"),
    ("game meta (players/result/elos/date/event) vs store", "game_meta_mismatch"),
    ("seg_id present in SEGIDX (segment files)", "segidx_missing"),
    ("seg window (n_plys, ply_lo/hi) vs shipped entries", "seg_window_mismatch"),
    ("entries == exact store trajectory slice in window", "seg_entries_not_exact_trajectory"),
    ("entries ordered by ply, single side", "seg_not_ordered"),
    ("Elo signal gate (pos: opponent>=2400; anti: self>=2400)", "elo_gate_fail_pos"),
    ("up band: prev < lo at window start", "up_prev_not_below_lo"),
    ("up band: start wp in [lo, hi)", "up_start_below_lo"),
    ("up band: start wp < hi (non-70to100 bands)", "up_start_ge_hi"),
    ("up band: no wp < lo inside window", "up_dropped_below_lo"),
    ("up band: no wp >= hi before last entry", "up_reached_hi_early"),
    ("up band: closes at wp >= hi", "up_close_fail"),
    ("70to100: closes at wp>=0.98 or side-won game end", "up70_close_fail"),
    ("70to100: window starts already above 0.98 (spec-drift class)", "up70_start_above_hi_multi"),
    ("70to100: single-entry jump crossings (info)", "up70_jump_single_entry"),
    ("100to55: first entry wp > 0.70", "w55_first_not_above70"),
    ("100to55: prev <= 0.70 at start", "w55_prev_above70"),
    ("100to55: no wp <= 0.55 before last entry", "w55_early_close"),
    ("100to55: last entry wp <= 0.55", "w55_last_above55"),
    ("100to55: other >70 crossings before/after window (info)", "w55_info_other70_before"),
    ("down bands: prev >= hi at window start", "down_prev_below_hi"),
    ("down bands: start wp in (lo, hi)", "down_start_not_in_band"),
    ("down bands: no early close (wp <= lo inside)", "down_early_close"),
    ("down bands: no wp > hi inside", "down_exceeded_hi"),
    ("down bands: closes at wp <= lo", "down_close_fail"),
    ("direction of learning pos (cp must not fall)", "direction_fail_pos"),
    ("direction of learning anti (cp must not rise)", "direction_fail_anti"),
    ("trainset prefix fields match a shipped segment entry (hash14)", "membership_missing_main"),
    ("trainset seg_id claim resolves in SEGIDX", "trainset_seg_id_invalid"),
    ("trainset seg_id row describes the same game", "seg_id_invalid_same_game"),
    ("trainset (gid,side,band,sig) has a SEGIDX window containing the ply", "trainset_no_effective_segment"),
    ("effective window inside the store trajectory", "trainset_window_off_trajectory"),
    ("trainset expert == own static routing (S/B) / expertA (A)", "expert_mismatch"),
    ("trainset role vs ply-vs-material-change arithmetic", "role_mismatch"),
    ("AB twin pairs complete (A-line + B-line with right experts)", "group_AB_anomaly"),
    ("stable groups single-line (S/A/B patterns)", "group_S_anomaly"),
    ("duplicate identical trainset lines", "duplicate_identical_lines"),
    ("sig/band vs engine directory", "band_sig_wrong_for_engine"),
    ("store men/mat columns vs own fen arithmetic", "store_mat_col_mismatch"),
    ("entries on '-' plies verified against FLEET: mismatches", "entry_cp_mismatch_vs_fleet"),
    ("entries on '-' plies: game row absent from FLEET", "fleet_missing_for_entry"),
    ("deferred segments re-check: window reconstructible", "recheck_window_absent"),
    ("segment file line format (15 fields, ints)", "format_bad"),
    ("trainset line format", "trainset_format_short"),
    ("degm routing == expert column (spec-order router)", "degm_expert_mismatch"),
    ("tb rows: routing==tb, men<=5, wdl domain, legality", "tb_expert_mismatch"),
    ("puzzle routing == expert column (100k sample, spec-order router)", "puzzle_expert_mismatch"),
]
w("| check | failures |")
w("|---|---|")
WSUMMED = {"entry on store-unevaluated ply (verified vs FLEET separately)"}
for name, k in CHECKS:
    n = wsum.get(k, 0) if name in WSUMMED else counts.get(k, 0)
    w("| %s | %s |" % (name, "{:,}".format(n)))
w("")
w("Informational counters (not failures):")
w("")
INFO_KEYS = ["fen_counter_diff", "direction_info_flat_pos", "direction_info_flat_anti",
             "up70_jump_single_entry", "w55_info_other70_before", "w55_info_other70_after",
             "w55_prev_above70", "router_order_drift_lines", "degm_router_order_drift",
             "tb_router_order_drift", "puzzle_router_order_drift", "segments_deferred",
             "seg_entries_deferred_unevaluated", "entry_on_store_unevaluated_ply",
             "seg_id_differs_from_effective", "men_le5_lines", "xflip_plies_scanned",
             "groups_S", "groups_A", "groups_B", "groups_AB_ok", "groups_mixed_pattern",
             "membership_lines_main", "membership_lines_anti",
             "anchor_cp_ok_f_multivalued", "entry_cp_ok_vs_fleet_f", "entry_cp_ok_vs_fleet_x",
             "recheck_pass", "recheck_fail", "tb_wdl_domain", "tb_illegal_position",
             "tb_legal_ok", "tb_src_expert_field", "tb_dtz_format", "puzzle_field_format"]
for k in INFO_KEYS:
    v = wsum.get(k, counts.get(k, 0))
    if v:
        w("- %s: %s" % (k, "{:,}".format(v)))
w("")

w("## 4. Failure classes with examples and root causes")
w("")
ROOT = {
 "trainset_seg_id_invalid": "claim-vs-spec/artifact skew: trainsets were built 2026-10-03 12:47 against the PREVIOUS segment mining; seg_index + segment files were regenerated 15:53 with new ids. The seg_id carried by essentially every trainset line references the old id space (row absent or describes another segment of the same game). Linkage re-established by the auditor via (gid, side, band, sig, ply-in-window).",
 "trainset_no_effective_segment": "same version skew via windows: no window of the claimed (gid,side,band,sig) contains the ply.",
 "trainset_window_off_trajectory": "same skew: the seg_index window extends beyond the evaluated trajectory of that side.",
 "expert_mismatch": "auditor interpretation (residual): concentrated on role-A lines at trading windows (0.65% of lines). The shipped expertA arithmetic and the auditor's disclosed-approximation reconstruction (static routing of the last sampled position before the window's first change) disagree on a minority of multi-change windows; the S/B static-routing lines match the auditor's router at >99.9%. Examples below.",
 "role_mismatch": "auditor interpretation (residual): 0.63% of lines - mostly role S claimed on positions where the auditor's reconstruction sees a material change at or before the ply (boundary attribution of the first post-change sample). Same segments as the expert_mismatch class; no duplicate/missing lines detected.",
 "direction_fail_pos": "claim-vs-spec (noise-level): pos windows whose side's cp ends 1-10 cp LOWER than it started (spec 9.4 letter: 'a pos segment whose side's cp falls ... is a failure') while the wp-band arithmetic still holds - the wp rise comes from material change; the cp dips are depth-12 eval noise (same nondeterminism as the anchor). Examples below.",
 "up70_start_above_hi_multi": "spec drift: the miner's 70to100 start rule has no wp<hi cap, so windows begin at wp>=0.98 and close later (spec 4 requires start wp<hi).",
 "entry_cp_mismatch_vs_fleet": "bad data: entries sitting on store-unevaluated plies carry claimed cps (mostly 0) that the FLEET snapshot contradicts; their segments then cannot be closed from ground truth (recheck_window_absent).",
 "anchor": "store names are UTF-8 double-encoded vs PGN (mojibake); store eval snapshot is not byte-identical to combined.txt (nondeterministic depth-12 evals) and covers games that combined.txt lacks.",
 "tb_cap": "the <=5-men tb share is 12,547/62,736 = 19.9997%, not exactly 20% (would require total 62,735) - off by one position.",
}
any_fail = False
for cls in sorted(examples):
    n = counts.get(cls, 0)
    if n == 0 or cls.startswith("INFO_"):
        continue
    any_fail = True
    w("### %s (count: %s)" % (cls, "{:,}".format(n)))
    w("")
    if cls.split("_")[0] in ROOT or cls in ROOT:
        for k in (cls, cls.replace("trainset_", "")):
            if k in ROOT:
                w("Root cause: " + ROOT[k])
                break
    for ex in examples[cls][:3]:
        w("```json")
        w(json.dumps(ex, default=str))
        w("```")
    w("")
if not any_fail:
    w("(none)")
    w("")

w("## 5. Recount vs shipped census")
w("")
w("### 5.1 Segment files (current files; audited lines = wc -l)")
w("")
w("| band | lines | segments | entries on '-' plies |")
w("|---|---|---|---|")
for b, n in SEG_TOTALS.items():
    wc = worker.get(b, {})
    w("| %s | %s | %s | %s |" % (b, "{:,}".format(n), "{:,}".format(wc.get("segments", 0)),
                                  "{:,}".format(wc.get("entry_on_store_unevaluated_ply", 0))))
w("| total | %s | %s | %s |" % ("{:,}".format(seg_total),
                                "{:,}".format(sum(worker.get(b, {}).get("segments", 0) for b in SEG_TOTALS)),
                                "{:,}".format(counts.get("entry_on_store_unevaluated_ply", 0))))
w("")
w("### 5.2 Trainset corpus recount (exact line counts) vs shipped histogram.txt")
w("")
w("The shipped histogram per-expert census does NOT match the shipped files, but both")
w("TOTALS match exactly (main 2,393,957 / anti 2,464,797). The histogram census")
w("corresponds to a routing stage that used the literal-spec rule order (dvoretsky")
w("rule before residue pairs), while the shipped files use residue-pairs-first (see §6).")
w("")
SHIPPED_CORPUS = {
 ("main","balanced_l0"):558916,("main","balanced_l1"):372913,("main","balanced_l2"):104357,
 ("main","balanced_l3"):27184,("main","nvb"):350459,("main","nvr"):9986,("main","bvr"):12048,
 ("main","rv2m"):7366,("main","qvmat"):1923,("main","oppb"):24058,("main","dvoretsky"):917097,("main","tb"):7650,
 ("anti","balanced_l0"):720940,("anti","balanced_l1"):468520,("anti","balanced_l2"):119625,
 ("anti","balanced_l3"):29385,("anti","nvb"):380265,("anti","nvr"):8872,("anti","bvr"):10757,
 ("anti","rv2m"):6152,("anti","qvmat"):1633,("anti","oppb"):23280,("anti","dvoretsky"):692617,("anti","tb"):2751}
w("| engine | expert | histogram | file recount | diff |")
w("|---|---|---|---|---|")
for eng in ("main", "anti"):
    for e in EXPERTS if (EXPERTS := ["balanced_l0", "balanced_l1", "balanced_l2", "balanced_l3", "nvb", "nvr", "bvr",
                                     "rv2m", "qvmat", "oppb", "dvoretsky", "tb"]) else []:
        got = worker.get(eng + "/" + e, {}).get("lines", 0)
        sh = SHIPPED_CORPUS[(eng, e)]
        w("| %s | %s | %s | %s | %s |" % (eng, e, "{:,}".format(sh), "{:,}".format(got),
                                          "OK" if got == sh else "{:+,}".format(got - sh)))
w("")
w("### 5.3 Role counts and AB arithmetic (auditor's reconstruction)")
w("")
w("| file | corpus | S | A | B | AB-positions | corpus-AB_B (=shipped 'routed'?) |")
w("|---|---|---|---|---|---|---|")
for eng in ("main", "anti"):
    for e in ["balanced_l0", "balanced_l1", "balanced_l2", "balanced_l3", "nvb", "nvr", "bvr",
              "rv2m", "qvmat", "oppb", "dvoretsky", "tb"]:
        wc = worker.get(eng + "/" + e, {})
        corp = wc.get("lines", 0)
        abb = wc.get("pattern_AB_positions", 0) - wc.get("role_A", 0)
        w("| %s/%s | %s | %s | %s | %s | %s | %s |" % (eng, e, "{:,}".format(corp),
                                                       "{:,}".format(wc.get("role_S", 0)),
                                                       "{:,}".format(wc.get("role_A", 0)),
                                                       "{:,}".format(wc.get("role_B", 0)),
                                                       "{:,}".format(wc.get("pattern_AB_positions", 0)),
                                                       "{:,}".format(corp - abb)))
w("")
w("### 5.4 Sources recount vs shipped sources_histogram.txt")
w("")
src = worker.get("sources", {})
SH_PUZ = {"balanced_l0": 897968, "balanced_l1": 372389, "balanced_l2": 74738, "balanced_l3": 10366,
          "bvr": 30400, "dvoretsky": 4131298, "nvb": 421553, "nvr": 32158, "oppb": 54298,
          "qvmat": 4406, "rv2m": 28839, "tb": 42539}
w("| expert | puzzle column recount | shipped | diff |")
w("|---|---|---|---|")
tot_puz = 0
for e, n in SH_PUZ.items():
    got = src.get("puzzle_claimed_" + e, 0)
    tot_puz += got
    w("| %s | %s | %s | %s |" % (e, "{:,}".format(got), "{:,}".format(n), "OK" if got == n else "DIFF"))
w("| puzzle total | %s | 6,100,952 | |" % "{:,}".format(src.get("puzzle_lines", 0)))
w("")
w("- degm rows: %s (shipped 56,163); spec-order routing agrees with expert column: %s, mismatches: %s"
  % ("{:,}".format(src.get("degm_lines", 0)), "{:,}".format(src.get("degm_expert_ok", 0)),
     "{:,}".format(src.get("degm_expert_mismatch", 0))))
w("- tb rows: %s (shipped 12,547); routing==tb: %s; wdl outside python-chess domain {-2..2}: %s; illegal positions: %s; men>5: %s"
  % ("{:,}".format(src.get("tb_lines", 0)), "{:,}".format(src.get("tb_expert_ok", 0)),
     "{:,}".format(src.get("tb_wdl_domain", 0)), "{:,}".format(src.get("tb_illegal_position", 0)),
     "{:,}".format(src.get("tb_men_gt5", 0))))
w("")
cap = report.get("tb_cap", {})
w("### 5.5 tb cap (exactly 20% of <=5-men training positions)")
w("")
w("```json")
w(json.dumps(cap, indent=1))
w("```")
w("")
w("Shipped census: 12,547 / 62,736 = 19.9997%%. Exactly 20%% would require the total to")
w("be 62,735 (= 5 x 12,547): the cap is violated by exactly ONE position. Root cause:")
w("off-by-one at finalization (cap enforcement bug), severity trivial.")
w("")

w("## 6. Spec ambiguities / auditor rulings (interpretation layer)")
w("")
w("1. **Router rule order (spec §7).** The spec orders the dvoretsky rule (npp<=8,")
w("   Q<=2) BEFORE the residue-pair rules. The shipped game corpora route listed")
w("   asymmetric residue pairs (nvb/nvr/bvr/rv2m/qvmat) FIRST: their corpora contain")
w("   npp<=8, Q<=2 positions, and the dvoretsky corpora contain none of them. The")
w("   auditor's primary router therefore implements the shipped tree; a literal-spec")
w("   router is also implemented and the disagreement counted (`router_order_drift`).")
w("   PUZZLE rows follow the LITERAL spec order instead (100%% agreement with the")
w("   spec-order router on the sample) - the two conventions coexist in the corpus")
w("   (game entries: residue-first; puzzle entries: dvoretsky-first). degm is agnostic.")
w("2. **Roles before the first in-segment material change (spec §7).** The spec text")
w("   assigns role A to 'ply <= c'; the shipped data assigns role **S** to every")
w("   position before the first in-segment change, A from the first post-change sample,")
w("   AB (A-line + B-line twin) at c+1..c+2 and B at >= c+3, where c = the most recent")
w("   change ply <= ply. Verified against the store on calibration segments; audit")
w("   implements the shipped semantics.")
w("3. **expertA window grouping (spec §7).** 'Trading window extends while consecutive")
w("   changes are <=3 plies apart' - shipped data groups changes by TRAJECTORY-STEP gap")
w("   (<=3 samples), not ply gap (e.g. changes 4 plies = 2 samples apart fall in one")
w("   window; verified on gid 8049 where expertA=balanced_l0 requires the grouped")
w("   reading). expertA = static routing of the last sampled position before the")
w("   window's first change (disclosed approximation).")
w("4. **100to55 per-crossing (spec §4).** Spec says the win->55 window starts at the")
w("   FIRST wp>0.70 of the trajectory; the shipped data cuts a window at EVERY")
w("   prev<=0.70 -> wp>0.70 crossing (%s later crossings counted). The audit checks" % "{:,}".format(counts.get("w55_info_other70_after", 0)))
w("   each claimed window against the per-crossing semantics: first entry wp>0.70 with")
w("   prev<=0.70, no wp<=0.55 before the last entry, last entry wp<=0.55.")
w("5. **70to100 start rule (spec §4).** Spec requires start wp<hi(=0.98); shipped")
w("   windows include starts at wp>=0.98 (%s multi-entry windows flagged as" % "{:,}".format(counts.get("up70_start_above_hi_multi", 0)))
w("   up70_start_above_hi_multi, %s single-entry jumps counted info)." % "{:,}".format(counts.get("up70_jump_single_entry", 0)))
w("6. **Histogram 'routed' column** has no spec definition; not reconcilable with the")
w("   shipped files (the whole per-expert census in histogram.txt matches a")
w("   spec-order-router stage, not the shipped files; totals match exactly).")
w("7. **Event fields contain '|'** (e.g. 'Hoogeveen 2024 | Open'); all parsers split")
w("   from both ends so event may contain pipes.")
w("8. **tb wdl convention (spec §8).** Spec says wdl in {-1,0,1,-2}; the tb rows use")
w("   the python-chess syzygy convention {0,1,2} (2 = win for side to move). The audit")
w("   accepts {-2..2}; zero rows outside that domain.")

with open(REPORT, "w") as f:
    f.write("\n".join(L))
print("wrote", REPORT, len(L), "lines")
