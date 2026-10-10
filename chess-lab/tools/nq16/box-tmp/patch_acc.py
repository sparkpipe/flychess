import sys
P = "/srv/workspace/flychess/src/Stockfish-act/src/nnue/nnue_accumulator.cpp"
s = open(P).read()
orig = s

# 1) includes + helpers after simd.h include
a = '#include "simd.h"\n\nnamespace Stockfish::Eval::NNUE {'
b = '''#include "simd.h"

#include <atomic>
#include <cstdio>
#include <cstdlib>

namespace Stockfish::Eval::NNUE {

// phase-moe perf: when the incremental replay distance for a perspective
// exceeds PHASE_MOE_ACC plies, refresh only the top state from the Finny
// cache (constant cost, no backfill). Values are identical either way.
namespace {
int moe_acc_cap() {
    static const int v = [] {
        const char* e = getenv("PHASE_MOE_ACC");
        return e ? atoi(e) : 0;
    }();
    return v;
}
std::atomic<long> accStatEvals{0}, accStatFwdPlies{0}, accStatCapped{0};
void acc_stats_report() {
    if (getenv("PHASE_MOE_ACC"))
        fprintf(stderr, "ACCSTAT evals=%ld fwd_plies=%ld capped=%ld\\n",
                accStatEvals.load(), accStatFwdPlies.load(), accStatCapped.load());
}
struct AccStatInit {
    AccStatInit() { atexit(acc_stats_report); }
};
const AccStatInit accStatInit;
}  // namespace'''
assert s.count(a) == 1, "anchor1"
s = s.replace(a, b)

# 2) evaluate(): distance-capped branch
a = '''    const usize last_white = find_last_usable_accumulator(WHITE);
    const usize last_black = find_last_usable_accumulator(BLACK);

    if (accumulators[last_white].computed[WHITE] && accumulators[last_black].computed[BLACK])
        forward_update_incremental_both(pos, featureTransformer, last_white, last_black);
    else
    {
        evaluate_side(WHITE, pos, featureTransformer, cache, last_white);
        evaluate_side(BLACK, pos, featureTransformer, cache, last_black);
    }
}'''
b = '''    const usize last_white = find_last_usable_accumulator(WHITE);
    const usize last_black = find_last_usable_accumulator(BLACK);

    const int   cap      = moe_acc_cap();
    const usize d_white  = size - 1 - last_white;
    const usize d_black  = size - 1 - last_black;
    if (cap > 0)
        accStatEvals.fetch_add(1, std::memory_order_relaxed);

    if (accumulators[last_white].computed[WHITE] && accumulators[last_black].computed[BLACK])
    {
        if (cap > 0 && (d_white > usize(cap) || d_black > usize(cap)))
        {
            accStatCapped.fetch_add(1, std::memory_order_relaxed);
            if (d_white > usize(cap))
                update_accumulator_refresh_cache(WHITE, featureTransformer, pos, mut_latest(),
                                                 cache);
            else
                forward_update_incremental(WHITE, pos, featureTransformer, last_white);
            if (d_black > usize(cap))
                update_accumulator_refresh_cache(BLACK, featureTransformer, pos, mut_latest(),
                                                 cache);
            else
                forward_update_incremental(BLACK, pos, featureTransformer, last_black);
        }
        else
        {
            if (cap > 0)
                accStatFwdPlies.fetch_add(long(d_white) + long(d_black),
                                          std::memory_order_relaxed);
            forward_update_incremental_both(pos, featureTransformer, last_white, last_black);
        }
    }
    else
    {
        evaluate_side(WHITE, pos, featureTransformer, cache, last_white);
        evaluate_side(BLACK, pos, featureTransformer, cache, last_black);
    }
}'''
assert s.count(a) == 1, "anchor2"
s = s.replace(a, b)

# 3) evaluate_side(): cap in the computed branch + skip backfill when capped
a = '''    if (accumulators[last_usable_accum].computed[perspective])
        forward_update_incremental(perspective, pos, featureTransformer, last_usable_accum);

    else
    {'''
b = '''    const int cap = moe_acc_cap();
    if (accumulators[last_usable_accum].computed[perspective])
    {
        if (cap > 0 && size - 1 - last_usable_accum > usize(cap))
        {
            accStatCapped.fetch_add(1, std::memory_order_relaxed);
            update_accumulator_refresh_cache(perspective, featureTransformer, pos, mut_latest(),
                                             cache);
            return;
        }
        if (cap > 0)
            accStatFwdPlies.fetch_add(long(size - 1 - last_usable_accum),
                                      std::memory_order_relaxed);
        forward_update_incremental(perspective, pos, featureTransformer, last_usable_accum);
    }

    else
    {'''
assert s.count(a) == 1, "anchor3"
s = s.replace(a, b)

a = '''        update_accumulator_refresh_cache(perspective, featureTransformer, pos, mut_latest(), cache);
        backward_update_incremental(perspective, pos, featureTransformer, last_usable_accum);'''
b = '''        update_accumulator_refresh_cache(perspective, featureTransformer, pos, mut_latest(), cache);
        if (cap > 0 && size - 1 - last_usable_accum > usize(cap))
            return;  // capped: skip backfill, chain stays lazy
        backward_update_incremental(perspective, pos, featureTransformer, last_usable_accum);'''
assert s.count(a) == 1, "anchor4"
s = s.replace(a, b)

assert s != orig
open(P, "w").write(s)
print("PATCH OK, 4 anchors replaced")
