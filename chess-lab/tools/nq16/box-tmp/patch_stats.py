P = "/srv/workspace/flychess/src/Stockfish-act/src/nnue/nnue_accumulator.cpp"
s = open(P).read()

a = 'std::atomic<long> accStatEvals{0}, accStatFwdPlies{0}, accStatCapped{0};'
b = 'std::atomic<long> accStatEvals{0}, accStatFwdPlies{0}, accStatCapped{0};\nstd::atomic<long> accStatRefresh{0}, accStatHybrid{0}, accStatBackPlies{0}, accStatIncr{0};'
assert s.count(a) == 1
s = s.replace(a, b)

a = '''        fprintf(stderr, "ACCSTAT evals=%ld fwd_plies=%ld capped=%ld\\n",
                accStatEvals.load(), accStatFwdPlies.load(), accStatCapped.load());'''
b = '''        fprintf(stderr,
                "ACCSTAT evals=%ld fwd_plies=%ld capped=%ld refresh=%ld hybrid=%ld "
                "back_plies=%ld incr=%ld\\n",
                accStatEvals.load(), accStatFwdPlies.load(), accStatCapped.load(),
                accStatRefresh.load(), accStatHybrid.load(), accStatBackPlies.load(),
                accStatIncr.load());'''
assert s.count(a) == 1
s = s.replace(a, b)

# count in evaluate_side: hybrid call, refresh call, backward plies
a = '''            update_accumulator_hybrid(perspective, pos, featureTransformer, mut_latest(),
                                      accumulators[size - 2], cache);
            return;'''
b = '''            accStatHybrid.fetch_add(1, std::memory_order_relaxed);
            update_accumulator_hybrid(perspective, pos, featureTransformer, mut_latest(),
                                      accumulators[size - 2], cache);
            return;'''
assert s.count(a) == 1
s = s.replace(a, b)

a = '''        update_accumulator_refresh_cache(perspective, featureTransformer, pos, mut_latest(), cache);
        if (cap > 0 && size - 1 - last_usable_accum > usize(cap))
            return;  // capped: skip backfill, chain stays lazy
        backward_update_incremental(perspective, pos, featureTransformer, last_usable_accum);'''
b = '''        accStatRefresh.fetch_add(1, std::memory_order_relaxed);
        update_accumulator_refresh_cache(perspective, featureTransformer, pos, mut_latest(), cache);
        if (cap > 0 && size - 1 - last_usable_accum > usize(cap))
            return;  // capped: skip backfill, chain stays lazy
        accStatBackPlies.fetch_add(long(size - 1 - last_usable_accum),
                                   std::memory_order_relaxed);
        backward_update_incremental(perspective, pos, featureTransformer, last_usable_accum);'''
assert s.count(a) == 1
s = s.replace(a, b)

# count every incremental update call (replay unit)
a = '''    for (usize next = begin + 1; next < size; next++)
        update_accumulator_incremental<true>(perspective, featureTransformer, ksq,
                                             accumulators[next], accumulators[next - 1]);

    assert(latest().computed[perspective]);'''
b = '''    accStatIncr.fetch_add(long(size - 1 - begin), std::memory_order_relaxed);
    for (usize next = begin + 1; next < size; next++)
        update_accumulator_incremental<true>(perspective, featureTransformer, ksq,
                                             accumulators[next], accumulators[next - 1]);

    assert(latest().computed[perspective]);'''
assert s.count(a) == 1
s = s.replace(a, b)

open(P, "w").write(s)
print("STATS PATCH OK")
