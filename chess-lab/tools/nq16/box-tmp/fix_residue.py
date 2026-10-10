p = "/srv/workspace/flychess/src/Stockfish-act/src/phase_moe.h"
t = open(p).read()

# Replace the broken extras/hasPair residue section with direct diff-pattern checks
start = t.index("    if (asym)\n    {")
end = t.index("    if (!asym && opp_bishops(pos))")

new_block = """    if (asym)
    {
        // diff patterns (wn,wb,wr,wq are W-minus-B differences) with the
        // mirrored case; equivalent to the python router's residue strings.
        const bool bvr = (wb == 1 && wr == -1) || (wb == -1 && wr == 1);
        const bool nvr = (wn == 1 && wr == -1) || (wn == -1 && wr == 1);
        const bool rv2nn = (wr == 1 && wn == -2);
        const bool rv2bb = (wr == 1 && wb == -2);
        const bool rv2nb = (wr == 1 && wn == -1 && wb == -1);
        const bool qvrr = (wq == 1 && wr == -2);
        const bool qvrn = (wq == 1 && wr == -1 && wn == -1);
        const bool qvrb = (wq == 1 && wr == -1 && wb == -1);
        const bool nnvnb = (wn == 1 && wb == -1) || (wn == -1 && wb == 1);
        const bool nnvbb = (wn == 2 && wb == -2) || (wn == -2 && wb == 2);
        // N v B residue: deferred until after dv domain (nvb LAST ruling)
        const bool bvn = nnvnb;  // same diff pattern; checked after dv

        if (bvr || nvr)
            return EX_MVR;
        if (rv2nn || rv2bb || rv2nb)
            return EX_RV2M;
        if (qvrr || qvrn || qvrb)
            return EX_QVMAT;
        if (nnvbb)
            return EX_N2V2;
        if (!bvn)
            return moverUp ? EX_PD_UP : EX_PD_DOWN;
    }
"""
t = t[:start] + new_block + t[end:]

# Remove the old deferred nvb check (was checking wrong patterns)
old_nvb = """    if (asym && ((extras(0, 1, 0, 0, WHITE) && extras(1, 0, 0, 0, BLACK))
              || (extras(1, 0, 0, 0, WHITE) && extras(0, 1, 0, 0, BLACK))))
        return EX_NVBLATE;"""
new_nvb = """    if (asym && ((wn == 1 && wb == -1) || (wn == -1 && wb == 1)))
        return EX_NVBLATE;"""
assert old_nvb in t
t = t.replace(old_nvb, new_nvb)

# Also remove the now-unused extras lambda and match lambda to avoid warnings
old_lambdas = t[t.index("    auto match = [&](int a_n"):t.index("    if (asym)\n    {\n        // diff patterns")]
t = t.replace(old_lambdas, "")

open(p, "w").write(t)
print("residue logic replaced with diff patterns")
