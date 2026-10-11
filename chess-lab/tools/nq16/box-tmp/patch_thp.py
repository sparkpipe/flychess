P = "/srv/workspace/flychess/src/Stockfish-act/src/nnue/network.cpp"
s = open(P).read()

a = '''template<>
std::optional<std::string> Network::load(std::istream& stream) {
    initialize();
    std::string description;

    return read_parameters(stream, description) ? std::make_optional(description) : std::nullopt;
}'''
b = '''template<>
std::optional<std::string> Network::load(std::istream& stream) {
    initialize();
    std::string description;

    // phase-moe perf: hint hugepages over the whole Network object (the feature
    // transformer dominates it); routed evals jump between ~23 such objects.
    if (read_parameters(stream, description))
    {
        const uintptr_t s = reinterpret_cast<uintptr_t>(this) & ~uintptr_t(4095);
        const uintptr_t e =
          (reinterpret_cast<uintptr_t>(this) + sizeof(Network) + 4095) & ~uintptr_t(4095);
        madvise(reinterpret_cast<void*>(s), e - s, MADV_HUGEPAGE);
        return std::make_optional(description);
    }
    return std::nullopt;
}'''
assert s.count(a) == 1
s = s.replace(a, b)

# include
a = '#include "../misc.h"'
assert s.count(a) == 1, "inc"
s = s.replace(a, '#include "../misc.h"\n\n#include <sys/mman.h>')

open(P, "w").write(s)
print("THP PATCH OK")
