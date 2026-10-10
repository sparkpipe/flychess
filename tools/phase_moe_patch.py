#!/usr/bin/env python3
"""phase-moe-v0 fork patcher: adds a second (endgame) NNUE network to
Stockfish, routed by piece count <= 10. Applies textual edits to the
five files that carry the single-network assumption. Idempotent-ish:
refuses to patch if markers already present."""
import sys

SRC = "/home/spec/Stockfish/src"


def patch(path, edits, marker):
    with open(path) as f:
        s = f.read()
    if marker in s:
        print(f"SKIP {path} (already patched)")
        return
    for old, new in edits:
        assert old in s, f"{path}: anchor missing:\n{old[:120]}"
        s = s.replace(old, new, 1)
    with open(path, "w") as f:
        f.write(s)
    print(f"PATCHED {path}")


# --- search.h: SharedState + Worker members -------------------------
patch(
    f"{SRC}/search.h",
    [(
        """                const LazyNumaReplicatedSystemWide<Eval::NNUE::Network>& net) :
        options(optionsMap),
        threads(threadPool),
        tt(transpositionTable),
        sharedHistories(sharedHists),
        network(net) {}""",
        """                const LazyNumaReplicatedSystemWide<Eval::NNUE::Network>& net,
                const LazyNumaReplicatedSystemWide<Eval::NNUE::Network>& netEnd) :
        options(optionsMap),
        threads(threadPool),
        tt(transpositionTable),
        sharedHistories(sharedHists),
        network(net), networkEnd(netEnd) {}"""),
    (
        """    const LazyNumaReplicatedSystemWide<Eval::NNUE::Network>& network;
};

class Worker;""",
        """    const LazyNumaReplicatedSystemWide<Eval::NNUE::Network>& network;
    const LazyNumaReplicatedSystemWide<Eval::NNUE::Network>& networkEnd;
};

class Worker;"""),
    (
        """    const LazyNumaReplicatedSystemWide<Eval::NNUE::Network>& network;

    // Used by NNUE
    Eval::NNUE::AccumulatorStack  accumulatorStack;
    Eval::NNUE::AccumulatorCaches refreshTable;""",
        """    const LazyNumaReplicatedSystemWide<Eval::NNUE::Network>& network;
    const LazyNumaReplicatedSystemWide<Eval::NNUE::Network>& networkEnd;

    // Used by NNUE
    Eval::NNUE::AccumulatorStack  accumulatorStack;
    Eval::NNUE::AccumulatorCaches refreshTable;
    Eval::NNUE::AccumulatorCaches refreshTableEnd;"""),
    (
        """                const LazyNumaReplicatedSystemWide<Eval::NNUE::Network>& net) :""",
        """                const LazyNumaReplicatedSystemWide<Eval::NNUE::Network>& net,
                const LazyNumaReplicatedSystemWide<Eval::NNUE::Network>& netEnd) :"""),
    (
        """    const LazyNumaReplicatedSystemWide<Eval::NNUE::Network>& network;
""",
        """    const LazyNumaReplicatedSystemWide<Eval::NNUE::Network>& network;
    const LazyNumaReplicatedSystemWide<Eval::NNUE::Network>& networkEnd;
"""),
],
    marker="networkEnd;")

# --- search.cpp: ctor init + eval routing + clears ------------------
patch(
    f"{SRC}/search.cpp",
    [(
        """    network(sharedState.network),
    refreshTable(network[token]) {""",
        """    network(sharedState.network),
    networkEnd(sharedState.networkEnd),
    refreshTable(network[token]), refreshTableEnd(networkEnd[token]) {"""),
    (
        """    // Access once to force lazy initialization, avoiding initialization during search
    (void) (network[numaAccessToken]);""",
        """    // Access once to force lazy initialization, avoiding initialization during search
    (void) (network[numaAccessToken]);
    (void) (networkEnd[numaAccessToken]);"""),
    (
        """    refreshTable.clear(network[numaAccessToken]);""",
        """    refreshTable.clear(network[numaAccessToken]);
    refreshTableEnd.clear(networkEnd[numaAccessToken]);"""),
    (
        """    return Eval::evaluate(network[numaAccessToken], pos, accumulatorStack, refreshTable,""",
        """    const bool endgamePhase = pos.count<ALL_PIECES>() <= 10;  // phase-moe
    return Eval::evaluate(endgamePhase ? networkEnd[numaAccessToken]
                                       : network[numaAccessToken],
                          pos, accumulatorStack,
                          endgamePhase ? refreshTableEnd : refreshTable,"""),
],
    marker="phase-moe")

# --- engine.h: second network member --------------------------------
patch(f"{SRC}/engine.h",
      [("    LazyNumaReplicatedSystemWide<Eval::NNUE::Network> network;",
        "    LazyNumaReplicatedSystemWide<Eval::NNUE::Network> network;\n"
        "    LazyNumaReplicatedSystemWide<Eval::NNUE::Network> networkEnd;"
        "  // phase-moe")],
      marker="networkEnd;")

# --- engine.cpp: init, option, load, replication --------------------
patch(f"{SRC}/engine.cpp",
      [("    network(numaContext, get_default_network()) {",
        "    network(numaContext, get_default_network()),\n"
        "    networkEnd(numaContext, get_default_network()) {  // phase-moe"),
       ("      \"EvalFile\", Option(EvalFileDefaultName, [this](const Option& o) {",
        "      \"EvalFile2\", Option(EvalFileDefaultName, [this](const Option& o) {\n"
        "          load_network(path_from_utf8(std::string(o)), true);\n"
        "          threads.ensure_network_replicated();\n"
        "      }),  // phase-moe\n"
        "      \"EvalFile\", Option(EvalFileDefaultName, [this](const Option& o) {"),
       ("    threads.set(numaContext.get_numa_config(), {options, threads, tt, sharedHists, network},",
        "    threads.set(numaContext.get_numa_config(), {options, threads, tt, sharedHists, network, networkEnd},"),
       ],
      marker="EvalFile2")

print("all patches applied")
