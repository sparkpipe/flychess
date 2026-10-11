s = open("/srv/workspace/flychess/src/Stockfish-act/src/eval_head.h").read()

old_ft = (
    "    // v3: raw board features \u2014 the head's own unfiltered view\n"
    "    if (head.has_ft && raw_ft != nullptr)\n"
    "        for (int i = 0; i < raw_ft_count; i++)\n"
    "            eval += head.ft_w[raw_ft[i]];"
)
new_ft = (
    "    // v3: raw board features \u2014 the head's own unfiltered view\n"
    "    float ft_contrib = 0.0f;\n"
    "    if (head.has_ft && raw_ft != nullptr)\n"
    "        for (int i = 0; i < raw_ft_count; i++)\n"
    "            ft_contrib += head.ft_w[raw_ft[i]];\n"
    "    eval += ft_contrib;"
)
assert old_ft in s, "ft block anchor"
s = s.replace(old_ft, new_ft)

old_act_anchor = "            act_contrib += acc;\n        }\n    }"
old_act = (
    "    // v2: all-expert activation contributions (operator design)\n"
    "    if (head.has_act && expert_acts != nullptr)\n"
    "    {\n"
    "        const u32 D = head.act_dims;\n"
    "        for (u32 e = 0; e < head.act_experts; e++)\n"
    "        {\n"
)
new_act = (
    "    // v2: all-expert activation contributions (operator design)\n"
    "    float act_contrib = 0.0f;\n"
    "    if (head.has_act && expert_acts != nullptr)\n"
    "    {\n"
    "        const u32 D = head.act_dims;\n"
    "        for (u32 e = 0; e < head.act_experts; e++)\n"
    "        {\n"
)
assert old_act in s, "act block anchor"
s = s.replace(old_act, new_act)

# accumulate per-expert acc into act_contrib and add at end
old_tail = (
    "            float acc = 0.0f;\n"
    "            for (u32 d = 0; d < D; d++)\n"
    "                acc += (float(a[d]) - am[d]) * w[d] / as[d];\n"
    "            eval += acc;\n"
    "        }\n"
    "    }"
)
new_tail = (
    "            float acc = 0.0f;\n"
    "            for (u32 d = 0; d < D; d++)\n"
    "                acc += (float(a[d]) - am[d]) * w[d] / as[d];\n"
    "            act_contrib += acc;\n"
    "        }\n"
    "        eval += act_contrib;\n"
    "    }\n"
    "    if (getenv(\"EVL_DEBUG\"))\n"
    "        fprintf(stderr, \"EVLDB ft=%d act=%d base=%d rawcnt=%d\\\\n\",\n"
    "                int(ft_contrib), int(act_contrib),\n"
    "                int(eval - ft_contrib - act_contrib), raw_ft_count);"
)
assert old_tail in s, "act tail anchor"
s = s.replace(old_tail, new_tail)

if "#include <cstdlib>" not in s:
    s = s.replace("#include <cstring>", "#include <cstring>\n#include <cstdlib>", 1)
open("/srv/workspace/flychess/src/Stockfish-act/src/eval_head.h", "w").write(s)
print("decomposition print installed")
