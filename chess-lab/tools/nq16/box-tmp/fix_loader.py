s = open("/srv/workspace/flychess/src/Stockfish-act/src/eval_head.h").read()
broken = '''        if (fread(&n_experts, 4, 1, f) != 1 ) { fprintf(stderr, "EVL fail experts\\\\n"); fclose(f); return false; } || n_experts > EVAL_MAX_EXPERTS)
            { fclose(f); return false; }
        if (fread(&hce_dims, 4, 1, f) != 1 ) { fprintf(stderr, "EVL fail hce\\\\n"); fclose(f); return false; } || hce_dims > EVAL_HCE_DIMS)
            { fclose(f); return false; }
        if (fread(&n_domains, 4, 1, f) != 1 ) { fprintf(stderr, "EVL fail doms\\\\n"); fclose(f); return false; } || n_domains > EVAL_MAX_DOMAINS)
            { fclose(f); return false; }'''
fixed = '''        if (fread(&n_experts, 4, 1, f) != 1 || n_experts > EVAL_MAX_EXPERTS)
            { fprintf(stderr, "EVL fail experts %u\\\\n", n_experts); fclose(f); return false; }
        if (fread(&hce_dims, 4, 1, f) != 1 || hce_dims > EVAL_HCE_DIMS)
            { fprintf(stderr, "EVL fail hce %u\\\\n", hce_dims); fclose(f); return false; }
        if (fread(&n_domains, 4, 1, f) != 1 || n_domains > EVAL_MAX_DOMAINS)
            { fprintf(stderr, "EVL fail doms %u\\\\n", n_domains); fclose(f); return false; }'''
assert broken in s, "broken block not found"
s = s.replace(broken, fixed)
open("/srv/workspace/flychess/src/Stockfish-act/src/eval_head.h", "w").write(s)
print("loader repaired")
