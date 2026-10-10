s = open("/srv/workspace/flychess/src/Stockfish-act/src/eval_head.h").read()
broken = """            u32 eemb;
            if (fread(&eemb, 4, 1, f) != 1 || eemb != EVAL_FT_DIMS * 1024)
                { fprintf(stderr, "EVLDBG eemb=%u want=%d ftell=%ld\\n", eemb, EVAL_FT_DIMS*1024, ftell(f)); }
                { fprintf(stderr, "EVL fail emb dims\\n"); fclose(f); return false; }"""
fixed = """            u32 eemb;
            if (fread(&eemb, 4, 1, f) != 1 || eemb != u32(EVAL_FT_DIMS) * 1024u)
                { fprintf(stderr, "EVL fail emb dims (%u)\\n", eemb); fclose(f); return false; }"""
assert broken in s, "anchor missing"
s = s.replace(broken, fixed)
open("/srv/workspace/flychess/src/Stockfish-act/src/eval_head.h", "w").write(s)
print("fixed")
