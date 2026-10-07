s = open("/srv/workspace/flychess/src/Stockfish-act/src/eval_head.h").read()
old = """            u32 fuse_in;
            if (fread(&fuse_in, 4, 1, f) != 1 || fuse_in != EVAL_FUSE_IN)
                { fprintf(stderr, "EVL fail v5 in %u\\n", fuse_in); fclose(f); return false; }
            fuse_V = new float[u64(EVAL_FUSE_IN) * EVAL_FUSE_DIMS]();"""
new = """            u32 fuse_cnt;
            if (fread(&fuse_cnt, 4, 1, f) != 1 || fuse_cnt != u32(EVAL_FUSE_IN) * EVAL_FUSE_DIMS)
                { fprintf(stderr, "EVL fail v5 cnt %u (want %u)\\n", fuse_cnt, u32(EVAL_FUSE_IN)*EVAL_FUSE_DIMS); fclose(f); return false; }
            fuse_V = new float[u64(EVAL_FUSE_IN) * EVAL_FUSE_DIMS]();"""
assert old in s, "v5 loader anchor"
s = s.replace(old, new)
open("/srv/workspace/flychess/src/Stockfish-act/src/eval_head.h", "w").write(s)
print("fixed: count field not dims")
