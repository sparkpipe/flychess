"""FULL-corpus sidecar extraction — every position of every aug bin.

For each aug_* bin not yet fully covered, extract per position:
  13 scalar expert evals (GPU batch)
  13x1024 activations (GPU, hook per expert, engine scale later)
  raw active feature indices (engine rawft)
  target cp + packed move (from bin records)
Writes per-bin sidecars compatible with the dense trainers, then a manifest.
Idempotent: skips bins already complete.
"""
import sys, os, glob, time, subprocess
sys.path.insert(0, "/srv/workspace/flychess/src/nnue-pytorch")
sys.path.insert(0, "/srv/workspace/flychess/src/chess-lab/tools")
import numpy as np
import torch
import audit_packer

R = "/mnt/cold-raid6/chess-audit"
E_N, ACT_D, FT_D, MAXF = 13, 1024, 86896, 160
OUT = "/mnt/cold-raid6/chess-audit/sidecars_full"
ACT_BIN = "/srv/workspace/flychess/src/Stockfish-act/src/stockfish"

def extract_bin(bin_path, chunk_positions=200000):
    import extract_stack_features as E
    import data_loader
    name = os.path.basename(bin_path)[:-4]
    n_total = os.path.getsize(bin_path) // 40
    done_marker = f"{OUT}/{name}.done"
    if os.path.exists(done_marker):
        print(f"  {name}: done already", flush=True)
        return
    raw = open(bin_path, "rb").read()

    nets = [E.load_expert(e) for e in
            ["balanced_l0","balanced_l1","balanced_l2","balanced_l3",
             "nvb","nvr","bvr","rv2m","qvmat","oppb",
             "dvoretsky","exchanges","tactics"]]
    hooks = []
    caps = [[] for _ in nets]
    for mdl, cap in zip(nets, caps):
        def mk(buf):
            def h(mod, inp, out):
                o = out[0] if isinstance(out, tuple) else out
                buf.append(o.detach().cpu().numpy().copy())
            return h
        hooks.append(mdl.input.register_forward_hook(mk(cap)))

    eng = subprocess.Popen([ACT_BIN], stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                           text=True, bufsize=1)
    eng.stdin.write("uci\n"); eng.stdin.flush(); time.sleep(1.0)

    t0 = time.time()
    st = 0
    part = 0
    while st < n_total:
        en = min(st + chunk_positions, n_total)
        fens = []
        cps = np.zeros(en - st, dtype=np.int16)
        for i in range(st, en):
            b = chess = audit_packer.unpack_sfen(raw[i*40:i*40+32])[0]
            fens.append(b.fen())
            cps[i - st] = int.from_bytes(raw[i*40+32:i*40+34], "little", signed=True)
        # evals + acts in one pass
        evals = np.zeros((en - st, E_N), dtype=np.float32)
        acts = np.zeros((en - st, E_N, ACT_D), dtype=np.uint8)
        B = 512
        for s2 in range(0, len(fens), B):
            chunk = fens[s2:s2+B]
            bs = data_loader.get_sparse_batch_from_fens(
                "Full_Threats+PP_3Wide+HalfKAv2_hm", chunk,
                [0]*len(chunk), [1]*len(chunk), [0]*len(chunk))
            us, them, wi, bi, o, sc, pc = bs.contents.get_tensors("cuda")
            for cap in caps:
                cap.clear()
            for mdl in nets:
                with torch.no_grad():
                    mdl.forward(us, them, wi, bi, pc)
            for ei in range(E_N):
                arr = caps[ei][-1]
                acts[s2:s2+len(chunk), ei, :] = (arr[:, 0, :ACT_D] if arr.ndim == 3 else arr[:, :ACT_D])
                del caps[ei][:-1]
            # evals from the same forward: use model output
            for ei, mdl in enumerate(nets):
                with torch.no_grad():
                    vs = mdl.forward(us, them, wi, bi, pc)
                evals[s2:s2+len(chunk), ei] = [
                    float(v) * mdl.quantization.nnue2score for v in vs]
            data_loader.destroy_sparse_batch(bs)
        # rawft indices
        ft = np.full((en - st, MAXF), -1, dtype=np.int32)
        cnt = np.zeros(en - st, dtype=np.int32)
        for k, fen in enumerate(fens):
            eng.stdin.write(f"rawft {fen}\n"); eng.stdin.flush()
            line = eng.stdout.readline()
            while line and not line.startswith("rawft"):
                line = eng.stdout.readline()
            toks = line.split()[1:]
            c = min(len(toks), MAXF)
            ft[k, :c] = [int(t) for t in toks[:c]]
            cnt[k] = c
        np.savez(f"{OUT}/{name}_p{part:03d}.npz", evals=evals, acts=acts,
                 ft=ft, ftcnt=cnt, cps=cps)
        st = en
        part += 1
        print(f"  {name}: {st}/{n_total} ({time.time()-t0:.0f}s)", flush=True)
    eng.stdin.write("quit\n"); eng.stdin.flush()
    for h in hooks:
        h.remove()
    open(done_marker, "w").write(str(n_total))
    print(f"  {name}: COMPLETE ({n_total})", flush=True)

def main():
    import chess  # noqa
    os.makedirs(OUT, exist_ok=True)
    for b in sorted(glob.glob(f"{R}/expert_bins_both/*.bin")):
        extract_bin(b)
    print("ALL_SIDECARS_DONE", flush=True)

if __name__ == "__main__":
    main()
