"""BUILD checkpoint engines — serialize .nnue from every intermediate ckpt.

For each runs2/runs3 expert checkpoint, run the nnue-pytorch serializer to
produce an engine-loadable .nnue. Names: {expert}_{round}_ep{N}.nnue
Output dir: /mnt/cold-raid6/chess-audit/nets_ckpts/
"""
import sys, os, glob, subprocess, re

R = "/mnt/cold-raid6/chess-audit"
OUT = f"{R}/nets_ckpts"
SER = "/srv/workspace/flychess/src/nnue-pytorch/serialize.py"

def main():
    os.makedirs(OUT, exist_ok=True)
    built = 0
    for runs_dir, tag in [("runs2", "r2"), ("runs3", "r3")]:
        for expert_dir in sorted(glob.glob(f"{R}/{runs_dir}/*/lightning_logs/version_*/checkpoints")):
            expert = expert_dir.split("/")[5]
            for ck in sorted(glob.glob(f"{expert_dir}/*.ckpt")):
                base = os.path.basename(ck)
                m = re.search(r"epoch=(\d+)", base)
                if m:
                    name = f"{expert}_{tag}_ep{m.group(1)}"
                else:
                    name = f"{expert}_{tag}_last"
                out = f"{OUT}/{name}.nnue"
                if os.path.exists(out):
                    continue
                cmd = ["python3", SER, ck, out,
                       "--features", "Full_Threats+PP_3Wide+HalfKAv2_hm^"]
                r = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
                if r.returncode == 0 and os.path.exists(out):
                    built += 1
                    print(f"  {name}: OK", flush=True)
                else:
                    print(f"  {name}: FAIL {r.stderr[-100:]}", flush=True)
    print(f"BUILT {built} checkpoint nets", flush=True)
    # list what we have
    nets = sorted(glob.glob(f"{OUT}/*.nnue"))
    print(f"total nets: {len(nets)}", flush=True)
    for n in nets[:20]:
        print(f"  {os.path.basename(n)}", flush=True)

if __name__ == "__main__":
    main()
