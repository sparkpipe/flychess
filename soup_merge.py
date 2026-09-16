#!/usr/bin/env python3
"""Parameter-soup merge of independently trained piece models + FAIL-hunt
relaunch. Usage: python3 soup_merge.py out.pt in1.pt in2.pt [...]
Uniform-averages every trained parameter (WT frozen/identical; readout_idx
identical via seeded init). After merging, run fly_curriculum stage 1 — its
milestone loop IS the FAIL-hunt (evals all, corrects only failures, sweeps)."""
import sys, torch
out, ins = sys.argv[1], sys.argv[2:]
acc = None
for i, p in enumerate(ins):
    sd = torch.load(p, weights_only=True, map_location="cpu")
    if acc is None:
        acc = {k: v.clone().float() for k, v in sd.items()}
    else:
        for k in acc:
            acc[k] += sd[k].float()
for k in acc:
    acc[k] /= len(ins)
# preserve original dtypes of buffers like readout_idx (long)
ref = torch.load(ins[0], weights_only=True, map_location="cpu")
for k in ref:
    if ref[k].dtype in (torch.int64, torch.bool):
        acc[k] = ref[k].clone()
torch.save(acc, out)
print(f"souped {len(ins)} models -> {out}")
