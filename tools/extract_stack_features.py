"""STACK EXTRACTION: every audited-bin position through ALL 14 frozen experts,
capturing each expert's feature-transformer activations. Cached uint8 on the
RAID (per-expert affine quantization); the stacked head trains from the cache.

Usage: extract_stack_features.py <bindir> <outdir> [max_per_bin]
"""
import os
import sys
import glob
import struct

import numpy as np
import torch

sys.path.insert(0, "/home/spec/nnue-pytorch")
sys.path.insert(0, "/home/spec/chess-lab/tools")
import data_loader
import model as M
from audit_packer import unpack_sfen
import chess

RUNS = "/mnt/cold-raid6/chess-audit/runs"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
BATCH = 2048

EXPERTS = ["balanced_l0", "balanced_l1", "nvb", "exchanges", "balanced_l2",
           "oppb", "bvr", "dvoretsky", "balanced_l3", "nvr", "rv2m",
           "qvmat", "tactics"]   # tb_training: no checkpoint yet (retrain queued)


def load_expert(name):
    ck = sorted(glob.glob(
        f"{RUNS}/{name}/lightning_logs/version_*/checkpoints/last.ckpt"))[-1]
    ckpt = torch.load(ck, map_location="cpu", weights_only=False)
    net = M.NNUE(config=M.NNUELightningConfig())
    net.load_state_dict(ckpt["state_dict"])
    net.to(DEVICE).eval()
    return net.model  # plain NNUEModel


class Activ:
    """Capture model.input output via forward hook."""

    def __init__(self, mdl):
        self.buf = None
        mdl.input.register_forward_hook(self._hook)

    def _hook(self, module, inp, out):
        o = out[0] if isinstance(out, tuple) else out
        self.buf = o.detach()


def transformer_acts(mdl, act, fens):
    b = data_loader.get_sparse_batch_from_fens(
        "Full_Threats+PP_3Wide+HalfKAv2_hm", fens, [0] * len(fens),
        [1] * len(fens), [0] * len(fens))
    (us, them, wi, bi, _o, _s, pc) = b.contents.get_tensors(DEVICE)
    with torch.no_grad():
        mdl.forward(us, them, wi, bi, pc)
    data_loader.destroy_sparse_batch(b)
    return act.buf  # shape (N, 2, 1032) or similar


def main():
    bindir, outdir = sys.argv[1], sys.argv[2]
    max_per_bin = int(sys.argv[3]) if len(sys.argv) > 3 else 0
    os.makedirs(outdir, exist_ok=True)

    print("loading 14 experts...", flush=True)
    nets, acts = [], []
    for e in EXPERTS:
        m = load_expert(e)
        nets.append(m)
        acts.append(Activ(m))
        print("  loaded", e, flush=True)

    # calibration pass on a sample to get per-expert min/max
    calib_fens = []
    with open(os.path.join(bindir, "balanced_l0.bin"), "rb") as f:
        for i in range(20000):
            r = f.read(40)
            if len(r) < 40:
                break
            b, hm, fm = unpack_sfen(r[:32])
            calib_fens.append(b.fen())
    print("calibrating...", flush=True)
    scales, zeros = [], []
    D = None
    for m, a in zip(nets, acts):
        smin = smin2 = 1e30
        smax = smax2 = -1e30
        for i in range(0, len(calib_fens), 4096):
            t = transformer_acts(m, a, calib_fens[i:i + 4096])
            if D is None:
                D = t.shape[-1]   # per-position activation dim
            flat = t.reshape(t.shape[0], -1)
            smin = min(smin, flat.min().item())
            smax = max(smax, flat.max().item())
        scale = (smax - smin) / 255.0
        scales.append(scale)
        zeros.append(smin)
        print("  calib %s: [%.3f, %.3f]" % (EXPERTS[len(scales) - 1], smin, smax),
              flush=True)
    np.savez(os.path.join(outdir, "calibration.npz"),
             experts=np.array(EXPERTS), scale=np.array(scales),
             zero=np.array(zeros), D=D)

    # extraction per bin
    SKIP = {"tactics", "tb_training"}   # huge; not stack training positions
    for path in sorted(glob.glob(bindir + "/*.bin")):
        name = os.path.basename(path)[:-4]
        if name in SKIP:
            continue
        n = os.path.getsize(path) // 40
        if max_per_bin:
            n = min(n, max_per_bin)
        out = np.lib.format.open_memmap(
            os.path.join(outdir, name + ".uint8.npy"), mode="w+",
            dtype=np.uint8, shape=(n, 14, D))
        targets = np.zeros(n, dtype=np.int16)
        fens = []
        idx = 0
        with open(path, "rb") as f:
            rec_i = 0
            while rec_i < n:
                r = f.read(40)
                if len(r) < 40:
                    break
                cp = struct.unpack("<h", r[32:34])[0]
                b, hm, fm = unpack_sfen(r[:32])
                fens.append((b.fen(), cp))
                rec_i += 1
                if len(fens) >= BATCH or rec_i == n:
                    fs = [x[0] for x in fens]
                    for ei, (m, a) in enumerate(zip(nets, acts)):
                        t = transformer_acts(m, a, fs)
                        flat = t.reshape(t.shape[0], -1).cpu().numpy()
                        q = np.clip(
                            (flat - zeros[ei]) / max(scales[ei], 1e-9), 0, 255)
                        out[idx:idx + len(fs), ei, :] = q.astype(np.uint8)
                    for j, x in enumerate(fens):
                        targets[idx + j] = max(-32000, min(32000, x[1]))
                    idx += len(fens)
                    fens = []
                    if idx % (BATCH * 25) == 0:
                        print("  %s: %d/%d" % (name, idx, n), flush=True)
        out.flush()
        np.save(os.path.join(outdir, name + ".targets.npy"), targets)
        print("DONE %s: %d positions, D=%d" % (name, idx, D), flush=True)
    print("EXTRACTION-COMPLETE")


if __name__ == "__main__":
    main()
