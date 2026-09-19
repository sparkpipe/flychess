"""H01 edge extraction v5 — THE CORRECT SEMANTICS (from the cracked
three-layer format): walk the by_id volume (190M synapse annotations)
and emit edges from each annotation's INLINE relationship lists:
  edge = (pre_cell, post_cell, type) for every synapse with both rels.

by_id payload layout (per cloud-volume's _decode_single_annotation):
  [ptA 3xf32][ptB 3xf32][type u32]               (28B annotation)
  [cnt_pre u32][pre_cell u64 x cnt_pre]
  [cnt_post u32][post_cell u64 x cnt_post]
Fast path: minishard rows are (segid, ABSOLUTE offset, length); one
contiguous read per minishard, slice + gunzip per entry.
"""
import os
import sys
import json
import gzip
import struct
import time
import numpy as np
import cloudfiles
from cloudvolume.datasource.precomputed.sharding import (
    ShardReader, ShardingSpecification)

os.environ.setdefault("CLOUD_FILES_LOCK_DIR", "/dev/shm/cloudfiles-locks")
os.makedirs("/dev/shm/cloudfiles-locks", exist_ok=True)

ROOT = ("/mnt/model-warm/human-h01-connectome/data/20210601/"
        "c3/synapses/precomputed")
INFO = "/home/spec/chess-lab/h01_synapses_info.json"
OUT = "/home/spec/chess-lab/h01_edges"
NSHARD = 32                    # by_id: shard_bits 5
PART_EVERY = 5_000_000


class Shim:
    enabled = False

    def get(self, paths, progress=False):
        return {}

    def __init__(self, cf, root=None):
        self.cf = cf
        self.root = root or ROOT

    def download_as(self, requests, progress=False):
        out = {}
        for r in requests:
            with open(os.path.join(self.root, r["path"]), "rb") as f:
                f.seek(r["start"])
                out[(r["path"], r["start"], r["end"])] = \
                    f.read(r["end"] - r["start"])
        return out


def main():
    info = json.load(open(INFO))
    spec = ShardingSpecification.from_dict(info["by_id"]["sharding"])
    rdir = ROOT + "/by_id"
    cf = cloudfiles.CloudFiles("file://" + rdir)
    r = ShardReader("file://" + rdir, Shim(cf, rdir), spec)
    os.makedirs(OUT, exist_ok=True)
    e_pre, e_post, e_typ = [], [], []
    part = 0
    t0 = time.time()
    for shard in range(NSHARD):
        try:
            idx = r.get_index(f"{shard}.shard")
        except Exception:
            continue
        nz = np.nonzero((idx[:, 1] - idx[:, 0]) > 0)[0]
        with open(os.path.join(rdir, f"{shard}.shard"), "rb") as f:
            for ms in nz:
                mi = r.get_minishard_index(f"{shard}.shard", idx, int(ms))
                if mi is None or not len(mi):
                    continue
                lo = int(mi[:, 1].min())
                hi = int((mi[:, 1] + mi[:, 2]).max())
                f.seek(lo)
                blob = f.read(hi - lo)
                for row in mi:
                    v = blob[int(row[1]) - lo:
                             int(row[1]) + int(row[2]) - lo]
                    if v[:1] == b"\x1f":
                        try:
                            v = gzip.decompress(v)
                        except Exception:
                            continue
                    if len(v) < 36:      # 28B + minimal rel lists
                        continue
                    typ = struct.unpack_from("<I", v, 24)[0]
                    off = 28
                    cnt_pre = struct.unpack_from("<I", v, off)[0]
                    off += 4
                    pre_ids = struct.unpack_from(
                        f"<{cnt_pre}Q", v, off) if cnt_pre else ()
                    off += 8 * cnt_pre
                    if off + 4 > len(v):
                        continue
                    cnt_post = struct.unpack_from("<I", v, off)[0]
                    off += 4
                    post_ids = struct.unpack_from(
                        f"<{cnt_post}Q", v, off) if cnt_post else ()
                    for p in pre_ids:
                        for q in post_ids:
                            e_pre.append(p)
                            e_post.append(q)
                            e_typ.append(typ)
                if len(e_pre) >= PART_EVERY:
                    part += 1
                    np.savez_compressed(
                        os.path.join(OUT, f"edgesV5_part{part:03d}.npz"),
                        pre=np.array(e_pre, dtype=np.uint64),
                        post=np.array(e_post, dtype=np.uint64),
                        typ=np.array(e_typ, dtype=np.uint32))
                    e_pre, e_post, e_typ = [], [], []
        print(json.dumps({"shard": shard + 1, "edges": PART_EVERY
                          * part + len(e_pre),
                          "elapsed_s": round(time.time() - t0)}),
            flush=True)
    if e_pre:
        part += 1
        np.savez_compressed(
            os.path.join(OUT, f"edgesV5_part{part:03d}.npz"),
            pre=np.array(e_pre, dtype=np.uint64),
            post=np.array(e_post, dtype=np.uint64),
            typ=np.array(e_typ, dtype=np.uint32))
    print(json.dumps({"H01-EDGES-V5-COMPLETE": True,
                      "edges": PART_EVERY * part + len(e_pre)}),
          flush=True)


if __name__ == "__main__":
    main()
