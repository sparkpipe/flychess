"""H01 full extraction v3 (VERIFIED construction):
- per-shard list_labels (path="") enumerates all synapse ids
- batched get_by_id pulls resolve pre/post/type
- checkpointed edge parts under h01_edges/
"""
import os, sys, json, time
import numpy as np
sys.path.insert(0, "/home/spec/chess-lab")
from cloudvolume.datasource.precomputed import create_precomputed_annotation
from cloudvolume.datasource.precomputed.sharding import ShardReader, ShardingSpecification
import os as _os
_os.environ.setdefault("CLOUD_FILES_LOCK_DIR", "/dev/shm/cloudfiles-locks")
_os.makedirs("/dev/shm/cloudfiles-locks", exist_ok=True)
from cloudfiles import CloudFiles

ROOT = "/mnt/model-warm/human-h01-connectome/data/20210601/c3/synapses/precomputed"

# cloud-volume 12.14.4 bug: np.hstack(a, b) misuse in the LINE decoder
_orig_hstack = np.hstack
def _hstack_shim(*args):
    if len(args) == 1:
        return _orig_hstack(args[0])
    return _orig_hstack(list(args))
np.hstack = _hstack_shim
OUT = "/home/spec/chess-lab/h01_edges"
BATCH = 2000
CKPT_EVERY = 50            # batches per checkpoint part


class _LocalCacheShim:
    def __init__(self, cf):
        self.cf = cf
    def download_as(self, requests, progress=False):
        out = {}
        for r in requests:
            p = os.path.join(ROOT, r["path"])
            with open(p, "rb") as f:
                f.seek(r["start"])
                out[(r["path"], r["start"], r["end"])] = f.read(r["end"] - r["start"])
        return out


def mk():
    # INODE LAW: cloudfiles mints one never-unlinked 0-byte lock per chunk
    # for file-protocol reads (single-writer local ceph here — locking is
    # pointless); ~820/s leaked = 14M inodes. locking=False + a per-part
    # clear_locks sweep keeps the working set at zero transient files.
    cf = CloudFiles("precomputed://" + ROOT, locking=False)
    cache = _LocalCacheShim(cf)
    spec = ShardingSpecification.from_dict(
        json.load(open("/home/spec/chess-lab/h01_synapses_info.json"))["by_id"]["sharding"])
    reader = ShardReader("precomputed://" + ROOT, cache, spec)
    return cf, reader


def enumerate_ids(cf, reader):
    names = sorted(x for x in cf.list(prefix="by_id", flat=True)
                   if x.endswith(".shard"))
    all_ids = []
    for fn in names:
        labels = reader.list_labels(fn, path="")
        all_ids.append(np.asarray(labels, dtype=np.uint64))
        print(json.dumps({"shard": fn, "labels": int(len(labels))}), flush=True)
    return np.concatenate(all_ids) if all_ids else np.zeros(0, np.uint64)


def main():
    cf, reader = mk()
    ids = enumerate_ids(cf, reader)
    total = len(ids)
    print(json.dumps({"total_ids": total}), flush=True)
    ann = create_precomputed_annotation(
        "precomputed://" + ROOT,
        {"bounded": True, "fill_missing": True, "progress": False,
         "parallel": 1, "mip": 0,
         "cache": _LocalCacheShim(cf)})
    os.makedirs(OUT, exist_ok=True)
    part = 0
    pre_l, post_l, typ_l = [], [], []
    t0 = time.time()
    for i in range(0, total, BATCH):
        batch = ids[i:i + BATCH]
        anns = ann.get_by_id(list(batch))
        for a in anns:
            pre = getattr(a, "pre_synaptic_cell", None)
            post = getattr(a, "post_synaptic_cell", None)
            if pre is None or post is None:
                continue
            pre_l.append(int(pre))
            post_l.append(int(post))
            typ_l.append(int(a.type))
        done = min(i + BATCH, total)
        if (i // BATCH) % 10 == 0:
            print(json.dumps({"ids_done": done, "edges": len(pre_l),
                              "elapsed_s": round(time.time() - t0)}), flush=True)
        if (i // BATCH) % CKPT_EVERY == CKPT_EVERY - 1 or done >= total:
            part += 1
            fpath = os.path.join(OUT, f"edges_part{part:03d}.npz")
            try:
                cf.clear_locks()
            except Exception:
                pass
            np.savez_compressed(fpath,
                pre=np.array(pre_l, dtype=np.uint64),
                post=np.array(post_l, dtype=np.uint64),
                typ=np.array(typ_l, dtype=np.uint8))
            print(f"CHECKPOINT {fpath}: {len(pre_l)} edges", flush=True)
            pre_l, post_l, typ_l = [], [], []
    print("H01-EXTRACT-COMPLETE", flush=True)


if __name__ == "__main__":
    main()
