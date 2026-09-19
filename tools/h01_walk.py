"""H01 edge walker v4: walk the relationship volumes' own minishard
indices directly (NOT the enumerated id stream — partner-bearing synapses
live at id >= 3e8 and the stream never reached them).

Per the cracked decode (2026-09-18): each relationship volume (pre/
post_synaptic_cell) declares its OWN sharding spec in info.relationships;
entries decode from 44-byte payloads:
  bytes 0-7   annotation id (u64)
  bytes 8-19  point A xyz (u32 x3, volume coords)
  bytes 20-31 point B xyz (u32 x3)
  bytes 32-35 type (u32: 1=inhibitory, 2=excitatory)
  bytes 36-43 PARTNER SEGMENT ID (u64)

Emits per-relationship (sid, partner, type) parts, then joins on sid:
an edge = a synapse with both pre and post entries.
"""
import os
import sys
import json
import time
import gzip
import struct
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
NSHARD = {"pre_synaptic_cell": 16, "post_synaptic_cell": 16}


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
            p = os.path.join(self.root, r["path"])
            with open(p, "rb") as f:
                f.seek(r["start"])
                out[(r["path"], r["start"], r["end"])] = \
                    f.read(r["end"] - r["start"])
        return out


def walk_relationship(rid, spec_d):
    spec = ShardingSpecification.from_dict(spec_d)
    rdir = ROOT + "/" + rid
    cf = cloudfiles.CloudFiles("file://" + rdir)
    r = ShardReader("file://" + rdir, Shim(cf, rdir), spec)
    sids, partners, types_ = [], [], []
    t0 = time.time()
    for shard in range(NSHARD[rid]):
        try:
            idx = r.get_index(f"{shard}.shard")
        except Exception:
            continue
        nz = np.nonzero((idx[:, 1] - idx[:, 0]) > 0)[0]
        if not len(nz):
            continue
        shard_path = os.path.join(rdir, f"{shard}.shard")
        with open(shard_path, "rb") as f:
            for ms in nz:
                mi = r.get_minishard_index(f"{shard}.shard", idx, int(ms))
                if mi is None or not len(mi):
                    continue
                # FAST PATH (verified byte-identical to get_data): rows
                # are (segid, absolute_offset, length); one contiguous
                # read per minishard, slice + gunzip per entry — the
                # per-id reader cost ~30 min/shard, this is IO-bound
                lo = int(mi[:, 1].min())
                hi = int((mi[:, 1] + mi[:, 2]).max())
                f.seek(lo)
                blob = f.read(hi - lo)
                for row in mi:
                    segid = int(row[0])
                    off = int(row[1]) - lo
                    ln = int(row[2])
                    v = blob[off:off + ln]
                    if len(v) < 44:
                        continue
                    if v[:1] == b"\x1f":           # gzip magic
                        try:
                            v = gzip.decompress(v)
                        except Exception:
                            continue
                        if len(v) < 44:
                            continue
                    typ = struct.unpack_from("<I", v, 32)[0]
                    partner = struct.unpack_from("<Q", v, 36)[0]
                    sids.append(segid)
                    partners.append(partner)
                    types_.append(typ)
        done = shard + 1
        print(json.dumps({"rel": rid, "shards": done, "entries":
                          len(sids), "elapsed_s":
                          round(time.time() - t0)}), flush=True)
    return (np.array(sids, dtype=np.uint64),
            np.array(partners, dtype=np.uint64),
            np.array(types_, dtype=np.uint32))


def main():
    os.makedirs(OUT, exist_ok=True)
    info = json.load(open(INFO))
    rels = {r["id"]: r["sharding"] for r in info["relationships"]}
    tables = {}
    for rid, spec_d in rels.items():
        sids, partners, types_ = walk_relationship(rid, spec_d)
        tables[rid] = sids
        np.savez_compressed(
            os.path.join(OUT, f"{rid}.npz"),
            sid=sids, partner=partners, typ=types_)
        print(f"WALKED {rid}: {len(sids)} entries", flush=True)
    # join: synapses with BOTH pre and post = directed edges
    pre = dict(zip(tables.get("pre_synaptic_cell", []).tolist(),
                   range(len(tables.get("pre_synaptic_cell", [])))))
    post_t = tables.get("post_synaptic_cell", [])
    pz = np.load(os.path.join(OUT, "post_synaptic_cell.npz"))
    prz = np.load(os.path.join(OUT, "pre_synaptic_cell.npz"))
    post_map = {int(s): (int(p), int(t))
                for s, p, t in zip(pz["sid"], pz["partner"], pz["typ"])}
    e_pre, e_post, e_typ = [], [], []
    for s, p, t in zip(prz["sid"], prz["partner"], prz["typ"]):
        po = post_map.get(int(s))
        if po is None:
            continue
        e_pre.append(int(p))
        e_post.append(po[0])
        e_typ.append(int(t))
    np.savez_compressed(
        os.path.join(OUT, "edges.npz"),
        pre=np.array(e_pre, dtype=np.uint64),
        post=np.array(e_post, dtype=np.uint64),
        typ=np.array(e_typ, dtype=np.uint32))
    print(json.dumps({"H01-EDGES-COMPLETE": True,
                      "edges": len(e_pre)}), flush=True)


if __name__ == "__main__":
    main()
