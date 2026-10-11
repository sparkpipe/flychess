"""Fix calibration: fold normalization into stored constants so the
engine's raw uint8 → dot product matches the trainer's output."""
import numpy as np
import torch, struct

CACHE = "/srv/workspace/flychess/cache"
R = "/mnt/cold-raid6/chess-audit"

calib = np.load(CACHE + "/calibration.npz")
scale, zero = calib["scale"], calib["zero"]
E = len(scale)

state = torch.load(R + "/head_v1.pt", map_location="cpu", weights_only=False)
gates = state["gates"].numpy()
lw = state["lin.weight"].numpy().flatten()
lb = state["lin.bias"].numpy().flatten()

# The trainer normalized by dividing by mean absolute value of calibrated activations.
# Compute what that constant approximately is:
# calibrated = uint8 * scale + zero; mean_abs ≈ mean(uint8)*scale + |zero| for uniform uint8
# But more accurately: sample some actual calibrated values
sample_files = ["balanced_l0", "nvb", "dvoretsky"]
vals = []
for name in sample_files:
    feats = np.load(CACHE + "/" + name + ".uint8.npy", mmap_mode="r")
    sample = feats[:100].astype(np.float32)
    sc_p = np.pad(scale, (0, 14-len(scale)), constant_values=1.0)
    ze_p = np.pad(zero, (0, 14-len(zero)), constant_values=0.0)
    cal = sample * sc_p[None, :, None] + ze_p[None, :, None]
    vals.append(np.abs(cal).mean())
norm_const = np.mean(vals)
print("norm constant:", norm_const)

# Fold 1/norm_const into activation linear weights
D = 1024
H = 36
act_count = D * 14
lw_adj = lw.copy()
lw_adj[:act_count] /= norm_const

# Store scale/zero divided by norm_const so engine gets pre-normalized values
adj_scale = (sc_p / norm_const).astype(np.float32)
adj_zero = (ze_p / norm_const).astype(np.float32)

with open(R + "/head_v1_cal.stkh", "wb") as f:
    f.write(struct.pack("<IIIII", 0x53544B48, 1, E, D, H))
    f.write(struct.pack("<I", 13))
    f.write(gates.astype(np.float32).tobytes())
    f.write(lw_adj.astype(np.float32).tobytes())
    f.write(lb.astype(np.float32).tobytes())
    full_scale = np.ones(14, dtype=np.float32)
    full_zero = np.zeros(14, dtype=np.float32)
    full_scale = adj_scale
    full_zero = adj_zero
    f.write(full_scale.tobytes())
    f.write(full_zero.tobytes())

print("saved:", R + "/head_v1_cal.stkh")
print("adj_scale:", adj_scale[:3])
print("adj_zero:", adj_zero[:3])
print("weight norm ratio:", np.linalg.norm(lw_adj[:act_count]) / np.linalg.norm(lw[:act_count]))
