"""Per-epoch metrics measured against the BIN'S OWN label field — no TSV
reconstruction, no perspective guessing: stream oppb.val.bin through each
exported checkpoint with the fork's own data pipeline.
"""
import sys, os, glob
import torch

NNUE = "/srv/workspace/flychess/src/nnue-pytorch"
sys.path.insert(0, NNUE)
os.chdir(NNUE)

from model import NNUE
import lightning.pytorch as pl

CKPTS = sorted(glob.glob("/extnvme/active/train16_test_oppb/lightning_logs/version_0/checkpoints/last.ckpt"))
VAL = "/extnvme/active/train16/oppb.val.bin"

# fresh-epoch nets live as .nnue; per-epoch metrics need ckpts — the trainer
# only kept last.ckpt (epoch files deleted post-export). Use last.ckpt +
# the .nnue files through torch via serialize's reverse? Simpler: load last.ckpt
# and ALSO e0 isn't available as ckpt... measure last.ckpt now; per-epoch via
# .nnue-in-engine comes separately.
ck = torch.load(CKPTS[0], map_location="cpu", weights_only=False)
print("ckpt epoch:", ck.get("epoch"))

# data loader (fork's own)
from data_loader import TrainingDataProvider
dl = TrainingDataProvider(
    filenames=[VAL],
    batch_size=8192,
    for_training=False,
    features_name="HalfKAv2_hm-Combined+Threats+PP_3Wide",
    val_only=True,
)

model = NNUE.load_from_checkpoint(CKPTS[0], map_location="cpu")
model.eval()

preds, scores = [], []
with torch.no_grad():
    for batch in dl:
        # fork batch layout: (stm, w, b, targets...) — introspect first batch
        break
print("batch type:", type(batch), len(batch) if hasattr(batch, "__len__") else "?")
