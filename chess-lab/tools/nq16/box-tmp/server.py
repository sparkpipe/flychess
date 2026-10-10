#!/usr/bin/env python
"""Qwen-Image-2.1 text-to-image web app.

Two quality modes, switchable at runtime (each switch reloads from the
local HF cache, ~1 min):

  speed   (default): transformer int8 weight-only resident on GPU (~7.5 GB),
          VAE bf16 resident, text encoder int8 streamed to GPU in block
          groups during the single per-prompt encode pass (~2 GB GPU peak).
          Peak VRAM @1536² ≈ 22 GB, ~41 s per image.

  quality (no quantization): full bf16 via diffusers component CPU
          offloading — each big component moves over PCIe once per image.
          Peak VRAM @1536² ≈ 27.5 GB, needs the card mostly to itself.

Every generation is saved to outputs/ with a thumbnail and logged to
outputs/gallery.jsonl for the gallery view.
"""
import gc
import json
import random
import re
import threading
import time
import uuid
from pathlib import Path

import torch
from flask import Flask, jsonify, request, send_from_directory
from PIL import Image
from diffusers import QwenImage21Pipeline
from diffusers.hooks.group_offloading import apply_group_offloading
from torchao.quantization import Int8WeightOnlyConfig, quantize_

MIN_SIDE, MAX_SIDE = 1536, 2752
HERETIC_PATH = Path("/mnt/model-warm/qwen-image-2.1-text-encoder-heretic-pottokao")

ROOT = Path(__file__).resolve().parent
OUT = ROOT / "outputs"
THUMB = OUT / "thumb"
UPLOADS = OUT / "uploads"
OUT.mkdir(parents=True, exist_ok=True)
THUMB.mkdir(parents=True, exist_ok=True)
UPLOADS.mkdir(parents=True, exist_ok=True)
META = OUT / "gallery.jsonl"

app = Flask(__name__, static_folder="static", static_url_path="/static")

_state = {"status": "loading model", "busy": False, "mode": "speed"}
_gpu_lock = threading.Lock()
_meta_lock = threading.Lock()
pipe = None

_NAME_RE = re.compile(r"^\d{8}-\d{6}-[a-f0-9]{6}\.png$")


def _group_offload(module):
    try:
        apply_group_offloading(
            module, onload_device="cuda",
            offload_type="block_level", num_blocks_per_group=4, use_stream=True,
        )
    except Exception as e:
        print(f"default block detection failed ({e}); retrying with explicit blocks", flush=True)
        apply_group_offloading(
            module, onload_device="cuda",
            offload_type="block_level", num_blocks_per_group=4, use_stream=True,
            block_modules=["model.language_model.layers", "model.visual.blocks"],
        )


def _apply_layout(p, mode):
    """Placement for an already-quantized pipeline (see configure_mode)."""
    if mode == "speed":
        p.transformer.to("cuda")
        p.vae.to("cuda")
        _group_offload(p.text_encoder)
    elif mode == "eco":
        _group_offload(p.transformer)
        if hasattr(p.vae, "enable_tiling"):
            p.vae.enable_tiling()
        p.vae.to("cuda")
        _group_offload(p.text_encoder)
    else:  # quality — zero quantization, everything streamed, tiled decode
        _group_offload(p.transformer)
        if hasattr(p.vae, "enable_tiling"):
            p.vae.enable_tiling()
        p.vae.to("cuda")
        _group_offload(p.text_encoder)


def _mem_available_gib():
    with open("/proc/meminfo") as f:
        for line in f:
            if line.startswith("MemAvailable"):
                return int(line.split()[1]) / 2**20
    return 0.0


def _own_rss_gib():
    with open("/proc/self/status") as f:
        for line in f:
            if line.startswith("VmRSS"):
                return int(line.split()[1]) / 2**20
    return 0.0


def _free_pipe():
    global pipe
    if pipe is not None:
        del pipe
        pipe = None
        gc.collect()
        gc.collect()
        torch.cuda.empty_cache()
        try:
            import ctypes
            ctypes.CDLL("libc.so.6").malloc_trim(0)
        except Exception:
            pass


# incremental-load RAM transients (GiB): each component is quantized (or moved
# to GPU) before the next one loads, so the peak is roughly one bf16 component
# plus what's already resident
_LOAD_TRANSIENT = {
    ("speed", False): 21, ("speed", True): 22,
    ("eco", False): 27, ("eco", True): 28,
    ("quality", False): 34, ("quality", True): 35,
}
_CONFIG = ROOT / "config.json"
REPO = "Qwen/Qwen-Image-2.1"


def configure_mode(mode, heretic=False):
    """(Re)load the pipeline from the local cache in the requested mode.

    Loads components incrementally (quantize each before loading the next) to
    keep the RAM transient low, and measures how much of the previous pipeline
    actually released rather than assuming the worst."""
    global pipe
    _free_pipe()
    residual = max(0.0, _own_rss_gib() - 4)  # 4 GiB ≈ bare interpreter
    avail = _mem_available_gib()
    need = _LOAD_TRANSIENT[(mode, heretic)] + residual
    if avail < need:
        raise ValueError(
            f"not enough free RAM: {avail:.0f} GB available, loading {mode}"
            f"{'/heretic' if heretic else ''} needs ~{need:.0f} GB"
            + (f" (incl. {residual:.0f} GB from the previous model not yet released — try again in a moment)" if residual > 2 else "")
            + " — pick a lighter config or restart the service"
        )
    t0 = time.time()
    from diffusers import QwenImage21Transformer2DModel
    from transformers.models.qwen3_vl import Qwen3VLForConditionalGeneration

    # transformer first: quantize immediately; in speed mode move to GPU now,
    # which drops it out of RAM before the encoder loads
    transformer = QwenImage21Transformer2DModel.from_pretrained(
        REPO, subfolder="transformer", torch_dtype=torch.bfloat16
    )
    if mode != "quality":
        quantize_(transformer, Int8WeightOnlyConfig())
    if mode == "speed":
        transformer.to("cuda")

    if heretic:
        if not HERETIC_PATH.exists():
            raise RuntimeError(f"heretic encoder not found at {HERETIC_PATH}")
        print("loading heretic text encoder from warm storage…", flush=True)
        text_encoder = Qwen3VLForConditionalGeneration.from_pretrained(
            HERETIC_PATH, dtype=torch.bfloat16
        )
    else:
        text_encoder = Qwen3VLForConditionalGeneration.from_pretrained(
            REPO, subfolder="text_encoder", torch_dtype=torch.bfloat16
        )
    if mode != "quality":
        quantize_(text_encoder, Int8WeightOnlyConfig())

    p = QwenImage21Pipeline.from_pretrained(
        REPO, torch_dtype=torch.bfloat16,
        transformer=transformer, text_encoder=text_encoder,
    )
    _apply_layout(p, mode)
    pipe = p
    _state["mode"] = mode
    _state["heretic"] = heretic
    _CONFIG.write_text(json.dumps({"mode": mode, "heretic": heretic}))
    print(f"model ready [{mode}{'/heretic' if heretic else ''}] in {time.time() - t0:.0f}s", flush=True)


_saved = {"mode": "speed", "heretic": False}
if _CONFIG.exists():
    try:
        _saved.update(json.loads(_CONFIG.read_text()))
    except Exception:
        pass
try:
    configure_mode(_saved["mode"], bool(_saved["heretic"]))
except Exception as e:
    print(f"saved config failed ({e}); falling back to speed", flush=True)
    configure_mode("speed", False)
_state["status"] = "ready"


def clamp_side(v):
    return max(MIN_SIDE, min(MAX_SIDE, round(v / 32) * 32))


@app.get("/")
def index():
    return send_from_directory(ROOT / "static", "index.html")


@app.get("/api/status")
def status():
    s = dict(_state)
    try:
        free, _total = torch.cuda.mem_get_info()
        s["free_vram_gib"] = round(free / 2**30, 1)
    except Exception:
        s["free_vram_gib"] = None
    return jsonify(s)


@app.post("/api/mode")
def set_mode():
    body = request.get_json(force=True)
    mode = (body.get("mode") or "").strip()
    heretic = bool(body.get("heretic"))
    if mode not in ("speed", "quality", "eco"):
        return jsonify(error="mode must be 'speed', 'quality' or 'eco'"), 400
    if heretic == _state.get("heretic") and mode == _state["mode"] and pipe is not None:
        return jsonify(mode=mode, heretic=heretic)
    if _gpu_lock.locked():
        return jsonify(error="generation in progress — try again after it finishes"), 409
    with _gpu_lock:
        _state["status"] = f"switching to {mode}{'/heretic' if heretic else ''}"
        try:
            configure_mode(mode, heretic)
            _state["status"] = "ready"
        except ValueError as e:
            # If a fresh process would fit this config, switch by restarting
            # ourselves: the OS reclaims everything the old process held
            # (the GC can't always release the previous pipeline).
            if _mem_available_gib() + _own_rss_gib() - 2 >= _LOAD_TRANSIENT[(mode, heretic)]:
                _CONFIG.write_text(json.dumps({"mode": mode, "heretic": heretic}))
                _state["status"] = "restarting"
                print(f"switching to {mode}/{'heretic' if heretic else 'stock'} via service restart", flush=True)
                import subprocess
                subprocess.Popen(
                    ["/bin/sh", "-c", "sleep 1.5; systemctl restart qwen-image-web"],
                    start_new_session=True,
                )
                return jsonify(
                    restarting=True, mode=mode, heretic=heretic,
                    note="applying via service restart — back in ~1 min",
                ), 202
            _state["status"] = "ready" if pipe is not None else "unloaded"
            return jsonify(error=str(e)), 409
        except Exception as e:
            print(f"mode switch failed: {e}", flush=True)
            try:
                configure_mode(_state["mode"], _state.get("heretic", False))
                _state["status"] = "ready"
                return jsonify(error=f"switch failed ({e}); reverted to previous config"), 500
            except Exception:
                _state["status"] = "error — restart needed"
                return jsonify(error=f"switch failed and fallback failed: {e}"), 500
    return jsonify(mode=mode, heretic=heretic)


@app.get("/api/images")
def images():
    if not META.exists():
        return jsonify([])
    records = [json.loads(line) for line in META.read_text().splitlines() if line.strip()]
    records.reverse()  # newest first
    for r in records:
        r["exists"] = (OUT / r["file"]).exists()
    return jsonify(records)


# additional VRAM needed on top of what's already resident / streamed:
# activations scale with pixel count; tiled VAE decode shaves ~4 GB above 1536²
def _needed_gib(mode, w, h):
    act = 13.2 * (w * h) / (1536 * 1536)
    if mode == "speed":
        # int8 transformer + VAE already resident before the call
        return act + 1.5 - (4 if w * h > 1536 * 1536 else 0)
    # quality (bf16 streamed) and eco (int8 streamed): weights stream in
    # block groups, so the peak is activation-dominated
    return 4 + act * 0.9


def _snap_edit_image(img):
    """Resize an uploaded edit source into the model's supported range
    (both sides ~1536–2752, /32) while preserving aspect as far as possible."""
    w, h = img.size
    scale = min(MAX_SIDE / max(w, h), 1.0)
    if min(w, h) * scale < MIN_SIDE:
        scale = MIN_SIDE / min(w, h)
    w, h = (round(w * scale / 32) * 32, round(h * scale / 32) * 32)
    if (w, h) != img.size:
        img = img.resize((w, h), Image.LANCZOS)
    return img.convert("RGB")


@app.route("/api/images/<name>", methods=["DELETE"])
def delete_image(name):
    if not _NAME_RE.match(name):
        return jsonify(error="invalid image name"), 400
    img, thumb = OUT / name, THUMB / (name + ".jpg")
    if not img.exists():
        return jsonify(error="not found"), 404
    img.unlink(missing_ok=True)
    thumb.unlink(missing_ok=True)
    with _meta_lock:
        if META.exists():
            kept = []
            for line in META.read_text().splitlines():
                if not line.strip():
                    continue
                try:
                    if json.loads(line).get("file") == name:
                        continue
                except Exception:
                    pass  # keep malformed lines as-is
                kept.append(line)
            META.write_text("\n".join(kept) + ("\n" if kept else ""))
    return jsonify(ok=True, deleted=name)


@app.post("/api/generate")
def generate():
    if pipe is None:
        return jsonify(error="model not loaded — pick a config and Apply, or restart the service"), 503
    if _gpu_lock.locked():
        return jsonify(error="a generation is already running — wait for it to finish"), 409
    if request.content_type and request.content_type.startswith("multipart/"):
        data = request.form.to_dict()
        upload = request.files.get("image")
    else:
        data = request.get_json(force=True)
        upload = None
    prompt = (data.get("prompt") or "").strip()
    if not prompt:
        return jsonify(error="prompt is required"), 400

    mode = _state["mode"]
    edit_image = None
    upload_name = None
    if upload is not None:
        try:
            edit_image = _snap_edit_image(Image.open(upload.stream))
        except Exception:
            return jsonify(error="attached file is not a readable image"), 400
        upload_name = f"{time.strftime('%Y%m%d-%H%M%S')}-src-{uuid.uuid4().hex[:6]}.png"
        edit_image.save(UPLOADS / upload_name)
        # editing follows the source image's snapped size; ignore w/h fields
        width, height = edit_image.size
    else:
        width = clamp_side(int(data.get("width", 1536)))
        height = clamp_side(int(data.get("height", 1536)))
    steps = max(4, min(60, int(data.get("steps", 30))))
    cfg = float(data.get("cfg", 1.0))
    seed = data.get("seed")
    seed = int(seed) if seed not in (None, "", 0) else random.randint(0, 2**31 - 1)

    try:
        free, _ = torch.cuda.mem_get_info()
        need = _needed_gib(mode, width, height)
        if free / 2**30 < need:
            return jsonify(
                error=f"only {free / 2**30:.1f} GB free — {mode} mode at {width}×{height} "
                      f"needs ≈{need:.0f} GB more; try eco mode, 1536², or fewer steps"
            ), 507
    except Exception:
        pass

    with _gpu_lock:
        _state["busy"] = True
        _state["status"] = "generating"
        try:
            # tile the VAE decode above 1536² in speed mode; quality and eco
            # tile at load time
            big = width * height > 1536 * 1536 and _state["mode"] == "speed"
            if big and hasattr(pipe.vae, "enable_tiling"):
                pipe.vae.enable_tiling()
            elif not big and _state["mode"] == "speed" and hasattr(pipe.vae, "disable_tiling"):
                pipe.vae.disable_tiling()
            torch.cuda.reset_peak_memory_stats()
            t0 = time.time()
            image = pipe(
                prompt=prompt,
                image=edit_image,
                width=width,
                height=height,
                num_inference_steps=steps,
                true_cfg_scale=cfg,
                generator=torch.Generator("cuda").manual_seed(seed),
            ).images[0]
            seconds = time.time() - t0
            peak_gib = torch.cuda.max_memory_allocated() / 2**30

            name = f"{time.strftime('%Y%m%d-%H%M%S')}-{uuid.uuid4().hex[:6]}.png"
            image.save(OUT / name)
            thumb = image.convert("RGB")
            thumb.thumbnail((320, 320))
            thumb.save(THUMB / (name + ".jpg"), "JPEG", quality=85)

            record = {
                "file": name,
                "thumb": f"thumb/{name}.jpg",
                "prompt": prompt,
                "seed": seed,
                "width": width,
                "height": height,
                "steps": steps,
                "cfg": cfg,
                "mode": mode,
                "heretic": _state.get("heretic", False),
                "edit": bool(edit_image),
                "source": upload_name,
                "seconds": round(seconds, 1),
                "vram_gib": round(peak_gib, 2),
                "ts": time.time(),
            }
            with META.open("a") as f:
                f.write(json.dumps(record) + "\n")
            return jsonify(record)
        except torch.cuda.OutOfMemoryError:
            torch.cuda.empty_cache()
            return jsonify(
                error="GPU out of memory — image edits need extra VRAM for the vision pass; "
                      "try eco mode, 1536², or fewer steps"
                if edit_image else
                "GPU out of memory — try eco mode, 1536×1536, or fewer steps; "
                "the shared GPU may be too full for this size"
            ), 507
        finally:
            torch.cuda.empty_cache()
            _state["busy"] = False
            _state["status"] = "ready"


@app.get("/outputs/<path:name>")
def output_file(name):
    return send_from_directory(OUT, name)


if __name__ == "__main__":
    app.run(host="100.123.97.61", port=7860, threaded=True)
