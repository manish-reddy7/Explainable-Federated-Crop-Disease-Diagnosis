"""
FastAPI inference + Grad-CAM overlay (project plan backend).

Usage (from `XFedCrop/`):
  uvicorn api.app:app --reload --host 0.0.0.0 --port 8000
"""

from __future__ import annotations

import io
import os
import sys
from contextlib import asynccontextmanager
from pathlib import Path
import logging

import numpy as np
import torch
import torch.nn as nn
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, JSONResponse, Response
from fastapi.staticfiles import StaticFiles
from PIL import Image

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.DEBUG)

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from explainability.grad_cam import (  # noqa: E402
    compute_grad_cam,
    overlay_heatmap_on_image,
    pil_to_model_tensor,
)
from models.base import build_model  # noqa: E402

@asynccontextmanager
async def _lifespan(_app: FastAPI):
    global _model
    try:
        load_checkpoint()
    except FileNotFoundError:
        _model = None
    yield


app = FastAPI(title="XFedCrop API", version="0.1.0", lifespan=_lifespan)
WEB_DIR = ROOT / "agri-ai-insights-main" / "dist"

# Enable CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

_model: nn.Module | None = None
_class_names: list[str] = []
_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _ensure_model_loaded() -> None:
    global _model
    if _model is not None:
        return
    try:
        load_checkpoint()
    except FileNotFoundError as e:
        raise HTTPException(503, f"Model checkpoint not found: {e}") from e
    except Exception as e:
        raise HTTPException(503, f"Model load failed: {e}") from e


def load_checkpoint(path: str | None = None) -> None:
    global _model, _class_names
    ck = path or os.environ.get("XFEDCROP_CHECKPOINT", str(ROOT / "checkpoints" / "central_resnet18_best.pt"))
    if not os.path.isfile(ck):
        raise FileNotFoundError(f"Checkpoint not found: {ck}")
    try:
        data = torch.load(ck, map_location=_device, weights_only=False)
    except TypeError:
        data = torch.load(ck, map_location=_device)
    _class_names = list(data.get("class_names", []))
    n = int(data.get("num_classes", len(_class_names)))
    if not _class_names and n > 0:
        _class_names = [f"class_{i}" for i in range(n)]
    _model = build_model(n, pretrained=False).to(_device)
    _model.load_state_dict(data["model_state_dict"], strict=True)
    _model.eval()


if (WEB_DIR / "assets").is_dir():
    app.mount("/assets", StaticFiles(directory=str(WEB_DIR / "assets")), name="assets")


@app.get("/", response_class=HTMLResponse)
def index() -> str:
    index_file = WEB_DIR / "index.html"
    logger.debug(f"Serving index from: {index_file}")
    logger.debug(f"File exists: {index_file.is_file()}")
    if not index_file.is_file():
        raise HTTPException(500, f"Web UI not found. Expected {index_file}")
    try:
        content = index_file.read_text(encoding="utf-8")
        logger.debug(f"Read {len(content)} bytes from index.html")
        return content
    except Exception as e:
        logger.error(f"Error reading index.html: {e}", exc_info=True)
        raise HTTPException(500, f"Failed to read index: {e}") from e


@app.get("/status")
def status():
    return {
        "ok": True,
        "model_loaded": _model is not None,
        "device": str(_device),
        "num_classes": len(_class_names) if _class_names else 0,
    }


@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    _ensure_model_loaded()
    model = _model
    if model is None:
        raise HTTPException(503, "Model not loaded")
    raw = await file.read()
    try:
        pil = Image.open(io.BytesIO(raw)).convert("RGB")
    except Exception as e:
        raise HTTPException(400, f"Invalid image: {e}") from e
    x = pil_to_model_tensor(pil).to(_device)
    with torch.no_grad():
        logits = model(x)
        prob = torch.softmax(logits, dim=1)[0]
    conf, idx = prob.max(dim=0)
    label = _class_names[int(idx)] if _class_names else str(int(idx))
    return JSONResponse(
        {
            "label": label,
            "class_index": int(idx),
            "confidence": float(conf),
            "probabilities_top5": _top5(prob),
        }
    )


def _top5(prob: torch.Tensor) -> list[dict]:
    vals, idx = torch.topk(prob, k=min(5, prob.numel()))
    out = []
    for v, i in zip(vals.tolist(), idx.tolist()):
        name = _class_names[i] if _class_names else str(i)
        out.append({"class_index": i, "label": name, "confidence": float(v)})
    return out


def _heatmap_region(heatmap: np.ndarray) -> str:
    h, w = heatmap.shape
    y, x = np.unravel_index(int(np.argmax(heatmap)), heatmap.shape)
    horiz = "left" if x < w / 3 else ("right" if x > (2 * w / 3) else "center")
    vert = "upper" if y < h / 3 else ("lower" if y > (2 * h / 3) else "middle")
    return f"{vert}-{horiz}"


def _confidence_band(conf: float) -> str:
    if conf >= 0.9:
        return "very_high"
    if conf >= 0.75:
        return "high"
    if conf >= 0.55:
        return "medium"
    return "low"


def _attention_profile(hm: np.ndarray) -> dict:
    h, w = hm.shape
    h_step = h // 3
    w_step = w // 3
    cells = {
        "upper-left": hm[:h_step, :w_step],
        "upper-center": hm[:h_step, w_step : 2 * w_step],
        "upper-right": hm[:h_step, 2 * w_step :],
        "middle-left": hm[h_step : 2 * h_step, :w_step],
        "middle-center": hm[h_step : 2 * h_step, w_step : 2 * w_step],
        "middle-right": hm[h_step : 2 * h_step, 2 * w_step :],
        "lower-left": hm[2 * h_step :, :w_step],
        "lower-center": hm[2 * h_step :, w_step : 2 * w_step],
        "lower-right": hm[2 * h_step :, 2 * w_step :],
    }
    scores = {k: float(v.mean()) for k, v in cells.items()}
    ranked = sorted(scores.items(), key=lambda kv: kv[1], reverse=True)
    return {
        "region_scores": scores,
        "top_regions": [name for name, _ in ranked[:2]],
    }


def _summarize_heatmap(heatmap: np.ndarray, label: str, conf: float, alternatives: list[dict]) -> dict:
    hotspot_ratio = float((heatmap >= 0.6).mean())
    center = heatmap[heatmap.shape[0] // 3 : (2 * heatmap.shape[0]) // 3, heatmap.shape[1] // 3 : (2 * heatmap.shape[1]) // 3]
    center_focus = float(center.mean() / max(float(heatmap.mean()), 1e-8))
    region = _heatmap_region(heatmap)
    profile = _attention_profile(heatmap)
    conf_band = _confidence_band(conf)

    if hotspot_ratio < 0.18:
        attention_style = "diffuse"
    elif hotspot_ratio > 0.4:
        attention_style = "highly_localized"
    else:
        attention_style = "moderately_localized"

    alt_txt = ""
    if alternatives:
        top_alt = alternatives[0]
        alt_txt = f" The closest alternative was {top_alt['label']} ({top_alt['confidence'] * 100:.1f}%)."

    explanation = (
        f"The model predicted {label} with {conf * 100:.1f}% confidence ({conf_band.replace('_', ' ')} confidence). "
        f"Grad-CAM attention is strongest in the {region} area and is {attention_style.replace('_', ' ')}. "
        f"About {hotspot_ratio * 100:.1f}% of the image contains strong evidence (hotspot >= 0.6), and center-focus is {center_focus:.2f}x."
        f" This suggests the decision relied on lesion-specific regions rather than the full leaf.{alt_txt}"
    )
    return {
        "hotspot_ratio": hotspot_ratio,
        "center_focus_ratio": center_focus,
        "hotspot_region": region,
        "attention_style": attention_style,
        "confidence_band": conf_band,
        "top_attention_regions": profile["top_regions"],
        "region_scores": profile["region_scores"],
        "top_alternatives": alternatives,
        "explanation": explanation,
    }


@app.post("/explain/gradcam")
async def explain_gradcam(file: UploadFile = File(...)):
    _ensure_model_loaded()
    model = _model
    if model is None:
        raise HTTPException(503, "Model not loaded")
    raw = await file.read()
    try:
        pil = Image.open(io.BytesIO(raw)).convert("RGB")
    except Exception as e:
        raise HTTPException(400, f"Invalid image: {e}") from e
    x = pil_to_model_tensor(pil).to(_device)
    pred = int(model(x).argmax(dim=1).item())
    hm = compute_grad_cam(model, x, target_class=pred)
    # resize cam to tensor spatial size then overlay uses pil size
    overlay = overlay_heatmap_on_image(pil, hm)
    buf = io.BytesIO()
    overlay.save(buf, format="PNG")
    return Response(content=buf.getvalue(), media_type="image/png")


@app.post("/explain/summary")
async def explain_summary(file: UploadFile = File(...)):
    _ensure_model_loaded()
    model = _model
    if model is None:
        raise HTTPException(503, "Model not loaded")
    raw = await file.read()
    try:
        pil = Image.open(io.BytesIO(raw)).convert("RGB")
    except Exception as e:
        raise HTTPException(400, f"Invalid image: {e}") from e

    x = pil_to_model_tensor(pil).to(_device)
    with torch.no_grad():
        logits = model(x)
        prob = torch.softmax(logits, dim=1)[0]
    conf, idx = prob.max(dim=0)
    pred = int(idx)
    hm = compute_grad_cam(model, x, target_class=pred)
    top5 = _top5(prob)
    alternatives = [p for p in top5 if p["class_index"] != pred][:3]
    label = _class_names[pred] if _class_names else str(pred)
    summary = _summarize_heatmap(hm, label=label, conf=float(conf), alternatives=alternatives)

    return JSONResponse(
        {
            "label": label,
            "class_index": pred,
            "confidence": float(conf),
            **summary,
        }
    )
