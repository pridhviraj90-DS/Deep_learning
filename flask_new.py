# flask_new.py
import os
import io
import time
import traceback
from typing import Tuple, Optional, List

import numpy as np
from flask import Flask, request, jsonify, make_response, Response
from PIL import Image, ImageOps
import tensorflow as tf
import keras
# -------------------- Configuration --------------------
MODEL_PATH = os.getenv(
    "MODEL_PATH",
    r"\home\mysqladm\best_model.keras"  # <- change if needed
)
LABELS_ENV = os.getenv("MODEL_LABELS", "").strip()
CLASS_LABELS: Optional[List[str]] = [s.strip() for s in LABELS_ENV.split(",")] if LABELS_ENV else None

# Inconclusive band for binary model
INCONCLUSIVE_LOW  = float(os.getenv("INCONCLUSIVE_LOW",  "0.45"))
INCONCLUSIVE_HIGH = float(os.getenv("INCONCLUSIVE_HIGH", "0.55"))

# Upload constraints
ALLOWED_EXTENSIONS = {"png", "jpg", "jpeg", "bmp", "gif", "webp"}
MAX_UPLOAD_MB = 10
TARGET_SIZE: Tuple[int, int] = (224, 224)

# -------------------- Heuristics (CXR / Doc / Blur) --------------------
# Extreme blur (very lenient; only blocks when almost blank/defocused)
EXTREME_BLUR_VAR = float(os.getenv("EXTREME_BLUR_VAR", "0.0015"))  # Laplacian variance on gray[0..1]
EXTREME_LOW_STD  = float(os.getenv("EXTREME_LOW_STD",  "0.010"))   # global contrast near zero

# CXR acceptance window (very wide / permissive)
CXR_MIN_STD        = float(os.getenv("CXR_MIN_STD",        "0.030"))
CXR_MAX_WHITE_FRAC = float(os.getenv("CXR_MAX_WHITE_FRAC", "0.75"))
CXR_MAX_BLACK_FRAC = float(os.getenv("CXR_MAX_BLACK_FRAC", "0.75"))
CXR_MEAN_LO        = float(os.getenv("CXR_MEAN_LO",        "0.08"))
CXR_MEAN_HI        = float(os.getenv("CXR_MEAN_HI",        "0.95"))
GRAYSCALE_TOL      = float(os.getenv("GRAYSCALE_TOL",      "0.30"))  # 0 strict .. 1 very loose

# Document / prescription detection (stronger):
#  - text-like small connected components density
#  - whiteness, border whiteness, axis dominance as backups
DOC_WHITE_FRAC_MIN = float(os.getenv("DOC_WHITE_FRAC_MIN", "0.55"))
BORDER_WHITE_MIN   = float(os.getenv("BORDER_WHITE_MIN",   "0.55"))
DOC_STD_MAX        = float(os.getenv("DOC_STD_MAX",        "0.28"))
AXIS_DOM_THRESH    = float(os.getenv("AXIS_DOM_THRESH",    "0.50"))  # edge energy near 0/90 deg

# Connected component (text-blob) thresholds (after downscale for analysis)
ANALYZE_MAX_DIM    = int(os.getenv("ANALYZE_MAX_DIM",      "640"))
CC_AREA_MIN        = int(os.getenv("CC_AREA_MIN",          "8"))     # small “ink” specks
CC_AREA_MAX        = int(os.getenv("CC_AREA_MAX",          "1600"))  # ignore huge shapes
CC_DENSITY_MIN     = float(os.getenv("CC_DENSITY_MIN",     "220.0")) # blobs per megapixel to call “doc”

# Minimum decode size (very permissive)
MIN_WIDTH  = int(os.getenv("MIN_WIDTH",  "64"))
MIN_HEIGHT = int(os.getenv("MIN_HEIGHT", "64"))

# Debug output
DEBUG_METRICS = int(os.getenv("DEBUG_METRICS", "1"))

# -------------------- App init --------------------
app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = MAX_UPLOAD_MB * 1024 * 1024

@app.after_request
def add_no_cache_headers(resp):
    resp.headers["Cache-Control"] = "no-store, no-cache, must-revalidate, max-age=0"
    resp.headers["Pragma"] = "no-cache"
    resp.headers["Expires"] = "0"
    return resp

# -------------------- Load model (fail-safe) --------------------
MODEL_ERROR = None
model = None
try:
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(f"Model file not found at: {MODEL_PATH}")
    #model = tf.keras.models.load_model(MODEL_PATH)
    model = keras.saving.load_model(MODEL_PATH,safe_mode=False, compile=False)
except Exception as e:
    MODEL_ERROR = f"Failed to load model: {e}"
    print("[WARN]", MODEL_ERROR)

# -------------------- Helpers --------------------
def allowed_file(filename: str) -> bool:
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS

def secure_open_image(raw_bytes: bytes) -> Image.Image:
    """Strictly validate & load an image. Raise ValueError('provide proper format') if invalid."""
    bio = io.BytesIO(raw_bytes)
    try:
        im = Image.open(bio)
        im.verify()
    except Exception:
        raise ValueError("provide proper format")
    bio2 = io.BytesIO(raw_bytes)
    im2 = Image.open(bio2)
    if im2.format not in {"PNG", "JPEG", "JPG", "BMP", "WEBP", "GIF"}:
        raise ValueError("provide proper format")
    im2.load()
    return im2

def exif_safe(img: Image.Image) -> Image.Image:
    return ImageOps.exif_transpose(img)

def to_gray01(img: Image.Image) -> np.ndarray:
    return np.asarray(img.convert("L"), dtype=np.float32) / 255.0

def channel_diff_mean(img: Image.Image) -> float:
    """Average absolute RGB differences (0..1). Lower == more grayscale-ish."""
    if img.mode not in ("RGB", "RGBA"):
        return 0.0
    arr = np.asarray(img.convert("RGB"), dtype=np.float32) / 255.0
    r, g, b = arr[..., 0], arr[..., 1], arr[..., 2]
    return float((np.mean(np.abs(r - g)) + np.mean(np.abs(r - b)) + np.mean(np.abs(g - b))) / 3.0)

def laplacian_variance(gray: np.ndarray) -> float:
    """Edge energy variance with 4-neighbor Laplacian."""
    pad = np.pad(gray, 1, mode="edge")
    center = pad[1:-1, 1:-1]
    edge = (pad[:-2, 1:-1] + pad[2:, 1:-1] + pad[1:-1, :-2] + pad[1:-1, 2:]) - 4.0 * center
    return float(np.var(edge))

def sobel_gradients(gray: np.ndarray):
    """Return gradient magnitude and orientation (degrees in [0,180))."""
    kx = np.array([[-1,0,1],[-2,0,2],[-1,0,1]], dtype=np.float32)
    ky = np.array([[-1,-2,-1],[0,0,0],[1,2,1]], dtype=np.float32)
    pad = np.pad(gray, 1, mode="edge")
    gx = (kx[0,0]*pad[:-2,:-2] + kx[0,1]*pad[:-2,1:-1] + kx[0,2]*pad[:-2,2:] +
          kx[1,0]*pad[1:-1,:-2] + kx[1,1]*pad[1:-1,1:-1] + kx[1,2]*pad[1:-1,2:] +
          kx[2,0]*pad[2:,:-2]   + kx[2,1]*pad[2:,1:-1]   + kx[2,2]*pad[2:,2:])
    gy = (ky[0,0]*pad[:-2,:-2] + ky[0,1]*pad[:-2,1:-1] + ky[0,2]*pad[:-2,2:] +
          ky[1,0]*pad[1:-1,:-2] + ky[1,1]*pad[1:-1,1:-1] + ky[1,2]*pad[1:-1,2:] +
          ky[2,0]*pad[2:,:-2]   + ky[2,1]*pad[2:,1:-1]   + ky[2,2]*pad[2:,2:])
    mag = np.hypot(gx, gy)
    ang = (np.degrees(np.arctan2(gy, gx)) + 180.0) % 180.0
    return mag, ang

def orientation_axis_dominance(mag: np.ndarray, ang: np.ndarray, width: float = 10.0) -> float:
    """Fraction of edge energy aligned near 0°/90° (documents show dominance)."""
    total = np.sum(mag) + 1e-9
    w = float(width)
    m0  = (ang <= w) | (ang >= 180.0 - w)
    m90 = (np.abs(ang - 90.0) <= w)
    return float((np.sum(mag[m0]) + np.sum(mag[m90])) / total)

def border_white_fraction(gray: np.ndarray, border_ratio: float = 0.05) -> float:
    h, w = gray.shape
    br = max(1, int(border_ratio * min(h, w)))
    top = gray[:br, :]
    bottom = gray[h-br:, :]
    left = gray[:, :br]
    right = gray[:, w-br:]
    border = np.concatenate([top.flatten(), bottom.flatten(), left.flatten(), right.flatten()])
    return float(np.mean(border >= 0.92))

# ---------- Otsu threshold + Connected Components (document text detection) ----------
def otsu_threshold(gray: np.ndarray) -> float:
    hist, _ = np.histogram(gray, bins=256, range=(0.0, 1.0))
    hist = hist.astype(np.float64)
    prob = hist / (np.sum(hist) + 1e-12)
    omega = np.cumsum(prob)
    mu = np.cumsum(prob * np.arange(256))
    mu_t = mu[-1]
    sigma_b2 = (mu_t * omega - mu)**2 / (omega * (1.0 - omega) + 1e-12)
    idx = int(np.nanargmax(sigma_b2))
    return idx / 255.0

def downscale_gray(gray: np.ndarray, max_dim: int = ANALYZE_MAX_DIM) -> np.ndarray:
    h, w = gray.shape
    scale = min(1.0, max_dim / max(h, w))
    if scale == 1.0:
        return gray
    # PIL resize for decent quality
    img = Image.fromarray((gray * 255.0).astype(np.uint8))
    if h >= w:
        new_size = (int(w * scale), int(h * scale))
    else:
        new_size = (int(w * scale), int(h * scale))
    img = img.resize(new_size, Image.BILINEAR)
    return np.asarray(img, dtype=np.uint8).astype(np.float32) / 255.0

def count_textlike_components(gray: np.ndarray) -> Tuple[int, float]:
    """
    Count small dark connected components (text/handwriting blobs) on a downscaled image.
    Returns: (count, density_per_megapixel)
    """
    g = downscale_gray(gray, ANALYZE_MAX_DIM)
    thr = otsu_threshold(g)
    # Text is typically darker than page; take foreground as <= thr (plus margin)
    fg = g <= min(thr, 0.85)
    # Connected components via iterative stack (8-connectivity)
    h, w = fg.shape
    visited = np.zeros_like(fg, dtype=np.uint8)
    count = 0
    area_min, area_max = CC_AREA_MIN, CC_AREA_MAX
    # neighbors
    nbr = [(-1,-1),(-1,0),(-1,1),(0,-1),(0,1),(1,-1),(1,0),(1,1)]
    for y in range(h):
        row = fg[y]
        for x in range(w):
            if not row[x] or visited[y, x]:
                continue
            # flood fill
            stack = [(y, x)]
            visited[y, x] = 1
            area = 0
            while stack:
                cy, cx = stack.pop()
                area += 1
                for dy, dx in nbr:
                    ny, nx = cy + dy, cx + dx
                    if 0 <= ny < h and 0 <= nx < w and fg[ny, nx] and not visited[ny, nx]:
                        visited[ny, nx] = 1
                        stack.append((ny, nx))
            if area_min <= area <= area_max:
                count += 1
    # density normalized per megapixel of the ANALYZE image
    mp = (h * w) / 1_000_000.0
    density = count / (mp + 1e-9)
    return count, float(density)

def image_stats(img: Image.Image) -> dict:
    img = exif_safe(img)
    gray = to_gray01(img)
    mean = float(np.mean(gray))
    std  = float(np.std(gray))
    lap  = laplacian_variance(gray)
    wf   = float(np.mean(gray >= 0.92))
    bf   = float(np.mean(gray <= 0.08))
    mag, ang = sobel_gradients(gray)
    axis_dom = orientation_axis_dominance(mag, ang, width=10.0)
    bwhite = border_white_fraction(gray, border_ratio=0.05)
    chdiff = channel_diff_mean(img)
    cc_count, cc_density = count_textlike_components(gray)
    h, w = gray.shape
    return {
        "w": w, "h": h, "mean": mean, "std": std, "lap_var": lap,
        "white_frac": wf, "black_frac": bf, "axis_dom": axis_dom,
        "border_white": bwhite, "chdiff": chdiff,
        "cc_small_count": cc_count, "cc_density": cc_density
    }

def is_extreme_blur(stats: dict) -> bool:
    return (stats["lap_var"] < EXTREME_BLUR_VAR) and (stats["std"] < EXTREME_LOW_STD)

def looks_document_like(stats: dict) -> bool:
    """
    Strong doc/prescription heuristic:
      - many small dark components per MP (handwriting/text), OR
      - strong axis dominance with white/flat background.
    """
    texty = stats["cc_density"] >= CC_DENSITY_MIN
    axis_white_flat = (stats["axis_dom"] >= AXIS_DOM_THRESH) and \
                      ((stats["white_frac"] >= DOC_WHITE_FRAC_MIN) or
                       (stats["border_white"] >= BORDER_WHITE_MIN) or
                       (stats["std"] <= DOC_STD_MAX))
    return bool(texty or axis_white_flat)

def looks_cxr_like(stats: dict) -> bool:
    """
    Permissive CXR gate:
      - grayscale-ish (allow mild color via GRAYSCALE_TOL),
      - reasonable mean & contrast,
      - not mostly white/black,
      - not document-like,
      - low axis dominance (edges not only horizontal/vertical).
    """
    if stats["w"] < MIN_WIDTH or stats["h"] < MIN_HEIGHT:
        return False
    if looks_document_like(stats):
        return False
    if stats["chdiff"] > GRAYSCALE_TOL:
        return False
    if not (CXR_MEAN_LO <= stats["mean"] <= CXR_MEAN_HI):
        return False
    if stats["std"] < CXR_MIN_STD:
        return False
    if stats["white_frac"] > CXR_MAX_WHITE_FRAC or stats["black_frac"] > CXR_MAX_BLACK_FRAC:
        return False
    if stats["axis_dom"] >= AXIS_DOM_THRESH:
        return False
    return True

def preprocess_image(image: Image.Image) -> np.ndarray:
    image = exif_safe(image).convert("RGB").resize(TARGET_SIZE)
    arr = np.asarray(image).astype("float32") / 255.0
    return np.expand_dims(arr, axis=0)

def get_pneumonia_index_and_labels(num_classes: int) -> Tuple[Optional[int], List[str]]:
    if num_classes == 1:
        return None, CLASS_LABELS if CLASS_LABELS else ["Pneumonia(1)"]
    if CLASS_LABELS and len(CLASS_LABELS) == num_classes:
        pneu_idx = None
        for i, name in enumerate(CLASS_LABELS):
            if name.lower() in ("pneumonia", "pneumonitis"):
                pneu_idx = i
                break
        if pneu_idx is None and num_classes == 2:
            pneu_idx = 1
        return pneu_idx, CLASS_LABELS
    default_labels = [f"class_{i}" for i in range(num_classes)]
    if num_classes == 2:
        default_labels = ["Normal", "Pneumonia"]
    return 1 if num_classes == 2 else None, default_labels

def three_way_decision(p_pneumonia: float) -> str:
    if p_pneumonia >= INCONCLUSIVE_HIGH:
        return "PNEUMONIA: Positive"
    if p_pneumonia <= INCONCLUSIVE_LOW:
        return "PNEUMONIA: Negative"
    return "cant conclude"

def interpret_predictions(preds: np.ndarray) -> dict:
    out = {
        "raw_predictions": preds.tolist(),
        "inconclusive_band": [INCONCLUSIVE_LOW, INCONCLUSIVE_HIGH],
    }
    if preds.ndim == 2 and preds.shape[1] == 1:
        p = float(preds[0, 0])
        out["probability_pneumonia"] = p
        out["top_index"] = int(p >= 0.5)
        out["top_label"] = "Pneumonia" if p >= 0.5 else "Normal"
        out["decision"] = three_way_decision(p)
        return out
    if preds.ndim == 2:
        num_classes = preds.shape[1]
        pneu_idx, labels = get_pneumonia_index_and_labels(num_classes)
        top_idx = int(np.argmax(preds, axis=1)[0])
        top_prob = float(preds[0, top_idx])
        out["top_index"] = top_idx
        out["top_label"] = labels[top_idx] if 0 <= top_idx < len(labels) else f"class_{top_idx}"
        p = float(preds[0, pneu_idx]) if (pneu_idx is not None and 0 <= pneu_idx < num_classes) else top_prob
        out["probability_pneumonia"] = p
        out["decision"] = three_way_decision(p)
        return out
    flat = preds.reshape(-1)
    top_idx = int(np.argmax(flat))
    p = float(flat[top_idx])
    out["probability_pneumonia"] = p
    out["top_index"] = top_idx
    out["top_label"] = f"class_{top_idx}"
    out["decision"] = three_way_decision(p)
    return out

# -------------------- HTML builder --------------------
def build_index_html() -> str:
    html = """
<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8" />
  <title>Pneumonia Detector • Inference UI</title>
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <style>
    :root { --bg:#0b1020; --card:#121831; --text:#e7ecff; --muted:#9aa4c7; --accent:#5b8cff; --green:#27c18b; --red:#ff5876; --amber:#ffb64d; --border:#1c2342; }
    * { box-sizing: border-box; }
    body { margin:0; padding:2rem; background: radial-gradient(1100px 500px at 20% -10%, #192045 0%, transparent 60%), var(--bg); color:var(--text); font-family:system-ui,-apple-system,"Segoe UI",Roboto,Arial; }
    .wrap { max-width:1100px; margin:0 auto; display:grid; gap:1rem; grid-template-columns:1.2fr 1fr; }
    .card { background: linear-gradient(180deg, rgba(255,255,255,0.03), rgba(255,255,255,0.01)); border:1px solid var(--border); border-radius:16px; padding:1.2rem; backdrop-filter:blur(8px); }
    .uploader { border:2px dashed #2a3568; padding:1rem; border-radius:12px; text-align:center; }
    .btn { display:inline-block; background:var(--accent); color:white; padding:.65rem 1rem; border-radius:10px; border:0; cursor:pointer; font-weight:600; }
    .row { display:flex; align-items:center; gap:.8rem; flex-wrap:wrap; }
    .grid2 { display:grid; grid-template-columns: 1fr 1fr; gap:1rem; }
    .badge { padding:.5rem .8rem; border-radius:999px; font-weight:700; border:1px solid var(--border); }
    .pill { padding:.35rem .6rem; border-radius:999px; border:1px solid var(--border); background:#0f1630; font-weight:600; }
    .muted { color:var(--muted); }
    .preview { width:100%; border-radius:12px; border:1px solid var(--border); display:none; }
    .mono { font-family:ui-monospace, SFMono-Regular, Menlo, Consolas, "Liberation Mono", monospace; }
  </style>
  https://cdn.jsdelivr.net/npm/chart.js@4.4.1</script>
</head>
<body>
  <div class="wrap">
    <div class="card">
      <h1>Pneumonia Detector</h1>
      <div class="muted">Upload a chest X‑ray to classify as <b>Positive</b>, <b>Negative</b>, or <b>cant conclude</b>.</div>
      <div class="uploader" style="margin-top:1rem;">
        <label for="fileInput" class="btn">Choose Image</label>
        <input id="fileInput" type="file" accept="image/*" />
        <button id="btnPredict" class="btn" style="margin-left:.5rem;">Run Prediction</button>
        <div class="muted" style="margin-top:.5rem;">Allowed: __ALLOWED__ • Max __MAX__ MB</div>
      </div>
      <img id="preview" class="preview" alt="preview"/>
    </div>

    <div class="card" id="resultsCard">
      <h1>Result</h1>
      <div id="resultLine" class="row" style="margin:.3rem 0 1rem;">
        <span class="badge">Awaiting image…</span>
      </div>

      <div class="grid2">
        <div class="card" style="border:1px dashed var(--border);">
          <div class="muted">Pneumonia Probability</div>
          <div style="display:flex; align-items:center; gap:1rem; justify-content:center; margin-top:.6rem;">
            <canvas id="probChart" width="320" height="320"></canvas>
          </div>
        </div>
        <div class="card" style="border:1px dashed var(--border);">
          <div class="muted">Details</div>
          <div style="margin-top:.6rem; display:grid; gap:.45rem;">
            <div class="row"><span class="pill">Latency</span><span id="latency" class="mono muted">—</span></div>
            <div class="row"><span class="pill">Band</span><span id="band" class="mono muted">__LOW__ – __HIGH__</span></div>
            <div class="row"><span class="pill">Top Label</span><span id="topLabel" class="mono muted">—</span></div>
            <div class="row"><span class="pill">Model</span><span class="mono muted">__MODEL__</span></div>
          </div>
        </div>
      </div>

      <div class="card" style="margin-top:1rem;">
        <div class="muted">Raw Response (debug)</div>
        <pre id="raw" class="mono" style="white-space:pre-wrap; margin-top:.5rem; overflow:auto; max-height:220px;">—</pre>
      </div>
    </div>
  </div>

  <script>
    const fileInput = document.getElementById('fileInput');
    const btnPredict = document.getElementById('btnPredict');
    const preview = document.getElementById('preview');
    const resultLine = document.getElementById('resultLine');
    const latency = document.getElementById('latency');
    const topLabel = document.getElementById('topLabel');
    const raw = document.getElementById('raw');
    let chart;

    function drawChart(prob) {
      if (!window.Chart) { return; }
      if (!chart) {
        const ctx = document.getElementById('probChart').getContext('2d');
        chart = new Chart(ctx, {
          type: 'doughnut',
          data: { labels: ['Pneumonia', 'Normal'], datasets: [{ data: [prob, 1 - prob], backgroundColor: ['#ff5876', '#27c18b'], borderWidth: 0 }] },
          options: { cutout: '70%', plugins: { legend: { display: false }, tooltip: { callbacks: { label: (ctx) => `${ctx.label}: ${(ctx.raw*100).toFixed(1)}%` } } } }
        });
      } else {
        chart.data.datasets[0].data = [prob, 1 - prob];
        chart.update();
      }
    }

    function setBadge(decision, prob, low, high) {
      const p = (prob*100).toFixed(1);
      let html = '';
      if (decision === 'PNEUMONIA: Positive') {
        html = `<span class="badge" style="color:#ff5876;border-color:#4d2030;background:rgba(255,88,118,.10)">PNEUMONIA: Positive</span>
                <span class="pill">Confidence</span><span class="mono">${p}%</span>`;
      } else if (decision === 'PNEUMONIA: Negative') {
        const conf = (100 - prob*100).toFixed(1);
        html = `<span class="badge" style="color:#27c18b;border-color:#1b3f35;background:rgba(39,193,139,.10)">PNEUMONIA: Negative</span>
                <span class="pill">Confidence</span><span class="mono">${conf}%</span>`;
      } else {
        html = `<span class="badge" style="color:#ffb64d;border-color:#5b4724;background:rgba(255,182,77,.12)">${decision}</span>
                <span class="pill">P(pneumonia)</span><span class="mono">${p}%</span>
                <span class="pill">Band</span><span class="mono">${(low*100).toFixed(0)}–${(high*100).toFixed(0)}%</span>`;
      }
      resultLine.innerHTML = html;
    }

    fileInput.addEventListener('change', () => {
      const file = fileInput.files[0];
      if (file) {
        const url = URL.createObjectURL(file);
        preview.src = url;
        preview.style.display = 'block';
        resultLine.innerHTML = '<span class="badge">Ready to predict…</span>';
      } else {
        preview.style.display = 'none';
        resultLine.innerHTML = '<span class="badge">Awaiting image…</span>';
      }
    });

    btnPredict.addEventListener('click', async () => {
      const file = fileInput.files[0];
      if (!file) {
        resultLine.innerHTML = '<span class="badge" style="color:#ffb64d;border-color:#5b4724;background:rgba(255,182,77,.12)">Choose an image first.</span>';
        return;
      }
      const form = new FormData();
      form.append('file', file);
      resultLine.innerHTML = '<span class="badge">Running inference…</span>';
      raw.textContent = '—';
      try {
        const res = await fetch('/predict', { method: 'POST', body: form });
        const json = await res.json();
        if (!res.ok) {
          const msg = json?.error || `HTTP ${res.status}`;
          resultLine.innerHTML = `<span class="badge" style="color:#ff5876;border-color:#4d2030;background:rgba(255,88,118,.10)">${msg}</span>`;
          raw.textContent = JSON.stringify(json, null, 2);
          return;
        }
        raw.textContent = JSON.stringify(json, null, 2);

        const prob = Math.max(0, Math.min(1, json.probability_pneumonia ?? 0));
        const decision = json.decision ?? "cant conclude";
        latency.textContent = (json.latency_ms != null) ? (json.latency_ms + ' ms') : '—';
        topLabel.textContent = json.top_label ?? '—';
        const band = json.inconclusive_band ?? [0.45, 0.55];
        const low = band[0], high = band[1];

        drawChart(prob);
        setBadge(decision, prob, low, high);
      } catch (err) {
        const msg = (err && err.message) ? err.message : 'Request failed';
        resultLine.innerHTML = `<span class="badge" style="color:#ff5876;border-color:#4d2030;background:rgba(255,88,118,.10)">${msg}</span>`;
        raw.textContent = String(err);
      }
    });
  </script>
</body>
</html>
"""
    html = (html
            .replace("__ALLOWED__", ", ".join(sorted(ALLOWED_EXTENSIONS)))
            .replace("__MAX__", str(MAX_UPLOAD_MB))
            .replace("__LOW__", f"{INCONCLUSIVE_LOW:.2f}")
            .replace("__HIGH__", f"{INCONCLUSIVE_HIGH:.2f}")
            .replace("__MODEL__", MODEL_PATH))
    return html

# -------------------- Routes --------------------
@app.route("/hello", methods=["GET"])
def hello():
    return make_response("<html><body><h1>Hello from Flask</h1></body></html>", 200)

@app.route("/__dump", methods=["GET"])
def dump():
    html = build_index_html()
    return Response(html, mimetype="text/plain; charset=utf-8")

@app.route("/", methods=["GET"])
def index():
    html = build_index_html()
    print(f"[DIAG] / HTML length: {len(html)}")
    return make_response(html, 200)

@app.route("/healthz", methods=["GET"])
def health():
    return jsonify({"status": "ok"}), 200

@app.route("/predict", methods=["POST"])
def predict():
    if MODEL_ERROR:
        return jsonify({"error": MODEL_ERROR}), 500
    try:
        # Input validation
        if "file" not in request.files:
            return jsonify({"error": "provide proper format"}), 400
        file = request.files["file"]
        if file.filename == "":
            return jsonify({"error": "provide proper format"}), 400
        if not (file.mimetype or "").startswith("image/"):
            return jsonify({"error": "provide proper format"}), 400
        if not allowed_file(file.filename):
            return jsonify({"error": "provide proper format"}), 400

        raw_bytes = file.read()
        if not raw_bytes:
            return jsonify({"error": "provide proper format"}), 400

        # Decode & analyze
        try:
            image = secure_open_image(raw_bytes)
        except ValueError as ve:
            return jsonify({"error": str(ve)}), 400

        stats = image_stats(image)

        # 1) Non-CXR (documents/prescriptions/photos) -> do not conclude
        if not looks_cxr_like(stats):
            payload = {"decision": "wrong kind", "message": "wrong kind"}
            if DEBUG_METRICS:
                payload["metrics"] = stats
            return jsonify(payload), 200

        # 2) Extreme blur -> do not conclude
        if is_extreme_blur(stats):
            payload = {"decision": "cant conclude", "message": "cant conclude", "reason": "extreme_blur"}
            if DEBUG_METRICS:
                payload["metrics"] = stats
            return jsonify(payload), 200

        # 3) Valid CXR -> run model
        x = preprocess_image(image)
        t0 = time.time()
        preds = model.predict(x)
        latency_ms = int((time.time() - t0) * 1000)

        result = interpret_predictions(preds)
        result["latency_ms"] = latency_ms
        if DEBUG_METRICS:
            result["metrics"] = stats
        return jsonify(result), 200

    except Exception as e:
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    port = int(os.getenv("PORT", 5050))
    app.run(host="0.0.0.0", port=port, debug=False)
