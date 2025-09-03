# app/app.py
import os, pathlib
if os.name == "nt":  # only on Windows
    # Map any PosixPath encountered during unpickling to WindowsPath
    # happens due to training on kaggle
    pathlib.PosixPath = pathlib.WindowsPath  # type: ignore[attr-defined]
    
from pathlib import Path
import tempfile
import time

import streamlit as st
import pandas as pd
from PIL import Image

# ---- CONFIG ----
REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_WEIGHTS = REPO_ROOT / "weights" / "best.pt"
ASSETS = REPO_ROOT / "app" / "assets"

# Import YOLOv5's detect.run
# Make sure yolov5/ exists (submodule) and its requirements are installed.
import sys
sys.path.append(str(REPO_ROOT / "yolov5"))
from detect import run as yolo_detect  # YOLOv5 detect.py

st.set_page_config(page_title="Recycle YOLO Demo", layout="wide")

# ---- SIDEBAR ----
st.sidebar.title("⚙️ Settings")
weights_path = st.sidebar.text_input("Weights (.pt)", str(DEFAULT_WEIGHTS))
imgsz_choice = st.sidebar.selectbox("Image size", [640, 512, 416], index=0)
imgsz = (int(imgsz_choice), int(imgsz_choice))  # making it iterable (h, w)
conf_thres = st.sidebar.slider("Confidence threshold", 0.05, 0.80, 0.25, 0.01)
iou_thres = st.sidebar.slider("NMS IoU", 0.10, 0.90, 0.45, 0.01)

st.title("♻️ Recyclables Detection — YOLOv5")
st.write("Upload an image; the model will draw boxes and list predictions with confidence.")

# ---- LEFT: UPLOAD & RUN ----
col1, col2 = st.columns([2, 1], gap="large")

with col1:
    uploaded = st.file_uploader("Upload an image", type=["jpg","jpeg","png"])
    if uploaded:
        # Save to a temp file
        tmpdir = Path(tempfile.mkdtemp())
        src_path = tmpdir / uploaded.name
        src_path.write_bytes(uploaded.read())

        # Run YOLO detect.py
        out_root = REPO_ROOT / "app" / "tmp"
        out_root.mkdir(parents=True, exist_ok=True)
        run_name = f"infer_{int(time.time())}"

        # detect.py will save annotated image under out_root/run_name
        yolo_detect(
            weights=str(weights_path),
            source=str(src_path),
            imgsz=imgsz,
            conf_thres=conf_thres,
            iou_thres=iou_thres,
            save_txt=True,
            save_conf=True,
            project=str(out_root),
            name=run_name,
            exist_ok=True,
            device="cpu"  # set "0" if you run on GPU
        )

        # Find the saved result image
        exp_dir = out_root / run_name
        # detect.py writes an image with the same filename in exp_dir
        result_img_path = exp_dir / uploaded.name
        # Sometimes detect.py saves under 'exp' then subfolders; fallback:
        if not result_img_path.exists():
            # Grab first image in exp_dir
            imgs = list(exp_dir.rglob("*.jpg")) + list(exp_dir.rglob("*.png"))
            result_img_path = imgs[0] if imgs else None

        if result_img_path and result_img_path.exists():
            st.image(Image.open(result_img_path), caption="Detections", use_container_width=True)
        else:
            st.warning("No detections image found (model may have predicted none).")

        # Build a small table from labels txt if present
        txts = list(exp_dir.rglob("*.txt"))
        rows = []
        names_path = REPO_ROOT / "data" / "processed" / "names.txt"
        names = [n.strip() for n in names_path.read_text().splitlines()] if names_path.exists() else None

        for t in txts:
            for line in t.read_text().strip().splitlines():
                parts = line.split()
                if len(parts) >= 6:
                    cls = int(parts[0])
                    conf = float(parts[-1]) if parts[-1].replace('.','',1).isdigit() else None
                    label = names[cls] if names and 0 <= cls < len(names) else str(cls)
                    rows.append({"label": label, "conf": conf})
        if rows:
            df = pd.DataFrame(rows).sort_values("conf", ascending=False)
            st.dataframe(df, use_container_width=True)
        else:
            st.info("No boxes above threshold.")

# ---- RIGHT: EVAL VISUALS ----
with col2:
    st.subheader("Evaluation Visuals")
    for fname, caption in [
        ("PR_curve.png", "Precision–Recall (per class, mAP@0.5)"),
        ("confusion_matrix.png", "Confusion Matrix (Predicted vs True + background)"),
        ("results.png", "Training Curves (loss, precision, recall, mAP)"),
    ]:
        p = ASSETS / fname
        if p.exists():
            st.image(str(p), caption=caption, use_container_width=True)
        else:
            st.caption(f"Missing: {fname} (place in app/assets/)")

st.markdown("---")
st.caption("Tip: to run with GPU on a server/Space, change detect device='0'.")
