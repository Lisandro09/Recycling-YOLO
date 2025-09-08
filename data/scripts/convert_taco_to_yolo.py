#!/usr/bin/env python
import argparse, json, shutil
from pathlib import Path
from PIL import Image, ImageOps

def coco_to_yolo_bbox_xywh(x, y, w, h, img_w, img_h):
    x = max(0.0, min(x, img_w))
    y = max(0.0, min(y, img_h))
    w = max(0.0, min(w, img_w - x))
    h = max(0.0, min(h, img_h - y))
    cx = x + w / 2.0
    cy = y + h / 2.0
    return cx / img_w, cy / img_h, w / img_w, h / img_h

def resolve_path(p: str, repo_root: Path) -> Path:
    """If p is absolute or exists as given, return it; else resolve relative to repo_root."""
    pp = Path(p)
    if pp.is_absolute() or pp.exists():
        return pp
    return (repo_root / pp).resolve()

def main():
    # infer repo root: script is at <repo>/data/scripts/convert_taco_to_yolo.py
    script_dir = Path(__file__).resolve().parent
    repo_root = script_dir.parent.parent  # go up from data/scripts -> data -> <repo>

    ap = argparse.ArgumentParser()
    ap.add_argument("--coco", default="data/raw/taco/annotations.json")
    ap.add_argument("--images", default="data/raw/taco/images")
    ap.add_argument("--class_map", default="data/scripts/class_map.json")
    ap.add_argument("--out_images", default="data/processed/images/all")
    ap.add_argument("--out_labels", default="data/processed/labels/all")
    ap.add_argument("--names_out", default="data/processed/names.txt")
    ap.add_argument("--min_box", type=float, default=4.0,
                    help="skip boxes smaller than N pixels after scaling")
    args = ap.parse_args()

    # resolve all paths robustly
    coco_path     = resolve_path(args.coco, repo_root)
    images_root   = resolve_path(args.images, repo_root)
    class_map_p   = resolve_path(args.class_map, repo_root)
    out_images_p  = resolve_path(args.out_images, repo_root)
    out_labels_p  = resolve_path(args.out_labels, repo_root)
    names_out_p   = resolve_path(args.names_out, repo_root)

    print(" Resolved paths:")
    print("  COCO       :", coco_path)
    print("  IMAGES     :", images_root)
    print("  CLASS_MAP  :", class_map_p)
    print("  OUT_IMAGES :", out_images_p)
    print("  OUT_LABELS :", out_labels_p)
    print("  NAMES_OUT  :", names_out_p)

    if not coco_path.exists():
        raise FileNotFoundError(f"COCO annotations not found at: {coco_path}")
    if not images_root.exists():
        raise FileNotFoundError(f"Images root not found at: {images_root}")
    if not class_map_p.exists():
        raise FileNotFoundError(f"class_map.json not found at: {class_map_p}")

    coco = json.loads(coco_path.read_text(encoding="utf-8"))
    id_to_img = {im["id"]: im for im in coco["images"]}
    id_to_cat = {c["id"]: c["name"] for c in coco["categories"]}

    anns_by_img = {}
    for ann in coco["annotations"]:
        anns_by_img.setdefault(ann["image_id"], []).append(ann)

    class_map = json.loads(class_map_p.read_text(encoding="utf-8"))

    targets = []
    for v in class_map.values():
        if v is not None and v not in targets:
            targets.append(v)

    out_images_p.mkdir(parents=True, exist_ok=True)
    out_labels_p.mkdir(parents=True, exist_ok=True)

    kept_imgs = kept_anns = 0
    mismatched_dims = 0
    skipped_tiny = 0
    missing_files = 0

    for img_id, ann_list in anns_by_img.items():
        im = id_to_img[img_id]
        file_name = im["file_name"]  # may include subdirs
        src_path = images_root / file_name
        if not src_path.exists():
            src_path = images_root / Path(file_name).name
            if not src_path.exists():
                missing_files += 1
                continue

        with Image.open(src_path) as pil_im:
            pil_im = ImageOps.exif_transpose(pil_im)
            actual_w, actual_h = pil_im.size

        ann_w = im.get("width", actual_w)
        ann_h = im.get("height", actual_h)
        sx = actual_w / ann_w if ann_w else 1.0
        sy = actual_h / ann_h if ann_h else 1.0
        if abs(sx - 1.0) > 1e-3 or abs(sy - 1.0) > 1e-3:
            mismatched_dims += 1

        label_lines = []
        for ann in ann_list:
            dst_cat = class_map.get(id_to_cat[ann["category_id"]])
            if dst_cat is None:
                continue
            if dst_cat not in targets:
                targets.append(dst_cat)
            cls_idx = targets.index(dst_cat)

            x, y, w, h = ann["bbox"]
            x *= sx; y *= sy; w *= sx; h *= sy

            if w < args.min_box or h < args.min_box:
                skipped_tiny += 1
                continue

            xc, yc, bw, bh = coco_to_yolo_bbox_xywh(x, y, w, h, actual_w, actual_h)
            if not (0.0 < bw <= 1.0 and 0.0 < bh <= 1.0):
                continue

            label_lines.append(f"{cls_idx} {xc:.6f} {yc:.6f} {bw:.6f} {bh:.6f}")
            kept_anns += 1

        if label_lines:
            dst_img = out_images_p / file_name
            dst_lbl = out_labels_p / Path(file_name).with_suffix(".txt")
            dst_img.parent.mkdir(parents=True, exist_ok=True)
            dst_lbl.parent.mkdir(parents=True, exist_ok=True)
            # save EXIF-corrected image alongside label
            with Image.open(src_path) as pil_im:
                pil_im = ImageOps.exif_transpose(pil_im)
                pil_im.save(dst_img)
            dst_lbl.write_text("\n".join(label_lines), encoding="utf-8")
            kept_imgs += 1

    names_out_p.parent.mkdir(parents=True, exist_ok=True)
    names_out_p.write_text("\n".join(targets), encoding="utf-8")

    print(f" Converted images: {kept_imgs}", flush=True)
    print(f" Converted annotations: {kept_anns}", flush=True)
    print(f" Size-mismatch images (auto-rescaled): {mismatched_dims}", flush=True)
    print(f" Skipped tiny boxes (< {args.min_box}px): {skipped_tiny}", flush=True)
    print(f" Missing image files: {missing_files}", flush=True)
    print(f" Class order saved to: {names_out_p}", flush=True)

if __name__ == "__main__":
    main()
