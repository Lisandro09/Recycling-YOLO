#!/usr/bin/env python
import argparse, json, shutil
from pathlib import Path

def coco_to_yolo_bbox(bbox, img_w, img_h):
    x,y,w,h = bbox
    cx = x + w/2; cy = y + h/2
    return cx/img_w, cy/img_h, w/img_w, h/img_h

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--coco", default="data/raw/taco/annotations.json")
    ap.add_argument("--images", default="data/raw/taco/images")
    ap.add_argument("--class_map", default="data/scripts/class_map.json")
    ap.add_argument("--out_images", default="data/processed/images/all")
    ap.add_argument("--out_labels", default="data/processed/labels/all")
    ap.add_argument("--names_out", default="data/processed/names.txt")
    args = ap.parse_args()

    coco = json.loads(Path(args.coco).read_text())
    id_to_img = {im["id"]: im for im in coco["images"]}
    id_to_cat = {c["id"]: c["name"] for c in coco["categories"]}
    anns_by_img = {}
    for ann in coco["annotations"]:
        anns_by_img.setdefault(ann["image_id"], []).append(ann)

    class_map = json.loads(Path(args.class_map).read_text())

    # Derive target class order from mapping values (stable, deterministic)
    targets = []
    for v in class_map.values():
        if v is not None and v not in targets:
            targets.append(v)
    # Optionally enforce your own order:
    # targets = ["plastic_bottle","aluminum_can","glass_bottle","metal_can"]

    out_images = Path(args.out_images); out_images.mkdir(parents=True, exist_ok=True)
    out_labels = Path(args.out_labels); out_labels.mkdir(parents=True, exist_ok=True)
    src_images = Path(args.images)

    kept_imgs = 0; kept_anns = 0
    for img_id, anns in anns_by_img.items():
        im = id_to_img[img_id]
        img_name = im["file_name"]
        img_w, img_h = im["width"], im["height"]
        label_lines = []

        for ann in anns:
            src_cat = id_to_cat[ann["category_id"]]
            dst_cat = class_map.get(src_cat)
            if dst_cat is None:
                continue
            try:
                cls_idx = targets.index(dst_cat)
            except ValueError:
                targets.append(dst_cat)
                cls_idx = targets.index(dst_cat)

            xcycwh = coco_to_yolo_bbox(ann["bbox"], img_w, img_h)
            label_lines.append(f"{cls_idx} " + " ".join(f"{v:.6f}" for v in xcycwh))
            kept_anns += 1

        if label_lines:
            dst_img = out_images / img_name
            dst_lbl = out_labels / (Path(img_name).with_suffix(".txt").name)
            dst_img.parent.mkdir(parents=True, exist_ok=True)
            dst_lbl.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(src_images / img_name, dst_img)
            Path(dst_lbl).write_text("\n".join(label_lines))
            kept_imgs += 1

    Path(args.names_out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.names_out).write_text("\n".join(targets))

    print(f"✅ Converted images: {kept_imgs}")
    print(f"✅ Converted annotations: {kept_anns}")
    print(f"✅ Class order saved to: {args.names_out}")

if __name__ == "__main__":
    main()
