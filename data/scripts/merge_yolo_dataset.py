#!/usr/bin/env python
import argparse, shutil
from pathlib import Path

def load_names(path: Path):
    if path.suffix == ".txt":
        return [l.strip() for l in path.read_text().splitlines() if l.strip()]
    elif path.suffix in {".yaml",".yml"}:
        import yaml
        return yaml.safe_load(path.read_text())["names"]
    else:
        raise ValueError("names must be .txt or .yaml")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--donor_images", required=True)
    ap.add_argument("--donor_labels", required=True)
    ap.add_argument("--donor_names",  required=True)  # txt or yaml
    ap.add_argument("--target_names", default="../data/processed/names.txt")
    ap.add_argument("--out_images",   default="../data/processed/images/all")
    ap.add_argument("--out_labels",   default="../data/processed/labels/all")
    ap.add_argument("--prefix",       default="dw_")  # to avoid filename collisions
    args = ap.parse_args()

    donor_images = Path(args.donor_images)
    donor_labels = Path(args.donor_labels)
    donor_names  = load_names(Path(args.donor_names))
    target_names = [l.strip() for l in Path(args.target_names).read_text().splitlines()]

    # ---- EDIT THIS mapping to your donor set ----
    name_map = {
        "PET": "plastic_bottle",
        "Glass":   "glass_bottle",
        "AluCan":   "metal_can",
        "HDPEM": "plastic_bottle",
    }
    # --------------------------------------------

    # donor idx -> target idx or None
    remap = {}
    for i, dn in enumerate(donor_names):
        tgt = name_map.get(dn, None)
        remap[i] = (target_names.index(tgt) if tgt in target_names else None)

    out_i = Path(args.out_images); out_i.mkdir(parents=True, exist_ok=True)
    out_l = Path(args.out_labels); out_l.mkdir(parents=True, exist_ok=True)

    kept = 0
    for img in donor_images.rglob("*"):
        if img.suffix.lower() not in {".jpg",".jpeg",".png"}: continue
        lbl = donor_labels / (img.stem + ".txt")
        if not lbl.exists(): continue

        new_lines = []
        for ln in lbl.read_text().splitlines():
            if not ln.strip(): continue
            parts = ln.split()
            old = int(parts[0])
            new = remap.get(old, None)
            if new is None: 
                continue
            parts[0] = str(new)
            new_lines.append(" ".join(parts))

        if not new_lines:
            continue

        # avoid filename collisions by prefixing
        dst_img = out_i / f"{args.prefix}{img.name}"
        dst_lbl = out_l / f"{args.prefix}{img.stem}.txt"
        shutil.copy2(img, dst_img)
        dst_lbl.write_text("\n".join(new_lines))
        kept += 1

    print(f"✅ Merged {kept} donor images with retained boxes.")

if __name__ == "__main__":
    main()
