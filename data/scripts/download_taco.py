# data/scripts/download_taco.py
"""
Use TACO's official downloader, then auto-discover where it saved images,
and copy them into <REPO_ROOT>/data/raw/taco/images plus annotations.json.

Examples:
  python data/scripts/download_taco.py
  python data/scripts/download_taco.py --use_unofficial
  python data/scripts/download_taco.py --branch master
"""

import argparse
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

TACO_REPO_URL = "https://github.com/pedropro/TACO.git"
IMG_EXTS = {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".gif", ".tiff"}


def repo_root_from_this_file() -> Path:
    here = Path(__file__).resolve()
    return here.parents[2]  # scripts -> data -> <repo>


def run(cmd, cwd=None):
    print(f"▶ {' '.join(cmd)}  (cwd={cwd or Path.cwd()})")
    p = subprocess.run(
        cmd,
        cwd=str(cwd) if cwd else None,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="ignore",
    )
    print("----- stdout (download.py) -----")
    print(p.stdout.strip())
    print("----- stderr (download.py) -----")
    print(p.stderr.strip())
    if p.returncode != 0:
        raise SystemExit(f"download.py failed with exit code {p.returncode}")


def count_images(d: Path) -> int:
    return sum(1 for f in d.rglob("*") if f.is_file() and f.suffix.lower() in IMG_EXTS)


def find_best_images_dir(root: Path, min_images: int = 10) -> Path | None:
    """
    Search under <root> (typically <TACO>/data) for a directory with many images.
    Pick the one with the most images.
    """
    best_dir, best_count = None, 0
    for p in root.rglob("*"):
        if p.is_dir():
            n = count_images(p)
            if n > best_count:
                best_count, best_dir = n, p
    if best_dir and best_count >= min_images:
        print(f"🔎 Found images dir candidate: {best_dir} (files: {best_count})")
        return best_dir
    return None


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--branch", default="master", help="TACO branch or tag")
    p.add_argument("--use_unofficial", action="store_true",
                   help="Use data/annotations_unofficial.json")
    p.add_argument("--out_root", default="data/raw/taco",
                   help="Destination (relative to repo root if relative)")
    p.add_argument("--python", default=sys.executable,
                   help="Python interpreter for download.py")
    args = p.parse_args()

    repo_root = repo_root_from_this_file()
    out_root = Path(args.out_root)
    if not out_root.is_absolute():
        out_root = (repo_root / out_root).resolve()
    images_out = out_root / "images"
    ann_out = out_root / "annotations.json"
    images_out.mkdir(parents=True, exist_ok=True)
    out_root.mkdir(parents=True, exist_ok=True)

    with tempfile.TemporaryDirectory() as tmpd:
        tmp = Path(tmpd)
        taco_dir = tmp / "TACO"

        # 1) Clone TACO
        print(f"⬇️  Cloning {TACO_REPO_URL}@{args.branch} into {taco_dir}")
        subprocess.check_call(
            ["git", "clone", "--depth", "1", "--branch", args.branch, TACO_REPO_URL, str(taco_dir)]
        )

        # 2) Run official downloader
        download_py = taco_dir / "download.py"
        cmd = [args.python, str(download_py)]
        if args.use_unofficial:
            cmd += ["--dataset_path", str(taco_dir / "data" / "annotations_unofficial.json")]
        run(cmd, cwd=str(taco_dir))

        # 3) Locate annotations.json
        ann_src = taco_dir / "data" / "annotations.json"
        if not ann_src.exists():
            # Try unofficial if requested
            if args.use_unofficial:
                ann_src = taco_dir / "data" / "annotations_unofficial.json"
        if not ann_src.exists():
            # As a fallback, search for any annotations.json
            candidates = list(taco_dir.rglob("annotations.json"))
            if candidates:
                ann_src = candidates[0]
        if not ann_src.exists():
            raise SystemExit("Could not find annotations.json in TACO repo after download.")

        ann_out.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(ann_src, ann_out)

        # 4) Locate images directory (be flexible)
        data_root = taco_dir / "data"
        images_src = data_root / "images"
        if not images_src.exists() or count_images(images_src) == 0:
            images_src = find_best_images_dir(data_root)
            if images_src is None:
                # print a small tree for debugging
                print("❌ Could not auto-locate images. Here's a quick listing of <TACO>/data:")
                for pth in sorted(data_root.glob("*")):
                    kind = "DIR " if pth.is_dir() else "FILE"
                    print(f"  {kind:4} {pth.name}")
                raise SystemExit(
                    "Could not find an images directory with files. "
                    "Check the download.py logs above (rate-limited? network blocked?)."
                )

        # 5) Copy images (merge-safe)
        print(f"📦 Copying images from {images_src} → {images_out}")
        for pth in images_src.rglob("*"):
            if pth.is_file() and pth.suffix.lower() in IMG_EXTS:
                rel = pth.relative_to(images_src)
                dest = images_out / rel
                dest.parent.mkdir(parents=True, exist_ok=True)
                if not dest.exists():
                    shutil.copy2(pth, dest)

    print("------------------------------------------------------")
    print(f"✅ Repo root:        {repo_root}")
    print(f"✅ Output root:      {out_root}")
    print(f"✅ Images directory: {images_out}")
    print(f"✅ Annotations:      {ann_out}")
    print("Done.")


if __name__ == "__main__":
    main()
