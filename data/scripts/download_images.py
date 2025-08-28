'''
This script downloads TACO's images from Flickr given an annotation json file
Code written by Pedro F. Proenza, 2019
'''

import os.path
import argparse
import json
from PIL import Image
import requests
from io import BytesIO
import sys

# Hard-coded locations
dataset_path = r"C:\Users\lisan\Jupyter Notebook Projects\Recycling-YOLO\data\raw\taco\annotations.json"
images_root  = r"C:\Users\lisan\Jupyter Notebook Projects\Recycling-YOLO\data\raw\taco\images"

dataset_dir = os.path.dirname(dataset_path)

# Load annotations
with open(dataset_path, "r") as f:
    annotations = json.loads(f.read())

nr_images = len(annotations['images'])
for i in range(nr_images):
    image = annotations['images'][i]
    file_name = image['file_name']                   # e.g. "batch_10/000074.jpg"
    url_original = image['flickr_url']
    url_resized  = image.get('flickr_640_url')

    file_path = os.path.join(images_root, file_name) # <-- now under images/
    subdir = os.path.dirname(file_path)
    os.makedirs(subdir, exist_ok=True)               # safer than os.mkdir

    if not os.path.isfile(file_path):
        try:
            response = requests.get(url_original, timeout=15)
            response.raise_for_status()
        except Exception:
            # optional fallback to the resized URL
            if url_resized:
                response = requests.get(url_resized, timeout=15)
                response.raise_for_status()
            else:
                print(f"Skip (no URL): {file_name}")
                continue

        img = Image.open(BytesIO(response.content))
        if hasattr(img, "_getexif") and img._getexif():
            img.save(file_path, exif=img.info.get("exif", b""))
        else:
            img.save(file_path)

    bar_size = 30
    x = int(bar_size * (i + 1) / nr_images)
    sys.stdout.write("%s[%s%s] - %i/%i\r" %
                     ('Loading: ', "=" * x, "." * (bar_size - x), i + 1, nr_images))
    sys.stdout.flush()

sys.stdout.write('\nFinished\n')
