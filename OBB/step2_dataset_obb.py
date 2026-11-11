# Step 2 (OBB): unzip the Roboflow export -> convert polygon labels to OBB -> augment the train split.
#
# Input : the same polygon (segmentation) zip used for the seg runs, in MyDrive/topgun/
# Output: /content/dataset_obb with OBB labels  (class x1 y1 x2 y2 x3 y3 x4 y4, normalised)
#         + data.yaml ready for YOLO OBB training.
#
# Each polygon becomes the smallest rotated rectangle that contains it (cv2.minAreaRect).
# A plain OBB rectangle has no "head / tail" direction, and none is needed for counting + classes.
#
# Colab: run this file (%run step2_dataset_obb.py) or paste it into a cell. Self-contained on purpose.
import glob
import os
import random
import shutil
import subprocess

import cv2
import numpy as np
import yaml

ZIP_GLOB = '/content/drive/MyDrive/topgun/TOPGUN*.zip'
SRC_DIR = '/content/dataset'       # unzipped Roboflow export (polygon labels)
OBB_DIR = '/content/dataset_obb'   # converted dataset used for training (rebuilt from scratch every run)
AUG_COPIES = 2                     # extra augmented copies per train image (0 = off). 2 = 3x train images (best in seg runs)
SEED = 0

AUG_TAG = '_augcopy'
GEOMETRY = ['hflip', 'vflip', 'rot180', 'rot90cw', 'rot90ccw']


# ---------------------------------------------------------------- polygon -> OBB
def polygon_to_obb(vals, w, h):
    """vals: normalised numbers after the class id (polygon x y x y ... or box cx cy bw bh).
    Returns 8 normalised numbers = 4 corners of the smallest rotated rectangle."""
    v = np.asarray(vals, dtype=np.float64)
    if len(v) == 4:  # plain box
        cx, cy, bw, bh = v
        pts = np.array([[cx - bw / 2, cy - bh / 2], [cx + bw / 2, cy - bh / 2],
                        [cx + bw / 2, cy + bh / 2], [cx - bw / 2, cy + bh / 2]]) * [w, h]
    else:
        pts = v.reshape(-1, 2) * [w, h]  # fit in pixel space so the rectangle is not distorted
    corners = cv2.boxPoints(cv2.minAreaRect(pts.astype(np.float32))) / [w, h]
    return np.clip(corners, 0.0, 1.0).reshape(-1)


def convert_split(split):
    """Convert one split (train / valid / test). Returns (images, objects, skipped_lines)."""
    img_out, lbl_out = f'{OBB_DIR}/{split}/images', f'{OBB_DIR}/{split}/labels'
    os.makedirs(img_out, exist_ok=True)
    os.makedirs(lbl_out, exist_ok=True)
    n_img = n_obj = n_skip = 0
    for path in sorted(glob.glob(f'{SRC_DIR}/{split}/images/*')):
        stem = os.path.splitext(os.path.basename(path))[0]
        img = cv2.imread(path)
        if img is None:
            continue
        h, w = img.shape[:2]
        lines = []
        src_lbl = f'{SRC_DIR}/{split}/labels/{stem}.txt'
        if os.path.exists(src_lbl):
            for line in open(src_lbl):
                p = line.split()
                if not p:
                    continue
                if len(p) < 5 or (len(p) - 1) % 2:  # need a box (4 numbers) or a polygon (even count)
                    n_skip += 1
                    continue
                corners = polygon_to_obb([float(t) for t in p[1:]], w, h)
                lines.append(p[0] + ' ' + ' '.join(f'{c:.6f}' for c in corners))
        with open(f'{lbl_out}/{stem}.txt', 'w') as f:
            f.write('\n'.join(lines) + ('\n' if lines else ''))
        shutil.copy(path, img_out)
        n_img += 1
        n_obj += len(lines)
    return n_img, n_obj, n_skip


# ---------------------------------------------------------------- offline augmentation (train only)
# Flips, 90-degree rotations and colour changes. The 4 corners are transformed exactly, so the
# labels stay correct. valid / test are never touched.
def _xy(x, y, how):
    return {
        'hflip': (1 - x, y),
        'vflip': (x, 1 - y),
        'rot180': (1 - x, 1 - y),
        'rot90cw': (1 - y, x),
        'rot90ccw': (y, 1 - x),
    }[how]


def _image(img, how):
    return {
        'hflip': lambda i: cv2.flip(i, 1),
        'vflip': lambda i: cv2.flip(i, 0),
        'rot180': lambda i: cv2.rotate(i, cv2.ROTATE_180),
        'rot90cw': lambda i: cv2.rotate(i, cv2.ROTATE_90_CLOCKWISE),
        'rot90ccw': lambda i: cv2.rotate(i, cv2.ROTATE_90_COUNTERCLOCKWISE),
    }[how](img)


def _transform_obb_line(line, how):
    p = line.split()
    v = [float(t) for t in p[1:9]]
    pts = []
    for i in range(0, 8, 2):
        pts += list(_xy(v[i], v[i + 1], how))
    return p[0] + ' ' + ' '.join(f'{c:.6f}' for c in pts)


def _colour(img):
    """Random brightness / contrast / saturation change (labels unaffected)."""
    out = img.astype(np.float32) * random.uniform(0.8, 1.2) + random.uniform(-20, 20)
    out = np.clip(out, 0, 255).astype(np.uint8)
    hsv = cv2.cvtColor(out, cv2.COLOR_BGR2HSV).astype(np.float32)
    hsv[..., 1] = np.clip(hsv[..., 1] * random.uniform(0.7, 1.3), 0, 255)
    return cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2BGR)


def augment_train(copies):
    """Write `copies` extra images + labels per train image. Returns number of copies written."""
    if copies <= 0:
        return 0
    random.seed(SEED)
    img_dir, lbl_dir = f'{OBB_DIR}/train/images', f'{OBB_DIR}/train/labels'
    originals = [p for p in sorted(glob.glob(f'{img_dir}/*')) if AUG_TAG not in os.path.basename(p)]
    made = 0
    for path in originals:
        stem, ext = os.path.splitext(os.path.basename(path))
        lbl_path = f'{lbl_dir}/{stem}.txt'
        if not os.path.exists(lbl_path):
            continue
        img = cv2.imread(path)
        lines = [l.strip() for l in open(lbl_path) if l.strip()]
        for k in range(1, copies + 1):
            new_stem = f'{stem}{AUG_TAG}{k}'
            if os.path.exists(f'{lbl_dir}/{new_stem}.txt'):
                continue
            how = random.choice(GEOMETRY)
            new_lines = [_transform_obb_line(l, how) for l in lines]
            cv2.imwrite(f'{img_dir}/{new_stem}{ext}', _colour(_image(img, how)))
            with open(f'{lbl_dir}/{new_stem}.txt', 'w') as f:
                f.write('\n'.join(new_lines) + ('\n' if new_lines else ''))
            made += 1
    return made


def class_counts(split, names):
    """Objects per class in an OBB split (includes augmented copies for train)."""
    ids = names if isinstance(names, dict) else dict(enumerate(names))
    cnt = {n: 0 for n in ids.values()}
    for f in glob.glob(f'{OBB_DIR}/{split}/labels/*.txt'):
        for line in open(f):
            if line.strip():
                cnt[ids[int(line.split()[0])]] += 1
    return cnt


# ---------------------------------------------------------------- main
def main():
    from google.colab import drive
    drive.mount('/content/drive')

    zips = sorted(glob.glob(ZIP_GLOB))
    if not zips:
        raise FileNotFoundError(f'No zip found with pattern {ZIP_GLOB}')
    if len(zips) > 1:
        print('WARNING: several zips match, using the first one:', zips)
    print('Using zip:', zips[0])

    os.makedirs(SRC_DIR, exist_ok=True)
    subprocess.run(['unzip', '-q', '-o', zips[0], '-d', SRC_DIR], check=True)
    names = yaml.safe_load(open(f'{SRC_DIR}/data.yaml'))['names']

    if os.path.isdir(OBB_DIR):  # generated folder: always rebuild so settings and files stay in sync
        shutil.rmtree(OBB_DIR)
    for split in ['train', 'valid', 'test']:
        n_img, n_obj, n_skip = convert_split(split)
        print(f'{split:6}: {n_img} images, {n_obj} objects converted' + (f', {n_skip} lines skipped!' if n_skip else ''))

    made = augment_train(AUG_COPIES)
    print(f'Augmentation: {made} copies written, train now has {len(glob.glob(f"{OBB_DIR}/train/images/*"))} images')

    yaml.safe_dump({'path': OBB_DIR, 'train': 'train/images', 'val': 'valid/images',
                    'test': 'test/images', 'names': names}, open(f'{OBB_DIR}/data.yaml', 'w'))
    for split in ['train', 'valid', 'test']:
        print(split, class_counts(split, names))
    print('OBB dataset ready:', f'{OBB_DIR}/data.yaml')


if __name__ == '__main__':
    main()
