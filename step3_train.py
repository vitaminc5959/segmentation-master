# Train -> graphs -> predict -> view results -> download weights, all in one step.
import glob
import os
import random

import cv2
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from google.colab import files
from IPython.display import Image, display
from ultralytics import YOLO

DATA_DIR = '/content/dataset'
YAML_PATH = f'{DATA_DIR}/data.yaml'
RUNS_DIR = '/content/drive/MyDrive/topgun/runs'  # on Drive so it survives disconnects
RUN_NAME = 'seg9-e300'  # new name per run; Ultralytics adds -2, -3... if the folder exists
MODEL_NAME = 'yolo11s-seg.pt'  # small model (same as seg3-6)
IMGSZ = 1024                   # used for train, val and predict (2048 did not help in seg5-2k)
BATCH = 16                     # A100 40GB at imgsz 1024
AUG_COPIES = 2                 # extra augmented copies per training image (0 = off). 2 copies = 3x train images
EPOCHS = 300                   # run seg9-e300: seg6-aug setup (2 copies) but 250 -> 300 epochs
PRED_DIR = '/content/predictions'

# ---------------- offline augmentation (train split only) ----------------
# Writes extra images + labels into train/. Only flips, 90-degree rotations and colour
# changes, so the polygon labels stay exactly correct. Never touches valid/test.
# Safe to re-run: existing *_aug* files are skipped, and originals are never changed.
GEOMETRY = ['hflip', 'vflip', 'rot180', 'rot90cw', 'rot90ccw']


def _xy(x, y, how):
    """Transform one normalized point."""
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


def _label_line(line, how):
    p = line.split()
    if len(p) < 5:
        return None
    cls, v = p[0], [float(t) for t in p[1:]]
    if len(v) == 4:  # box: cx cy w h
        cx, cy = _xy(v[0], v[1], how)
        w, h = (v[3], v[2]) if how in ('rot90cw', 'rot90ccw') else (v[2], v[3])
        return f'{cls} {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}'
    pts = []  # polygon: x1 y1 x2 y2 ...
    for i in range(0, len(v) - 1, 2):
        pts += [min(max(c, 0.0), 1.0) for c in _xy(v[i], v[i + 1], how)]
    return cls + ' ' + ' '.join(f'{c:.6f}' for c in pts)


def _colour(img):
    """Random brightness / contrast / saturation change (labels unaffected)."""
    out = img.astype(np.float32) * random.uniform(0.8, 1.2) + random.uniform(-20, 20)
    out = np.clip(out, 0, 255).astype(np.uint8)
    hsv = cv2.cvtColor(out, cv2.COLOR_BGR2HSV).astype(np.float32)
    hsv[..., 1] = np.clip(hsv[..., 1] * random.uniform(0.7, 1.3), 0, 255)
    return cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2BGR)


def augment_train_split(copies):
    if copies <= 0:
        return
    random.seed(0)
    img_dir, lbl_dir = f'{DATA_DIR}/train/images', f'{DATA_DIR}/train/labels'
    originals = [p for p in sorted(glob.glob(f'{img_dir}/*')) if '_aug' not in os.path.basename(p)]
    before, made = len(originals), 0
    for path in originals:
        stem, ext = os.path.splitext(os.path.basename(path))
        lbl_path = f'{lbl_dir}/{stem}.txt'
        if not os.path.exists(lbl_path):
            continue
        img = cv2.imread(path)
        lines = [l.strip() for l in open(lbl_path) if l.strip()]
        for k in range(1, copies + 1):
            new_stem = f'{stem}_aug{k}'
            if os.path.exists(f'{lbl_dir}/{new_stem}.txt'):
                continue
            how = random.choice(GEOMETRY)
            new_lines = [x for x in (_label_line(l, how) for l in lines) if x]
            cv2.imwrite(f'{img_dir}/{new_stem}{ext}', _colour(_image(img, how)))
            open(f'{lbl_dir}/{new_stem}.txt', 'w').write('\n'.join(new_lines) + '\n')
            made += 1
    print(f'Augmentation: {before} original train images, {made} new copies written, '
          f'{len(glob.glob(f"{img_dir}/*"))} train images in total')


augment_train_split(AUG_COPIES)

# ---------------- train ----------------
# Use 'yolo26s-seg.pt' if your ultralytics version supports it.
model = YOLO(MODEL_NAME)

model.train(
    data=YAML_PATH,
    task='segment',
    epochs=EPOCHS,
    imgsz=IMGSZ,        # photos are ~2048px and parts are small
    batch=BATCH,        # A100 40GB (16 at imgsz 1024, 4 at imgsz 2048)
    workers=8,          # faster data loading
    cache='ram',        # small dataset: keep images in RAM (High-RAM runtime recommended)
    amp=True,           # mixed precision uses the A100 tensor cores
    patience=0,         # 0 = never early-stop; best.pt is still kept
    optimizer='AdamW',  # explicit, gentle optimizer instead of 'auto'
    lr0=0.0005,
    cos_lr=True,
    warmup_epochs=5,
    close_mosaic=30,
    project=RUNS_DIR,
    name=RUN_NAME,
    # stronger augmentation to stretch a very small dataset
    degrees=15, fliplr=0.5, flipud=0.2, scale=0.5, translate=0.1,
    hsv_h=0.02, hsv_s=0.6, hsv_v=0.4, mosaic=1.0,
)

# Folder this run really saved to (may differ from RUN_NAME if that name already existed)
SAVE_DIR = str(model.trainer.save_dir)
print('Run saved to:', SAVE_DIR)

# ---------------- training graphs ----------------
df = pd.read_csv(f'{SAVE_DIR}/results.csv')
df.columns = df.columns.str.strip()

groups = {
    'Box loss': ['train/box_loss', 'val/box_loss'],
    'Seg loss': ['train/seg_loss', 'val/seg_loss'],
    'Cls loss': ['train/cls_loss', 'val/cls_loss'],
    'Box mAP': ['metrics/mAP50(B)', 'metrics/mAP50-95(B)'],
    'Mask mAP': ['metrics/mAP50(M)', 'metrics/mAP50-95(M)'],
    'Precision / Recall (mask)': ['metrics/precision(M)', 'metrics/recall(M)'],
}

fig, axes = plt.subplots(2, 3, figsize=(16, 8))
for ax, (title, cols) in zip(axes.flat, groups.items()):
    for c in cols:
        if c in df.columns:
            ax.plot(df['epoch'], df[c], label=c)
    ax.set_title(title)
    ax.set_xlabel('epoch')
    ax.grid(alpha=0.3)
    ax.legend()
plt.tight_layout()
plt.show()

# Ultralytics' own plots
for name in ['results.png', 'confusion_matrix.png', 'MaskPR_curve.png', 'MaskF1_curve.png']:
    try:
        display(Image(f'{SAVE_DIR}/{name}'))
    except Exception as e:
        print('skip', name, e)

# ---------------- validate + predict ----------------
best = YOLO(f'{SAVE_DIR}/weights/best.pt')
best.val(data=YAML_PATH, split='test', imgsz=IMGSZ, conf=0.4, agnostic_nms=True)

os.makedirs(PRED_DIR, exist_ok=True)

FONT_SIZE = 8       # label text size in px; lower = smaller
LINE_WIDTH = 1      # box / mask outline thickness
SHOW_BOXES = False  # False = masks only, less clutter

results = best.predict(f'{DATA_DIR}/test/images', conf=0.4, imgsz=IMGSZ, agnostic_nms=True, save=False)
for r in results:
    im = r.plot(font_size=FONT_SIZE, line_width=LINE_WIDTH, boxes=SHOW_BOXES)
    cv2.imwrite(f'{PRED_DIR}/{os.path.basename(r.path)}', im)

# ---------------- view results ----------------
for f in sorted(glob.glob(f'{PRED_DIR}/*.jpg'))[:5]:
    display(Image(f, width=900))

# ---------------- download weights ----------------
files.download(f'{SAVE_DIR}/weights/best.pt')
