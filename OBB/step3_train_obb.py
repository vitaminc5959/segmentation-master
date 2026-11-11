# Step 3 (OBB): train -> graphs -> validate -> predict + count table -> download weights.
# Run step2_dataset_obb.py first (it builds /content/dataset_obb).
# Settings = the best segmentation setup (seg6-aug) with an OBB model, so the runs are comparable.
import collections
import glob
import math
import os

import cv2
import matplotlib.pyplot as plt
import pandas as pd
from google.colab import files
from IPython.display import Image, display
from ultralytics import YOLO

OBB_DIR = '/content/dataset_obb'
YAML_PATH = f'{OBB_DIR}/data.yaml'
RUNS_DIR = '/content/drive/MyDrive/topgun/runs'  # on Drive so it survives disconnects
RUN_NAME = 'obb1-s'            # new name per run; Ultralytics adds -2, -3... if the folder exists
MODEL_NAME = 'yolo11s-obb.pt'  # pretrained OBB model (use 'yolo26s-obb.pt' if your ultralytics supports it)
IMGSZ = 1024
BATCH = 16                     # A100 40GB at imgsz 1024
EPOCHS = 250
CONF = 0.4                     # confidence threshold for val / predict pictures
CONF_SWEEP = [0.25, 0.3, 0.4, 0.5, 0.6, 0.7]  # thresholds for the count table
PRED_DIR = '/content/predictions_obb'

# ---------------- train ----------------
model = YOLO(MODEL_NAME)

model.train(
    data=YAML_PATH,
    task='obb',
    epochs=EPOCHS,
    imgsz=IMGSZ,
    batch=BATCH,
    workers=8,
    cache='ram',        # small dataset: keep images in RAM (High-RAM runtime recommended)
    amp=True,
    patience=0,         # 0 = never early-stop; best.pt is still kept
    optimizer='AdamW',
    lr0=0.0005,
    cos_lr=True,
    warmup_epochs=5,
    close_mosaic=30,
    project=RUNS_DIR,
    name=RUN_NAME,
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
    'Cls loss': ['train/cls_loss', 'val/cls_loss'],
    'DFL loss': ['train/dfl_loss', 'val/dfl_loss'],
    'Angle loss': ['train/angle_loss', 'val/angle_loss'],
    'mAP': ['metrics/mAP50(B)', 'metrics/mAP50-95(B)'],
    'Precision / Recall': ['metrics/precision(B)', 'metrics/recall(B)'],
}
groups = {k: [c for c in v if c in df.columns] for k, v in groups.items()}
groups = {k: v for k, v in groups.items() if v}  # drop groups this ultralytics version does not log

rows = math.ceil(len(groups) / 3)
fig, axes = plt.subplots(rows, 3, figsize=(16, 4 * rows), squeeze=False)
for ax in axes.flat[len(groups):]:
    ax.axis('off')
for ax, (title, cols) in zip(axes.flat, groups.items()):
    for c in cols:
        ax.plot(df['epoch'], df[c], label=c)
    ax.set_title(title)
    ax.set_xlabel('epoch')
    ax.grid(alpha=0.3)
    ax.legend()
plt.tight_layout()
plt.show()

# Ultralytics' own plots
for name in ['results.png', 'confusion_matrix.png', 'BoxPR_curve.png', 'BoxF1_curve.png']:
    try:
        display(Image(f'{SAVE_DIR}/{name}'))
    except Exception as e:
        print('skip', name, e)

# ---------------- validate (valid and test split) ----------------
best = YOLO(f'{SAVE_DIR}/weights/best.pt')
for split in ['val', 'test']:
    print(f'===== {split} split =====')
    best.val(data=YAML_PATH, split=split, imgsz=IMGSZ, conf=CONF, agnostic_nms=True)

# ---------------- predict pictures (oriented boxes) ----------------
os.makedirs(PRED_DIR, exist_ok=True)
FONT_SIZE = 8       # label text size in px; lower = smaller
LINE_WIDTH = 1      # box outline thickness

for folder, tag in [('valid', 'valid'), ('test', 'test')]:
    results = best.predict(f'{OBB_DIR}/{folder}/images', conf=CONF, imgsz=IMGSZ, agnostic_nms=True, save=False)
    for r in results:
        im = r.plot(font_size=FONT_SIZE, line_width=LINE_WIDTH)
        cv2.imwrite(f'{PRED_DIR}/{tag}_{os.path.basename(r.path)}', im)

for f in sorted(glob.glob(f'{PRED_DIR}/*.jpg'))[:5]:
    display(Image(f, width=900))

# ---------------- count table: predicted vs true parts per class (valid split) ----------------
names = [best.names[i] for i in sorted(best.names)]

true = collections.Counter()
for f in glob.glob(f'{OBB_DIR}/valid/labels/*.txt'):
    for line in open(f):
        if line.strip():
            true[best.names[int(line.split()[0])]] += 1

print(f"{'conf':>10} | " + ' '.join(f'{n:>9}' for n in names) + ' | total abs error')
print(f"{'TRUE':>10} | " + ' '.join(f'{true[n]:>9}' for n in names) + ' |')
for conf in CONF_SWEEP:
    pred = collections.Counter()
    for r in best.predict(f'{OBB_DIR}/valid/images', conf=conf, imgsz=IMGSZ, agnostic_nms=True, verbose=False):
        for c in r.obb.cls.tolist():
            pred[best.names[int(c)]] += 1
    err = sum(abs(pred[n] - true[n]) for n in names)
    print(f'{conf:>10} | ' + ' '.join(f'{pred[n]:>9}' for n in names) + f' | {err}')

# ---------------- download weights ----------------
files.download(f'{SAVE_DIR}/weights/best.pt')
