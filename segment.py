# YOLO segmentation training for Google Colab.
# Each "# %%" block is a cell: paste them into Colab one at a time
# (or upload this file and run it with `%run train_seg_colab.py`).
# Runtime -> Change runtime type -> T4 GPU before starting.

# %% Cell 1: install and check GPU
# !pip install -q ultralytics
# !nvidia-smi

# %% Cell 2: mount Drive and unzip the dataset
# Put the zip at MyDrive/topgun/ first, or upload it to /content and edit ZIP_PATH.
from google.colab import drive
drive.mount('/content/drive')

ZIP_PATH = '/content/drive/MyDrive/topgun/TOPGUN 3rd.v3i.yolo26.zip'
DATA_DIR = '/content/dataset'

import os, subprocess
os.makedirs(DATA_DIR, exist_ok=True)
subprocess.run(['unzip', '-q', '-o', ZIP_PATH, '-d', DATA_DIR], check=True)
print(os.listdir(DATA_DIR))

# %% Cell 3: fix paths in data.yaml (Roboflow's relative paths break in Colab)
import yaml

yaml_path = f'{DATA_DIR}/data.yaml'
cfg = yaml.safe_load(open(yaml_path))
cfg['path'] = DATA_DIR
cfg['train'] = 'train/images'
cfg['val'] = 'valid/images'
cfg['test'] = 'test/images'
yaml.safe_dump(cfg, open(yaml_path, 'w'))
print(cfg)

# %% Cell 4: train
from ultralytics import YOLO

RUNS_DIR = '/content/drive/MyDrive/topgun/runs'  # on Drive so it survives disconnects

# Use 'yolo26s-seg.pt' if your ultralytics version supports it.
model = YOLO('yolo11s-seg.pt')

model.train(
    data=yaml_path,
    task='segment',
    epochs=250,
    imgsz=1024,         # photos are ~2048px and parts are small
    batch=2,              # 1024px images: smaller batch to fit T4 memory
    patience=0,           # 0 = never early-stop; best.pt is still kept
    optimizer='AdamW',    # explicit, gentle optimizer instead of 'auto'
    lr0=0.0005,
    cos_lr=True,
    warmup_epochs=5,
    close_mosaic=30,
    project=RUNS_DIR,
    name='seg3',
    # stronger augmentation to stretch a very small dataset
    degrees=15, fliplr=0.5, flipud=0.2, scale=0.5, translate=0.1,
    hsv_h=0.02, hsv_s=0.6, hsv_v=0.4, mosaic=1.0,
)

# %% Cell 4b: training graphs (run right after training)
import pandas as pd
import matplotlib.pyplot as plt
from IPython.display import Image, display

df = pd.read_csv(f'{RUNS_DIR}/seg3/results.csv')
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
        display(Image(f'{RUNS_DIR}/seg3/{name}'))
    except Exception as e:
        print('skip', name, e)

# %% Cell 5: validate on the test split and predict
best = YOLO(f'{RUNS_DIR}/seg3/weights/best.pt')
best.val(data=yaml_path, split='test', imgsz=1024, conf=0.4, agnostic_nms=True)  # one box per object, realistic confusion matrix

import cv2

PRED_DIR = '/content/predictions'
os.makedirs(PRED_DIR, exist_ok=True)

FONT_SIZE = 8      # label text size in px; lower = smaller (default is scaled to image size)
LINE_WIDTH = 1     # box / mask outline thickness
SHOW_BOXES = False # False = masks only, less clutter

results = best.predict(f'{DATA_DIR}/test/images', conf=0.4, imgsz=1024, agnostic_nms=True, save=False)
for r in results:
    im = r.plot(font_size=FONT_SIZE, line_width=LINE_WIDTH, boxes=SHOW_BOXES)
    cv2.imwrite(f'{PRED_DIR}/{os.path.basename(r.path)}', im)

# %% Cell 6: view results
import glob
from IPython.display import Image, display

display(Image(f'{RUNS_DIR}/seg3/results.png'))
for f in sorted(glob.glob(f'{PRED_DIR}/*.jpg'))[:5]:
    display(Image(f, width=900))

# %% Cell 7: download the weights
from google.colab import files
files.download(f'{RUNS_DIR}/seg3/weights/best.pt')