# Ideas to test – TOPGUN segmentation (small dataset)

Goal: correct **part count + class** per photo. Mask outline does not matter.
Best so far: `seg6-aug` setup (yolo11s-seg, imgsz 1024, batch 16, AdamW lr0 0.0005, cosine, 2 offline augmented copies) → valid mask mAP50 ≈ 0.86–0.88.
Weak points: MIDBig (recall 0.40, confused with SMALLBig / MIDSmall) and many low-confidence false positives.
Already tried, no gain: medium model, imgsz 2048, 3–4 aug copies, 300–350 epochs.

Rules for testing:
- Change **one thing** per run. Start from the `seg6-aug` settings in `step3_train.py`.
- Use one fixed epoch count for the whole campaign. `EPOCHS = 150` is enough (every run flattens by epoch ~100) and halves the cost per test.
- New `RUN_NAME` per run. Add a row to the table in section 6 (and to `progress.md`).
- Compare on the **same** numbers every time (0.3).
- After the single tests, combine the winners into one run and check they still add up.

---

## 0. Before testing anything

### 0.1 Check the dataset is what we think it is
Why: valid = 2 images, test = 2 images. If the images are not really ~2048 px, or if valid/test contain copies of training images, every result so far is misleading. Also: how many MIDBig objects are in **train**? If it is under ~30, that alone explains MIDBig.

```python
import glob, os, cv2, yaml, collections
D = '/content/dataset'
names = yaml.safe_load(open(f'{D}/data.yaml'))['names']

for s in ['train', 'valid', 'test']:
    imgs = [p for p in glob.glob(f'{D}/{s}/images/*') if '_aug' not in p]
    sizes = collections.Counter(cv2.imread(p).shape[:2] for p in imgs)
    cnt = collections.Counter()
    for p in imgs:
        lbl = f'{D}/{s}/labels/{os.path.splitext(os.path.basename(p))[0]}.txt'
        if os.path.exists(lbl):
            for line in open(lbl):
                if line.strip():
                    cnt[names[int(line.split()[0])]] += 1
    print(s, len(imgs), 'images', dict(sizes), '| objects per class:', dict(cnt))

# Leakage check. Roboflow names look like <original>_jpg.rf.<hash>.jpg
def stem(p):
    return os.path.basename(p).split('.rf.')[0]
sets = {s: {stem(p) for p in glob.glob(f'{D}/{s}/images/*')} for s in ['train', 'valid', 'test']}
print('train/valid overlap:', sets['train'] & sets['valid'])
print('train/test overlap :', sets['train'] & sets['test'])
```
Look for: sizes ≈ (1536, 2048) or similar, not (640, 640); MIDBig count in train; both overlap sets empty.
Result: ______

### 0.2 Measure the noise floor
Why: with 2 valid images, the same settings with a different seed can move mAP50 by a few points. Without knowing this spread you cannot tell a real improvement from noise. (seg6-aug / seg7 / seg8 / seg9 all landed at 0.845–0.88, which is probably this spread.)
How: run the baseline (seg6-aug settings, `EPOCHS=150`) three times with `seed=0`, `seed=1`, `seed=2` in `model.train(...)`.
Result: mAP50 = ____ / ____ / ____ → an idea must beat the **best** of these by more than the spread to count.

### 0.3 Fix the metric
From `best.val(...)` on the **valid** split (155 objects) — not the 2-image test split:
1. overall mask mAP50 (best.pt)
2. MIDBig precision and recall
3. count error per class from 1.1 (predicted count − true count)
Record all three for every run.

### 0.4 Bigger valid set (recommended)
Why: 2 images cannot rank ideas reliably, whatever we do.
How: in Roboflow, re-split so valid has 4–5 images (move them from train if needed), export a new version, update `ZIP_PATH` in `step2_dataset.py`. Then redo 0.2 on the new split before testing ideas.

---

## 1. No retraining needed (use the existing best.pt)

### 1.1 Confidence threshold per class → better counts  ← do this first
Hypothesis: the confusion matrix shows 100+ "background" false positives per class, but at conf 0.4 precision is 0.85, so most of them are low-confidence. The right threshold per class gives correct counts without any training.
How: sweep conf and compare predicted vs true counts on valid.
```python
import glob, collections
from ultralytics import YOLO
D = '/content/dataset'
W = '/content/drive/MyDrive/topgun/runs/seg6-aug/weights/best.pt'
m = YOLO(W); names = m.names

true = collections.Counter()
for f in glob.glob(f'{D}/valid/labels/*.txt'):
    for line in open(f):
        if line.strip(): true[names[int(line.split()[0])]] += 1
print('TRUE      ', dict(true))

for conf in [0.25, 0.3, 0.4, 0.5, 0.6, 0.7]:
    pred = collections.Counter()
    for r in m.predict(f'{D}/valid/images', conf=conf, imgsz=1024, agnostic_nms=True, verbose=False):
        for c in r.boxes.cls.tolist(): pred[names[int(c)]] += 1
    print(f'conf {conf:<5}', dict(pred))
```
Pick, per class, the conf where predicted ≈ true. `MaskF1_curve.png` in the run folder shows the same thing: the peak of each class curve is its best conf. Apply per-class thresholds at predict time:
```python
CONF = {'BK': 0.5, 'LONG': 0.5, 'MIDBig': 0.3, 'MIDSmall': 0.5, 'SMALLBig': 0.4}  # fill from the sweep
r = m.predict(img, conf=min(CONF.values()), imgsz=1024, agnostic_nms=True)[0]
keep = [i for i, (c, p) in enumerate(zip(r.boxes.cls.tolist(), r.boxes.conf.tolist()))
        if p >= CONF[names[int(c)]]]
r = r[keep]
```
Cost: minutes. Expected: the biggest cheap gain for counting.
Result: ______

### 1.2 NMS IoU threshold → fewer double counts
Hypothesis: two predictions on the same part = one correct + one false positive. Default `iou=0.7` only merges boxes that overlap a lot; a lower value merges more duplicates.
How: add `iou=0.5` (also 0.6) to `predict` / `val`. Check recall does not drop (touching parts must not be merged away).
Result: ______

### 1.3 Test-time augmentation
How: `m.predict(..., augment=True)` — flips and scales, averaged. Slower; sometimes +1–2 points.
Result: ______

---

## 2. Diagnostics (tell you where the problem is)

### 2.1 Validate on the training split
Why: if MIDBig recall is low even on the images the model trained on, the labels are inconsistent (the model cannot learn a rule that is not there). If train recall is high and valid is low, it is too few examples / overfitting.
How: `m.val(data=YAML_PATH, split='train', imgsz=1024, conf=0.4)` → read the MIDBig row.
Result: train MIDBig recall ____ vs valid ____ → ______

### 2.2 single_cls run: the count ceiling without classes
Why: separates "can it find every part" from "can it name the part". If counting is near perfect with one class, the remaining problem is purely classification → ideas 3.1, 3.3, 3.4, 3.6 and 5.3 are the right direction.
How: `single_cls=True` in `model.train(...)`, `RUN_NAME='diag-singlecls'`.
Result: mAP50 ____ / recall ____ → ______

---

## 3. Training changes (one parameter per run, from seg6-aug settings)

Ranked by how likely I think each one is to help **this** problem.

### 3.1 Less scale augmentation  ← test first
Hypothesis: the classes are size-based (MIDBig / MIDSmall / SMALLBig). `scale=0.5` randomly resizes every training image by 0.5–1.5×, which destroys the size cue the model needs. This matches the symptom: MIDBig is confused with its size neighbours, never with LONG or BK.
How: `scale=0.1` (then `scale=0.0`). Keep everything else.
Risk: less robust to camera distance — only matters if the distance changes in production.
Result: ______

### 3.2 Mosaic off
Hypothesis: mosaic stitches 4 images and rescales them — same size-cue problem as 3.1. With offline copies we may not need it.
How: `mosaic=0.0`.
Result: ______

### 3.3 Copy-paste augmentation (segmentation only)
Hypothesis: MIDBig has few examples. `copy_paste` cuts object masks out of one image and pastes them into another, so rare parts appear in more images and contexts. It is the built-in augmentation that most directly targets a rare class.
How: `copy_paste=0.3` (then 0.5). Needs polygon labels — we have them.
Result: ______

### 3.4 Oversample images that contain MIDBig
Hypothesis: same aim as 3.3, simpler — show MIDBig images more often.
How: in `augment_train_split`, give images whose labels contain MIDBig twice as many copies:
```python
# top of the script, after YAML_PATH
import yaml
names = yaml.safe_load(open(YAML_PATH))['names']
names = dict(enumerate(names)) if isinstance(names, list) else names
MIDBIG_ID = str(next(k for k, v in names.items() if v == 'MIDBig'))

# inside augment_train_split, after `lines = ...`
has_midbig = any(l.split()[0] == MIDBIG_ID for l in lines)
n_copies = copies * 2 if has_midbig else copies
for k in range(1, n_copies + 1):          # was: range(1, copies + 1)
```
Result: ______

### 3.5 Freeze the backbone
Hypothesis: with tiny data the pretrained backbone features are better than anything we can learn; training only the head overfits less.
How: `freeze=10` (first 10 layers = yolo11 backbone). Then `freeze=5`.
Result: ______

### 3.6 Higher classification loss weight
Hypothesis: our errors are class errors, not position errors. Raising `cls` makes the loss care more about the class.
How: `cls=1.0` (default 0.5). Then 2.0.
Result: ______

### 3.7 Calmer warm-up
Hypothesis: every run has a loss spike at epochs 10–15 (end of warm-up). It does not seem to hurt the end result, so expect little.
How: `warmup_epochs=10, lr0=0.0003`.
Result: ______

### 3.8 Cosine end point
Note: `cos_lr=True` is already cosine annealing. The only knob left is where it ends.
How: `lrf=0.1` (ends at 10 % of lr0 instead of 1 %). Expect little.
Result: ______

### 3.9 imgsz 1280 at a real batch size
Hypothesis: 2048 lost, but the batch also dropped to 4. 1280 keeps more detail for small parts and still fits batch 12–16 on the A100.
How: `IMGSZ=1280`, `BATCH=12` (16 if memory allows).
Result: ______

### 3.10 Nano model
Hypothesis: medium did not beat small; a smaller model overfits even less. Low expectation, cheap.
How: `MODEL_NAME='yolo11n-seg.pt'`.
Result: ______

### 3.11 Stronger weight decay (regularization)
Note: `dropout=` is classification-only in Ultralytics and does nothing for a segmentation model, so don't test it.
Hypothesis: train loss keeps falling while val loss is flat (mild overfitting). Higher weight decay is the available regularizer.
How: `weight_decay=0.005` (default 0.0005). Then 0.01 if it helps.
Result: ______

---

## 4. Data (the biggest lever for a small dataset)

### 4.1 Label audit for the size classes
Why: MIDBig has never passed recall ~0.45 in any run. The model cannot learn a boundary that is not consistent in the labels.
How:
1. Write down the rule that separates MIDBig / MIDSmall / SMALLBig (length in mm? compared to a reference part?). If the rule is "by eye", that is the problem.
2. In Roboflow, filter by class MIDBig and look at every instance next to SMALLBig instances. Relabel the borderline ones with the rule.
3. Use 2.1 to confirm: after relabelling, MIDBig recall on **train** should go up.
Result: ______

### 4.2 More real photos with MIDBig parts
Why: augmented copies only repeat what we have. Even 5–10 new photos with many MIDBig parts should move it more than any parameter.
How: photograph, label with the rule from 4.1, add to **train** (not valid). Keep lighting and angle close to production.
Result: ______

### 4.3 Merge classes (business question)
Why: if production does not need MIDBig and SMALLBig (or MIDBig and MIDSmall) counted separately, merging them removes the hardest decision.
How: Roboflow → modify classes → re-export. Decide this before spending more training time.
Result: ______

### 4.4 Bigger valid set
See 0.4. Not an accuracy gain by itself, but it makes every other test trustworthy.

---

## 5. Bigger changes (test after 1–4)

### 5.1 Tiled training + tiled inference (native resolution)
Hypothesis: photos are ~2048 px and parts are small; at imgsz 1024 every part is halved. Cutting each image into 1024 px tiles (with overlap) trains at native resolution **and** gives ~4× more training samples — both problems of this project at once. The 2048 run did not test this fairly (batch 4).
How: offline script that cuts train images into 1024×1024 tiles with 256 px overlap and clips the polygons; at inference predict on tiles and merge with NMS (the SAHI approach). Parts on tile borders are handled by the overlap + merge.
Cost: half a day. Expected: possibly the largest model-side gain. I can write it.
Result: ______

### 5.2 Detection model instead of segmentation
Hypothesis: we do not use the masks. A detect model (`yolo11s.pt`) trains on the same labels (polygons → boxes), with a simpler loss, faster. Box mAP50 has always been ≥ mask mAP50.
How: `MODEL_NAME='yolo11s.pt'`, `task='detect'`, remove the mask plots.
Result: ______

### 5.3 Two-stage: detect every part, then classify each crop
Hypothesis: if 2.2 shows near-perfect counting with one class, classification is the only problem. A small classifier (`yolo11n-cls`) trained on crops gets hundreds of samples from the same photos (one per part). To keep the size cue, crop each part from the **original** 2048 image onto a fixed canvas (e.g. 256×256, padded, not resized) so a bigger part really looks bigger.
Cost: a day.
Result: ______

### 5.4 Seed ensemble
How: train the winning config with seeds 0/1/2, predict with all three, merge boxes (WBF or NMS). Usually +1–3 points and steadier counts. Costs 3× training and 3× inference.
Result: ______

---

## 6. Results table

| # | Idea | Run name | mAP50 (valid) | MIDBig P / R | Count error per class | Keep? |
|---|------|----------|---------------|--------------|-----------------------|-------|
| – | Baseline seg6-aug, seed 0, 150 ep | | | | | – |
| 0.2 | seed 1 | | | | | – |
| 0.2 | seed 2 | | | | | – |
| 1.1 | per-class conf | (no training) | | | | |
| 1.2 | iou=0.5 | (no training) | | | | |
| 2.1 | val on train | (no training) | | | | – |
| 2.2 | single_cls | | | | | – |
| 3.1 | scale=0.1 | | | | | |
| 3.2 | mosaic=0 | | | | | |
| 3.3 | copy_paste=0.3 | | | | | |
| 3.4 | MIDBig oversample | | | | | |
| 3.5 | freeze=10 | | | | | |
| 3.6 | cls=1.0 | | | | | |
| 3.9 | imgsz 1280 | | | | | |
| | combined winners | | | | | |

---

## 7. What carries over to the next small-dataset project

- Pretrained weights + offline augmentation (flips / 90° rotations / colour) gave the biggest training-side gain here. Start there.
- Decide the labelling rule for every class **before** labelling; size-based classes need a measurable rule.
- Keep a valid set of at least 4–5 images and know the seed-to-seed spread before comparing ideas.
- Tune `conf` per class for counting; it is free.
- Resolution and model size are rarely the answer for small data; labels and examples are.
