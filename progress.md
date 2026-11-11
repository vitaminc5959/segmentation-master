# Progress Log – TOPGUN segmentation

Goal: train a YOLO model on the TOPGUN 3rd dataset (Roboflow export) in Google Colab.
What matters: **correct part count and correct class**. Mask outline quality does not matter.

Classes: BK, LONG, MIDBig, MIDSmall, SMALLBig

> Numbers marked `~` were read by eye from the plots, so they are approximate.
> Replace them with exact values from `results.csv` when you can.

---

## 1. Code changes

| # | Change | File(s) | Reason / impact |
|---|--------|---------|-----------------|
| 1 | Split `segment.py` into one script per cell | `step1_install.py`, `step2_dataset.py`, `step3_train.py` (plus shared `config.py`, later removed) | Easier to run and edit step by step |
| 2 | Merged train + graphs + predict + view results + download into one step | `step3_train.py` | One run does the whole pipeline; old `step4`–`step7` files deleted |
| 3 | Tuned training for an A100: `batch` 2 -> 16, `workers=8`, `cache='ram'`, `amp=True` | `step3_train.py` | Much faster training. Fewer updates per epoch, so `lr0` may need a small increase (suggested ~0.001) |
| 4 | Removed `config.py`; each step defines its own paths | `step2_dataset.py`, `step3_train.py` | Fixes `ModuleNotFoundError: No module named 'config'` in Colab |
| 5 | Zip is found with `glob('/content/drive/MyDrive/topgun/TOPGUN*.zip')` | `step2_dataset.py` | Works even if the exact file name differs. Raises `IndexError` if the zip is not in `MyDrive/topgun/` |

`imgsz` is still 1024 in training, `val` and `predict`. If it is changed, change it in all three places.

---

## 2. Hardware decision

| Option | Speed | Memory | Verdict |
|--------|-------|--------|---------|
| T4 GPU | Slowest GPU | 16 GB | Safe free-tier choice, needs `batch=2` at 1024 px |
| L4 / G4 GPU | Fast | 24 GB | Good middle ground |
| A100 GPU | Fastest on Colab | 40 GB | **Chosen**, allows `batch=16` |
| RTX 5080 (local) | Comparable to A100 at small batch | 16 GB | Fine if owned: no session limits, needs recent PyTorch/CUDA |
| CPU / TPU | – | – | Not suitable for YOLO training |

---

## 3. Run results: `seg3` (yolo11s-seg, imgsz 1024, batch 2, 250 epochs)

### 3.1 Final metrics (epoch 250)

| Metric | Box (B) | Mask (M) |
|--------|---------|----------|
| mAP50 | ~0.71 | ~0.69 |
| mAP50-95 | ~0.52 | ~0.385 |
| Precision | – | ~0.64 |
| Recall | – | ~0.71 |

| Loss | Train | Val |
|------|-------|-----|
| Box | ~0.8–1.0 | ~1.12 |
| Seg | ~1.3–1.5 | ~1.9 |
| Cls | ~0.9–1.0 | ~1.05 |

Relevant for this project: **mAP50, precision, recall and the confusion matrix**. mAP50-95 can be ignored (it measures outline tightness).

### 3.2 Training behaviour

| Epochs | What happened |
|--------|---------------|
| 0–50 | Almost no learning (mAP ~0) |
| 50–90 | Loss spikes (box up to ~3.2, seg ~4.6, val cls ~11.5), model unstable |
| 90–200 | Steady improvement |
| 200–250 | Flat, mAP no longer improving |

Conclusions:
- About a third of the training was wasted on the unstable start.
- Train seg loss (~1.4) vs val seg loss (~1.9) suggests some overfitting on a small dataset.

### 3.3 Confusion matrix (counts)

Rows = predicted, columns = true. `-` = 0.

| Predicted \ True | BK | LONG | MIDBig | MIDSmall | SMALLBig | background (no label) |
|------------------|----|------|--------|----------|----------|-----------------------|
| BK | **8** | 1 | - | - | - | 24 |
| LONG | - | **29** | - | 1 | 1 | 77 |
| MIDBig | - | - | **2** | 3 | 7 | 149 |
| MIDSmall | - | - | 6 | **67** | 2 | 124 |
| SMALLBig | - | - | 6 | 1 | **16** | 76 |
| background (missed) | - | - | 1 | 3 | 1 | - |

### 3.4 Per-class summary (derived from the matrix above)

| Class | True objects | Correct | Total predicted | Recall (correct / true) | Precision (correct / predicted) |
|-------|--------------|---------|-----------------|-------------------------|---------------------------------|
| BK | 8 | 8 | 33 | 1.00 | 0.24 |
| LONG | 30 | 29 | 108 | 0.97 | 0.27 |
| MIDBig | 15 | 2 | 161 | 0.13 | 0.01 |
| MIDSmall | 74 | 67 | 199 | 0.91 | 0.34 |
| SMALLBig | 27 | 16 | 99 | 0.59 | 0.16 |
| **Total** | **154** | **122** | **600** | 0.79 | 0.20 |

Caution: these precision values are far lower than the 0.64 precision reported in the training plot. The confusion matrix was probably built with a lower confidence threshold, so the false positives are inflated compared to `conf=0.4`. Treat the matrix as showing *where* errors happen, not exact rates.

### 3.5 Findings

1. **Most errors are false positives** (predictions with no labeled object): 149 for MIDBig, 124 for MIDSmall, 77 for LONG, 76 for SMALLBig, 24 for BK. These directly inflate part counts.
2. **MIDBig is the weakest class**: 2 of 15 found correctly. It is confused with SMALLBig and MIDSmall (6 + 6 predicted as other classes, and 3 + 7 mispredicted into MIDBig).
3. **Missed objects are few** (5 total), so recall is not the main problem.
4. **LONG and BK are reliable** when detected.

---

## 4. Open ideas (not yet applied)

| Idea | Expected impact | Status |
|------|-----------------|--------|
| Check MIDBig labels (missing labels or MIDBig/SMALLBig mix-up) | Likely the largest gain, no cost | To do |
| Lighter augmentation (`degrees`, `scale`, `hsv_*`) | Remove the unstable first ~90 epochs | To do |
| Raise `conf` to 0.5–0.6 when predicting | Fewer false positives, may miss a few parts | To test |
| Bigger model `yolo11m-seg.pt` | Possible small gain, may overfit more | To test |
| `imgsz=1280` (or 2048) | Better small-part detection. 2048 costs ~4x time and needs `batch` ~4–8 | To test |
| Switch to a detection model (`yolo11m.pt`), no masks | Faster, more stable. Box mAP50 (~0.71) already >= mask mAP50 (~0.69). Should train from the polygon labels, verify in epoch 1 | Proposed |
| Add part count per class to the predict step | Directly measures what matters | Proposed |
| More labeled images | Helps more than any model change | If possible |

---

## 5. Run history (add a column after each training run)

### 5.1 Parameters used per run

`(d)` = Ultralytics default, not set explicitly in the script. Check the `args.yaml` in the run folder to confirm the defaults for your installed version.

| Parameter | seg3 (baseline) | seg3-6 / seg3-7 (A100 settings) |
|-----------|-----------------|-----------------------------------------------------|
| Model | yolo11s-seg.pt | yolo11s-seg.pt |
| Task | segment | segment |
| GPU | T4 | A100 |
| Epochs | 250 | 250 |
| imgsz | 1024 | 1024 |
| batch | 2 | 16 |
| workers | (d) 8 | 8 |
| cache | (d) off | ram |
| amp | (d) True | True |
| patience | 0 (never early-stop) | 0 (never early-stop) |
| **Optimizer** | AdamW | AdamW |
| lr0 (initial LR) | 0.0005 | 0.0005 |
| lrf (final LR fraction) | (d) 0.01 | (d) 0.01 |
| momentum | (d) 0.937 | (d) 0.937 |
| weight_decay | (d) 0.0005 | (d) 0.0005 |
| **Scheduler** | cos_lr=True (cosine decay) | cos_lr=True (cosine decay) |
| warmup_epochs | 5 | 5 |
| close_mosaic | 30 (mosaic off for last 30 epochs) | 30 |
| mosaic | 1.0 | 1.0 |
| degrees | 15 | 15 |
| fliplr | 0.5 | 0.5 |
| flipud | 0.2 | 0.2 |
| scale | 0.5 | 0.5 |
| translate | 0.1 | 0.1 |
| hsv_h / hsv_s / hsv_v | 0.02 / 0.6 / 0.4 | 0.02 / 0.6 / 0.4 |
| Val/predict conf | 0.4 | 0.4 |
| agnostic_nms (val/predict) | True | True |
| Run folder | `runs/seg3` | `runs/seg3-6`, `runs/seg3-7` (Ultralytics adds `-N` when the name exists) |

Note: when `RUN_NAME` already exists, Ultralytics saves to a new folder (`seg3-2`, `seg3-3`...). Before the fix, the graph/confusion-matrix cells kept reading the old `seg3` folder. `step3_train.py` now reads from `model.trainer.save_dir`, and `RUN_NAME` is `seg4`.

### 5.2 Results per run

mAP50 values are from the **last epoch** in `results.csv`, read from `args.yaml` / `results.csv` in each run folder. Optimizer and scheduler are only listed for runs where I know them; `seg1`, `seg2` and `seg2-2` came from earlier script versions, so check their `args.yaml` for `optimizer` and `cos_lr`.

| Run | Model | imgsz | batch | Epochs planned / done | Box mAP50 | Mask mAP50 | Notes |
|-----|-------|-------|-------|-----------------------|-----------|------------|-------|
| seg1 | yolo11s-seg | 640 | 8 | 150 / 57 | 0.000 | 0.000 | Stopped early, never learned |
| seg2 | yolo11s-seg | 640 | 4 | 300 / 300 | 0.727 | 0.690 | First complete run |
| seg2-2 | yolo11s-seg | 640 | 4 | 300 / 300 | 0.727 | 0.690 | Same settings as seg2, identical result |
| seg3 | yolo11s-seg | 1024 | 2 | 250 / 250 | 0.714 | 0.693 | T4 baseline (AdamW, lr0 0.0005, cosine). Unstable epochs 50–90, many false positives |
| seg3-2 | yolo11s-seg | 2048 | 2 | 250 / 6 | 0.015 | 0.008 | `imgsz=2048` attempt, stopped after 6 epochs (cause unknown) |
| seg3-3 | – | – | – | – | – | – | No results.csv, failed at start |
| seg3-4 | – | – | – | – | – | – | No results.csv, failed at start |
| seg3-5 | yolo11s-seg | 1024 | 2 | 250 / 21 | 0.002 | 0.002 | Stopped after 21 epochs |
| **seg3-6** | yolo11s-seg | 1024 | 16 | 250 / 250 | **0.825** | **0.825** | A100 settings (AdamW, lr0 0.0005, cosine, cache=ram) |
| **seg3-7** | yolo11s-seg | 1024 | 16 | 250 / 250 | **0.825** | **0.825** | Repeat of seg3-6, identical result |
| seg4-m | yolo11m-seg | 1024 | 16 | 250 / 250 | ~0.805 | ~0.805 | Only change vs seg3-6: s -> m model. **Assumed** to be the run behind the plots below (final val mAP50 ~0.805, lower than seg3-6's 0.825). Confirm with `args.yaml` |
| seg5-2k | yolo11s-seg | 2048 | 4 | 250 / 250 | ~0.745 | ~0.75 | Back to the small model. Changes vs seg3-6: imgsz 1024 -> 2048 and batch 16 -> 4. **Assumed** to be the run behind the plots in 5.5. Worse mAP50 than seg3-6 (0.825), see 5.5 |
| **seg6-aug** | yolo11s-seg | 1024 | 16 | 250 / 250 | **~0.865** | **~0.86** | Same as seg3-6 plus offline augmentation of the train split (`AUG_COPIES=2`: flips, 90° rotations, colour jitter -> about 3x train images). **Best run so far.** **Assumed** to be the run behind the plots in 5.6 |
| seg7-aug4 | yolo11s-seg | 1024 | 16 | 250 / 250 | ~0.845 | ~0.845 | Same as seg6-aug but `AUG_COPIES=4` (about 5x train images). **Assumed** run behind plots in 5.7. No gain over seg6-aug (slightly lower), 5x the epoch time |
| seg8-aug3-e350 | yolo11s-seg | 1024 | 16 | 350 / 350 | ~0.86 | ~0.86 | `AUG_COPIES=3` (about 4x train images) and `epochs=350`. **Assumed** run behind plots in 5.8. Same level as seg6-aug, no real gain from more copies/epochs |
| seg9-e300 | yolo11s-seg | 1024 | 16 | 300 / 300 | ~0.845 (best.pt 0.869) | ~0.845 (best.pt 0.867) | seg6-aug setup (`AUG_COPIES=2`) with `epochs=300` (only change: 250 -> 300). **Assumed** run behind plots in 5.9. Same plateau as seg6-aug, no gain from 300 epochs |
| obb1-s | yolo11s-obb | 1024 | 16 | 250 / – | | | **Planned.** Code in `OBB/`. seg6-aug settings with an oriented-box model; polygon labels converted to rotated rectangles (`cv2.minAreaRect`), `AUG_COPIES=2`. Compare the box-style mAP50 and the per-class **count table**, not mask mAP. Fill in after the run |

### 5.10 OBB preparation notes

- Dataset check (local copy of the v3 zip): **10 images only** (6 train / 2 valid / 2 test), 387 / 155 / 142 labeled parts, images 2048x2048 with black padding.
- Polygon -> OBB conversion tested locally on all 684 parts: counts match the source labels, 0 bad lines. A rotated rectangle covers ~65 % of its box area with real part vs ~51 % for an ordinary box.
- Neighbouring boxes rarely overlap in this data (pairs with IoU > 0.3: 8 with ordinary boxes, 5 with OBB), so a big change in counts from OBB is not expected.
- Label problems seen in the pictures (`obb_preview/`): at least one clearly visible isolated screw in the valid image has **no label**; one diagonal screw labeled SMALLBig looks as long as the LONG screws (check against the class definition).
- OBB has no head/tail direction; not needed for count + class.

### 5.4 seg4-m details (valid split, from plots)

| Metric | Value |
|--------|-------|
| Box / Mask mAP50 | ~0.805 / ~0.805 |
| Box / Mask mAP50-95 | ~0.585 / ~0.43 |
| Mask precision / recall | ~0.70 / ~0.85 |

Training behaviour: unstable. Loss spike at epochs ~25–40 (seg loss up to ~15, val cls loss up to ~110), then mAP collapsed to 0 around epochs ~95–135, and only recovered to a stable level after ~140. Val cls loss spiked again around epochs 55, 100 and 125.

Confusion matrix (valid split; rows = predicted, columns = true):

| Predicted \ True | BK | LONG | MIDBig | MIDSmall | SMALLBig | background (no label) |
|------------------|----|------|--------|----------|----------|-----------------------|
| BK | **8** | - | - | - | - | 14 |
| LONG | - | **30** | 1 | 3 | 1 | 96 |
| MIDBig | - | - | **4** | 2 | 5 | 145 |
| MIDSmall | - | - | 4 | **65** | 1 | 115 |
| SMALLBig | - | - | 6 | 4 | **20** | 76 |
| background (missed) | - | - | - | 1 | - | - |

Change vs seg3 (small model, T4 run): correct detections MIDBig 2 -> 4, SMALLBig 16 -> 20, LONG 29 -> 30, MIDSmall 67 -> 65, BK 8 -> 8. False positives: BK 24 -> 14, MIDBig 149 -> 145, MIDSmall 124 -> 115, SMALLBig 76 -> 76, LONG 77 -> 96 (worse).

Test split (2 images, 142 objects, `conf=0.4`):

| Class | Instances | Precision | Recall | mAP50 (mask) |
|-------|-----------|-----------|--------|--------------|
| all | 142 | 0.812 | 0.772 | 0.737 |
| BK | 12 | 1.00 | 1.00 | 0.995 |
| LONG | 33 | 1.00 | 1.00 | 0.995 |
| MIDBig | 15 | 0.50 | 0.067 | 0.065 |
| MIDSmall | 64 | 0.921 | 0.906 | 0.865 |
| SMALLBig | 18 | 0.64 | 0.889 | 0.765 |

Findings:
- Test split has only 2 images, so these numbers are very noisy.
- At `conf=0.4` precision is 0.81, so most of the false positives in the confusion matrix are probably low-confidence predictions (the matrix uses a lower threshold). A higher `conf` cuts them.
- **MIDBig is still the problem**: only 1 of 15 found on test (recall 0.067).
- Training is still unstable even at batch 16 (collapse in epochs ~95–135).

### 5.3 What changed between runs

| Change | From | To | Effect on mask mAP50 |
|--------|------|----|----------------------|
| `imgsz` 640 -> 1024 (seg2 -> seg3) | 0.690 | 0.693 | Almost none (box mAP50 even dropped 0.727 -> 0.714) |
| `batch` 2 -> 16, `cache=ram`, `workers=8` (seg3 -> seg3-6) | 0.693 | **0.825** | **+0.13, the largest gain so far** |

Observations:
- Larger `imgsz` alone did not help. The big gain came with the batch-size change, so training stability (batch 16) probably mattered more than resolution.
- seg2/seg2-2 and seg3-6/seg3-7 give **identical** results with identical settings, so training is deterministic (default `seed=0`). Repeating a run with the same settings gives no new information. To test variation, change a parameter or the seed.
- I haven't yet seen the curves or confusion matrix for seg3-6/seg3-7. Check whether the unstable epochs 50–90 are gone and whether the false positives dropped (section 3.3).
- The 2048 px attempt (seg3-2) failed early. The later seg5-2k run completed (see 5.5).

### 5.5 seg5-2k details (valid split, from plots)

| Metric | seg3-6 (1024, s, batch 16) | seg4-m (1024, m, batch 16) | seg5-2k (2048, s, batch 4) |
|--------|----------------------------|----------------------------|----------------------------|
| Box mAP50 | 0.825 | ~0.805 | ~0.745 |
| Mask mAP50 | 0.825 | ~0.805 | ~0.75 |
| Box mAP50-95 | – | ~0.585 | ~0.54 |
| Mask mAP50-95 | – | ~0.43 | ~0.455 |
| Mask precision / recall | – | ~0.70 / ~0.85 | ~0.67 / ~0.80 |

Training behaviour: loss spike at epochs ~50–85 (seg loss up to ~7, val cls loss up to ~13), mAP near 0 until ~epoch 90, then steady improvement and flat from ~epoch 170. No collapse after epoch 100 (unlike seg4-m).

Confusion matrix (valid split; rows = predicted, columns = true):

| Predicted \ True | BK | LONG | MIDBig | MIDSmall | SMALLBig | background (no label) |
|------------------|----|------|--------|----------|----------|-----------------------|
| BK | **8** | - | - | - | - | 19 |
| LONG | - | **30** | - | - | 1 | 48 |
| MIDBig | - | - | **1** | 2 | 4 | 160 |
| MIDSmall | - | - | 7 | **68** | 2 | 125 |
| SMALLBig | - | - | 7 | 5 | **20** | 93 |
| background (missed) | - | - | - | - | - | - |

Test split (2 images, 142 objects, `conf=0.4`):

| Class | Instances | Precision | Recall | mAP50 (mask) |
|-------|-----------|-----------|--------|--------------|
| all | 142 | 0.934 | 0.737 | 0.714 |
| BK | 12 | 1.00 | 0.917 | 0.915 |
| LONG | 33 | 0.97 | 0.97 | 0.965 |
| MIDBig | 15 | 1.00 | 0.067 | 0.065 |
| MIDSmall | 64 | 0.924 | 0.953 | 0.946 |
| SMALLBig | 18 | 0.778 | 0.778 | 0.678 |

Findings:
- **`imgsz=2048` did not improve mAP50** (about 0.75 vs 0.825 at 1024) and takes far longer per epoch. Mask mAP50-95 rose slightly, but outline quality does not matter here.
- Caveat: `batch` also dropped from 16 to 4 in this run, so the effect of image size is not fully separated. The validation set is small, so differences of a few points may be noise.
- LONG false positives dropped (96 -> 48 vs seg4-m), BK and MIDSmall false positives are similar, MIDBig and SMALLBig false positives rose (160 and 93).
- **MIDBig is the same in every run**: 1–4 correct out of 15, with 145–160 predictions that have no label. Changing model size or image size has not moved it, which points to a data/label problem, not a model problem.

### 5.6 seg6-aug details (valid split, from plots): offline augmentation

| Metric | seg3-6 | seg4-m | seg5-2k | **seg6-aug** |
|--------|--------|--------|---------|--------------|
| Box mAP50 (valid, last epoch) | 0.825 | ~0.805 | ~0.745 | **~0.865** (peak ~0.88) |
| Mask mAP50 (valid, last epoch) | 0.825 | ~0.805 | ~0.75 | **~0.86** (peak ~0.88) |
| Mask mAP50-95 | – | ~0.43 | ~0.455 | ~0.515 |
| Mask precision / recall | – | ~0.70 / ~0.85 | ~0.67 / ~0.80 | ~0.84 / ~0.82 |
| Test mAP50 (2 images) | – | 0.737 | 0.714 | **0.82** |

Training behaviour: much better than all earlier runs. Only a short loss spike at epochs ~10–15 (during warmup), no collapse, mAP starts rising from epoch ~20 (earlier runs: ~90) and flattens at epoch ~100. Train/val losses are close together.

Confusion matrix (valid split; rows = predicted, columns = true):

| Predicted \ True | BK | LONG | MIDBig | MIDSmall | SMALLBig | background (no label) |
|------------------|----|------|--------|----------|----------|-----------------------|
| BK | **8** | - | - | - | - | 8 |
| LONG | - | **30** | - | - | 1 | 96 |
| MIDBig | - | - | **10** | 1 | 8 | 142 |
| MIDSmall | - | - | 4 | **74** | 1 | 127 |
| SMALLBig | - | - | 1 | - | **17** | 72 |
| background (missed) | - | - | - | - | - | - |

Correct detections vs seg5-2k: MIDBig 1 -> **10**, MIDSmall 68 -> **74**, SMALLBig 20 -> 17, LONG 30 -> 30, BK 8 -> 8. BK false positives dropped (19 -> 8). No missed objects.

Test split (2 images, 142 objects, `conf=0.4`):

| Class | Instances | Precision | Recall | mAP50 (mask) |
|-------|-----------|-----------|--------|--------------|
| all | 142 | 0.889 | 0.854 | 0.82 |
| BK | 12 | 1.00 | 1.00 | 0.995 |
| LONG | 33 | 1.00 | 1.00 | 0.995 |
| MIDBig | 15 | 0.70 | 0.467 | 0.384 |
| MIDSmall | 64 | 0.954 | 0.969 | 0.959 |
| SMALLBig | 18 | 0.789 | 0.833 | 0.768 |

Findings:
- **Offline augmentation (`AUG_COPIES=2`) was the biggest single improvement**: mask mAP50 0.825 -> ~0.86 on valid, and 0.714–0.737 -> 0.82 on test. Training became stable.
- **MIDBig improved the most**: test recall 0.067 -> 0.467, mAP50 0.065 -> 0.384, 10 correct on the valid confusion matrix (was 1–4). Still the weakest class.
- Likely reason for the stability: about 3x more images per epoch means about 3x more weight updates per epoch, so the model gets through the slow early phase much sooner.
- Large false-positive counts remain in the confusion matrix (MIDBig 142, MIDSmall 127, LONG 96). At `conf=0.4` precision is 0.89, so most are low-confidence. Tune `conf` on the F1 curve.
- Test split is only 2 images, so treat test numbers as rough.

### 5.7 seg7-aug4 details (valid split, from plots): 4 augmented copies (5x train images)

| Metric | seg6-aug (2 copies) | **seg7-aug4 (4 copies)** |
|--------|---------------------|--------------------------|
| Mask mAP50 (valid, last epoch) | ~0.86 | ~0.845 |
| Mask mAP50 (valid, peak) | ~0.88 | ~0.865 |
| Mask mAP50-95 | ~0.515 | ~0.50 |
| Mask precision / recall | ~0.84 / ~0.82 | ~0.84 / ~0.80 |
| Correct MIDBig on confusion matrix | 10 | 7 |

Training behaviour: stable, like seg6-aug. Short spike at epochs ~10–15, no collapse. mAP50 plateaus at ~epoch 100, with a slight decline after ~epoch 180–200 (best.pt keeps the best epoch).

Confusion matrix (valid split; rows = predicted, columns = true):

| Predicted \ True | BK | LONG | MIDBig | MIDSmall | SMALLBig | background (no label) |
|------------------|----|------|--------|----------|----------|-----------------------|
| BK | **8** | - | - | - | - | 8 |
| LONG | - | **30** | - | - | 1 | 118 |
| MIDBig | - | - | **7** | - | 8 | 138 |
| MIDSmall | - | - | 4 | **73** | 1 | 107 |
| SMALLBig | - | - | 4 | 2 | **17** | 74 |
| background (missed) | - | - | - | - | - | - |

Per-class table printed at the end of training (best.pt). **Note: 155 instances, matching the valid-split counts, so this is the valid split. The earlier "test" tables had 142 instances (a different split). Do not compare them directly.**

| Class | Instances | Precision | Recall | mAP50 (mask) |
|-------|-----------|-----------|--------|--------------|
| all | 155 | 0.811 | 0.797 | 0.859 |
| BK | 8 | 0.936 | 1.00 | 0.995 |
| LONG | 30 | 0.883 | 1.00 | 0.976 |
| MIDBig | 15 | 0.55 | 0.40 | 0.611 |
| MIDSmall | 75 | 0.907 | 0.933 | 0.944 |
| SMALLBig | 27 | 0.779 | 0.653 | 0.769 |

Findings:
- **4 copies were no better than 2.** mAP50 is about the same or slightly lower, and MIDBig correct detections dropped (10 -> 7). Diminishing returns, as expected for copies that are only flips, rotations and colour changes of the same photos.
- Each epoch takes about 5x the original time (vs 3x for 2 copies), so this run cost more for no gain. Differences of this size are within noise given the small valid set.
- **Keep `AUG_COPIES=2`** as the better setting.
- MIDBig is still the weakest class (mAP50 0.61 on valid, recall 0.40). More real labeled MIDBig examples or a label check are the next lever.

### 5.8 seg8-aug3-e350 details (valid split, from plots): 3 copies (4x train images), 350 epochs

| Metric | seg6-aug (2 copies, 250 ep) | seg7-aug4 (4 copies, 250 ep) | **seg8-aug3-e350 (3 copies, 350 ep)** |
|--------|-----------------------------|------------------------------|----------------------------------------|
| Mask mAP50 (valid, last epoch) | ~0.86 | ~0.845 | ~0.86 |
| Mask mAP50 (valid, peak) | ~0.88 | ~0.865 | ~0.875 |
| Mask mAP50-95 | ~0.515 | ~0.50 | ~0.495 |
| Mask precision / recall (last epoch) | ~0.84 / ~0.82 | ~0.84 / ~0.80 | ~0.76 / ~0.87 |
| Correct MIDBig (confusion matrix) | 10 | 7 | 5 |

Training behaviour: same stable shape as the other augmented runs (short spike at epochs ~10–15, plateau from epoch ~100). Train loss keeps falling to epoch 350 while val loss is flat from epoch ~150, so the train/val gap keeps growing (slight overfitting). Precision drifted down late in training (~0.85 -> ~0.76) while recall went up.

Confusion matrix (valid split; rows = predicted, columns = true):

| Predicted \ True | BK | LONG | MIDBig | MIDSmall | SMALLBig | background (no label) |
|------------------|----|------|--------|----------|----------|-----------------------|
| BK | **8** | - | - | - | - | 21 |
| LONG | - | **30** | - | - | 1 | 94 |
| MIDBig | - | - | **5** | - | 3 | 129 |
| MIDSmall | - | - | 6 | **73** | 3 | 123 |
| SMALLBig | - | - | 4 | 2 | **20** | 78 |
| background (missed) | - | - | - | - | - | - |

No per-class table was pasted for this run (the pasted text was the script itself).

Findings:
- **No real gain from 3 copies + 350 epochs.** Valid mAP50 is about the same as seg6-aug (2 copies, 250 epochs), with about 2x the training time.
- All three augmented runs land at about 0.845–0.865 mask mAP50, so this setup is at its **plateau**. Differences between them are within noise.
- The MIDBig count varies between runs (10, 7, 5) with no trend, which also points to noise and to MIDBig being limited by the data.
- Best value for cost: **seg6-aug settings (2 copies, 250 epochs, or ~150 epochs since mAP flattens at ~100)**.
- Next gains need to come from the data (MIDBig labels / more examples), choosing `conf`, and a larger valid/test set, not from more augmentation or epochs.

### 5.9 seg9-e300 details (valid split): seg6-aug setup with 300 epochs

| Metric | seg6-aug (250 ep) | seg8-aug3-e350 | **seg9-e300 (300 ep)** |
|--------|-------------------|----------------|-------------------------|
| Mask mAP50 (valid, last epoch, plot) | ~0.86 | ~0.86 | ~0.845 |
| Mask mAP50 (valid, best.pt table) | – | – | **0.867** |
| Mask mAP50-95 (best.pt table) | – | – | 0.542 |
| Mask precision / recall (best.pt table) | – | – | 0.85 / 0.832 |
| Correct MIDBig (confusion matrix) | 10 | 5 | 5 |

Training behaviour: same shape as the other augmented runs. mAP50 plateaus from epoch ~100 and declines slightly after ~epoch 200 (about 0.865 -> 0.845), while train loss keeps falling and val loss is flat. best.pt keeps the best epoch.

Confusion matrix (valid split; rows = predicted, columns = true):

| Predicted \ True | BK | LONG | MIDBig | MIDSmall | SMALLBig | background (no label) |
|------------------|----|------|--------|----------|----------|-----------------------|
| BK | **8** | - | - | - | - | 17 |
| LONG | - | **30** | - | - | 1 | 96 |
| MIDBig | - | - | **5** | - | 1 | 133 |
| MIDSmall | - | - | 5 | **73** | 2 | 114 |
| SMALLBig | - | - | 5 | 2 | **23** | 85 |
| background (missed) | - | - | - | - | - | - |

Per-class table printed at the end of training (best.pt, valid split, 155 instances):

| Class | Instances | Precision | Recall | mAP50 (mask) |
|-------|-----------|-----------|--------|--------------|
| all | 155 | 0.85 | 0.832 | 0.867 |
| BK | 8 | 0.921 | 1.00 | 0.995 |
| LONG | 30 | 0.915 | 1.00 | 0.98 |
| MIDBig | 15 | 0.714 | 0.40 | 0.605 |
| MIDSmall | 75 | 0.906 | 0.92 | 0.943 |
| SMALLBig | 27 | 0.791 | 0.842 | 0.815 |

Findings:
- **300 epochs gave no real gain.** Valid mAP50 is 0.867 vs 0.859 for seg7-aug4, within noise. The plateau holds for 250, 300 and 350 epochs and for 2, 3 and 4 copies.
- Best per-class numbers so far: SMALLBig 0.815 (seg7: 0.769) and overall precision 0.85. MIDBig is unchanged (mAP50 0.605, recall 0.40).
- **Summary of the augmented-run family** (valid mAP50, best.pt/peak): seg6-aug about 0.88 (plot), seg7-aug4 0.859, seg8-aug3-e350 about 0.875 (plot), seg9-e300 0.867. All within noise of each other.
- Conclusion: training settings are tuned out. Remaining levers are MIDBig data, `conf` threshold and a larger valid/test set.
