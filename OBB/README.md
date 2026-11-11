# OBB training (oriented bounding boxes)

Trains a YOLO OBB model on the same Roboflow export used for the segmentation runs. No relabeling:
the polygon labels are converted to rotated rectangles automatically.

## Run order in Colab (A100 GPU runtime)
1. `step1_install.py` – installs ultralytics, shows the GPU.
2. `step2_dataset_obb.py` – unzips the dataset from `MyDrive/topgun/`, converts polygons to OBB, augments the train split (`AUG_COPIES`), writes `data.yaml`. Output: `/content/dataset_obb`.
3. `step3_train_obb.py` – trains `yolo11s-obb.pt`, plots graphs, validates on the valid and test split, saves prediction pictures with rotated boxes, prints a predicted-vs-true count table per class for several `conf` values, downloads `best.pt`.

Upload the three files to Colab (same folder) or paste each into a cell. Each file is self-contained.

## Same settings as the best segmentation run (seg6-aug)
yolo11s, imgsz 1024, batch 16, 250 epochs, AdamW lr0 0.0005, cosine schedule, 5 warm-up epochs, close_mosaic 30,
same built-in augmentation, 2 offline augmented copies. Only the model/label type changes (seg -> OBB),
so the result is directly comparable. Log it in `../progress.md`.

## Things to know
- **Metrics:** OBB reports box-style metrics (`metrics/mAP50(B)` etc.) computed with rotated-box IoU. They are close to, but not identical to, the mask mAP50 of the seg runs. For counting, compare the **count table** at the end of step 3.
- **No head/tail direction:** a rotated rectangle looks the same turned 180 degrees. Not needed for count + class.
- **Offline augmentation:** only flips, 90-degree rotations and colour changes, applied to the train split only. The 4 corners are transformed exactly.
- **Parts rarely overlap in this dataset** (8 neighbouring pairs overlapping >30 % in 684 parts with ordinary boxes, 5 with OBB), so do not expect OBB to change counts much; the main hope is cleaner size information for the size-based classes.
- **Missing labels hurt every model.** At least one clearly visible screw in the valid image has no label (see `../obb_preview/unlabeled_screw_valid.jpg`). Fix such labels in Roboflow before judging results.
- `step2` rebuilds `/content/dataset_obb` from scratch every time it runs.
