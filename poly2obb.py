"""Convert YOLO polygon (segment) labels to YOLO OBB labels and draw comparison pictures.

usage: python -I poly2obb.py <dataset_dir> <out_dir>
OBB label format: class x1 y1 x2 y2 x3 y3 x4 y4   (4 corners, normalised 0..1)
"""
import glob
import os
import shutil
import sys

import cv2
import numpy as np

NAMES = ['BK', 'LONG', 'MIDBig', 'MIDSmall', 'SMALLBig']
COLORS = {0: (60, 180, 75), 1: (255, 128, 0), 2: (0, 0, 255), 3: (0, 215, 255), 4: (255, 0, 255)}  # BGR

src, out = sys.argv[1], sys.argv[2]


def read_polys(label_path, w, h):
    objs = []
    for line in open(label_path):
        p = line.split()
        if len(p) < 7:
            continue
        cls = int(p[0])
        v = np.array(p[1:], dtype=np.float64).reshape(-1, 2)
        v[:, 0] *= w
        v[:, 1] *= h
        objs.append((cls, v.astype(np.float32)))
    return objs


def aabb_corners(poly):
    x0, y0 = poly.min(axis=0)
    x1, y1 = poly.max(axis=0)
    return np.array([[x0, y0], [x1, y0], [x1, y1], [x0, y1]], dtype=np.float32)


def obb_corners(poly):
    rect = cv2.minAreaRect(poly)
    return cv2.boxPoints(rect).astype(np.float32)


def iou(a, b):
    inter, _ = cv2.intersectConvexConvex(a, b)
    if inter <= 0:
        return 0.0
    ua = cv2.contourArea(a) + cv2.contourArea(b) - inter
    return float(inter / ua) if ua > 0 else 0.0


def overlap_pairs(boxes, thr):
    n, cnt = len(boxes), 0
    for i in range(n):
        for j in range(i + 1, n):
            if iou(boxes[i], boxes[j]) > thr:
                cnt += 1
    return cnt


def draw_poly(img, pts, color, th):
    cv2.polylines(img, [np.round(pts).astype(np.int32)], True, color, th, cv2.LINE_AA)


total = {'n': 0, 'aabb_fill': [], 'obb_fill': [], 'aabb_p3': 0, 'obb_p3': 0, 'aabb_p5': 0, 'obb_p5': 0}
per_image = []
os.makedirs(out, exist_ok=True)

for split in ['train', 'valid', 'test']:
    os.makedirs(f'{out}/dataset_obb/{split}/images', exist_ok=True)
    os.makedirs(f'{out}/dataset_obb/{split}/labels', exist_ok=True)
    for img_path in sorted(glob.glob(f'{src}/{split}/images/*')):
        stem = os.path.splitext(os.path.basename(img_path))[0]
        lbl_path = f'{src}/{split}/labels/{stem}.txt'
        img = cv2.imread(img_path)
        h, w = img.shape[:2]
        objs = read_polys(lbl_path, w, h)
        aabbs = [aabb_corners(p) for _, p in objs]
        obbs = [obb_corners(p) for _, p in objs]

        # write OBB label file + copy the image
        with open(f'{out}/dataset_obb/{split}/labels/{stem}.txt', 'w') as f:
            for (cls, _), q in zip(objs, obbs):
                q = q.copy()
                q[:, 0] = np.clip(q[:, 0] / w, 0, 1)
                q[:, 1] = np.clip(q[:, 1] / h, 0, 1)
                f.write(f'{cls} ' + ' '.join(f'{c:.6f}' for c in q.reshape(-1)) + '\n')
        shutil.copy(img_path, f'{out}/dataset_obb/{split}/images/')

        # statistics
        area_poly = [cv2.contourArea(p) for _, p in objs]
        a_fill = [pa / max(cv2.contourArea(b), 1) for pa, b in zip(area_poly, aabbs)]
        o_fill = [pa / max(cv2.contourArea(b), 1) for pa, b in zip(area_poly, obbs)]
        r = dict(split=split, stem=stem[:22], n=len(objs),
                 aabb_fill=float(np.mean(a_fill)), obb_fill=float(np.mean(o_fill)),
                 aabb_p3=overlap_pairs(aabbs, 0.3), obb_p3=overlap_pairs(obbs, 0.3),
                 aabb_p5=overlap_pairs(aabbs, 0.5), obb_p5=overlap_pairs(obbs, 0.5))
        per_image.append(r)

        # pictures for the first image of train and valid
        if stem in {os.path.splitext(os.path.basename(p))[0] for p in
                    [sorted(glob.glob(f'{src}/train/images/*'))[0], sorted(glob.glob(f'{src}/valid/images/*'))[0]]}:
            tag = f'{split}_{stem[:14]}'
            # overview: OBB only, coloured by class
            ov = img.copy()
            for (cls, _), q in zip(objs, obbs):
                draw_poly(ov, q, COLORS[cls], 3)
            ov = cv2.resize(ov, (1500, 1500), interpolation=cv2.INTER_AREA)
            for i, nm in enumerate(NAMES):
                cv2.rectangle(ov, (10, 10 + i * 34), (34, 34 + i * 34), COLORS[i], -1)
                cv2.putText(ov, nm, (44, 31 + i * 34), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2, cv2.LINE_AA)
            cv2.imwrite(f'{out}/obb_overview_{tag}.jpg', ov, [cv2.IMWRITE_JPEG_QUALITY, 92])

            # zoom: region where ordinary boxes overlap the most
            best, best_n = 0, -1
            for i in range(len(objs)):
                c = sum(1 for j in range(len(objs)) if j != i and iou(aabbs[i], aabbs[j]) > 0.2)
                if c > best_n:
                    best, best_n = i, c
            cx, cy = objs[best][1].mean(axis=0)
            half = 190
            x0 = int(min(max(cx - half, 0), w - 2 * half))
            y0 = int(min(max(cy - half, 0), h - 2 * half))
            crop = img[y0:y0 + 2 * half, x0:x0 + 2 * half].copy()
            S = 3
            crop = cv2.resize(crop, None, fx=S, fy=S, interpolation=cv2.INTER_CUBIC)
            left, right = crop.copy(), crop.copy()
            for (cls, poly), a, q in zip(objs, aabbs, obbs):
                sh = np.array([x0, y0], dtype=np.float32)
                draw_poly(left, (a - sh) * S, COLORS[cls], 2)           # ordinary box
                draw_poly(right, (q - sh) * S, COLORS[cls], 2)          # rotated box
                draw_poly(right, (poly - sh) * S, (255, 255, 255), 1)   # original outline
            cv2.putText(left, 'ordinary box', (12, 34), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 3, cv2.LINE_AA)
            cv2.putText(right, 'OBB (white = original polygon)', (12, 34), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 3, cv2.LINE_AA)
            cv2.imwrite(f'{out}/obb_zoom_{tag}.jpg', np.hstack([left, right]), [cv2.IMWRITE_JPEG_QUALITY, 92])

# report
print(f"{'split':6} {'image':22} {'objs':>4} | box fill: ordinary / OBB | pairs IoU>0.3: ordinary / OBB | pairs IoU>0.5: ordinary / OBB")
for r in per_image:
    print(f"{r['split']:6} {r['stem']:22} {r['n']:4d} | {r['aabb_fill']:.2f} / {r['obb_fill']:.2f}"
          f"{'':12} | {r['aabb_p3']:4d} / {r['obb_p3']:4d}{'':12} | {r['aabb_p5']:4d} / {r['obb_p5']:4d}")
n = sum(r['n'] for r in per_image)
print('TOTAL objects', n,
      '| mean fill ordinary %.2f vs OBB %.2f' % (np.average([r['aabb_fill'] for r in per_image], weights=[r['n'] for r in per_image]),
                                               np.average([r['obb_fill'] for r in per_image], weights=[r['n'] for r in per_image])),
      '| pairs>0.3: %d vs %d' % (sum(r['aabb_p3'] for r in per_image), sum(r['obb_p3'] for r in per_image)),
      '| pairs>0.5: %d vs %d' % (sum(r['aabb_p5'] for r in per_image), sum(r['obb_p5'] for r in per_image)))
