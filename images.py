"""Load the evaluation images listed in data/manifest.csv.

Each image keeps its stored integer values, shifted so that its minimum is zero;
nothing is rescaled. CT and MR slices are centre-cropped to a multiple of 32
pixels so that five decomposition levels are defined, and mammograms are cropped
to 1024 x 1024 pixels around the centroid of the pixels brighter than 10 percent
of the image maximum. The peak used for PSNR and SSIM is the maximum of the
shifted image, and the bit depth declared to the reference codecs is the
smallest that holds that maximum.
"""
import csv
import math
import os

import numpy as np
import pydicom

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, 'data')
LEVELS = 5
MG_SIZE = 1024


def crop_mult(a, m):
    H, W = a.shape
    h, w = (H // m) * m, (W // m) * m
    y0, x0 = (H - h) // 2, (W - w) // 2
    return a[y0:y0 + h, x0:x0 + w]


def crop_tissue(a, size):
    ys, xs = np.nonzero(a > 0.1 * a.max())
    cy, cx = (int(ys.mean()), int(xs.mean())) if ys.size else (a.shape[0] // 2, a.shape[1] // 2)
    h, w = min(size, a.shape[0]), min(size, a.shape[1])
    y0 = min(max(0, cy - h // 2), a.shape[0] - h)
    x0 = min(max(0, cx - w // 2), a.shape[1] - w)
    return a[y0:y0 + h, x0:x0 + w]


def manifest(tag=None):
    with open(os.path.join(DATA, 'manifest.csv'), newline='') as f:
        rows = list(csv.DictReader(f))
    return [r for r in rows if tag is None or r['tag'] == tag]


def load_row(r):
    path = os.path.join(DATA, 'dicom', r['tag'], r['sop_uid'] + '.dcm')
    a = pydicom.dcmread(path).pixel_array.astype(np.int64)
    a = a - a.min()
    if r['tag'] == 'MG':
        a = crop_tissue(a, MG_SIZE)
        a = a - a.min()
    a = crop_mult(a, 1 << LEVELS)
    peak = float(a.max())
    return dict(id=f"{r['tag']}-{r['patient']}-{r['slice_rank']}", tag=r['tag'],
                patient=r['patient'], split=r['split'], slice_rank=int(r['slice_rank']),
                sop_uid=r['sop_uid'], img=a.astype(np.float64), peak=peak,
                bits=max(2, int(math.ceil(math.log2(peak + 1)))),
                levels_used=int(np.unique(a).size), stored_bits=int(r['bits_stored']))


def load(tag, split=None):
    return [load_row(r) for r in manifest(tag) if split is None or r['split'] == split]
