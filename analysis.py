"""Aggregate the per-image rate-distortion curves stored in results/<tag>.json.

BD-rate follows Bjontegaard: log10 rate is fitted by least squares as a cubic
polynomial of quality (of lower degree where fewer than four points fall in the
fitting range) and the two fits are integrated over the quality interval common to
both curves.
Only the points inside that interval, plus one neighbour on each side, enter each
fit, so that points far outside the comparison (for example header-dominated
JPEG 2000 streams at very low quality) do not shape it. SSIM is used on the
decibel scale -10 log10(1 - SSIM). Uncertainty is a 95 percent percentile
bootstrap interval that resamples patients, and the paired test is a two-sided
Wilcoxon signed-rank test on patient means with Holm correction. This module
reads only the results files and needs neither the images nor the codecs.
"""
import json
import math
import os

import numpy as np
from scipy.interpolate import PchipInterpolator
from scipy.stats import wilcoxon

HERE = os.path.dirname(os.path.abspath(__file__))
RESULTS = os.path.join(HERE, 'results')
TAGS = ('CT', 'MR', 'MG')
BOOT = 10000
SEED = 20261016
MIN_SPAN = {1: 3.0, 2: 3.0}      # minimum common quality span in dB (PSNR, SSIM in dB)


def load(tag):
    with open(os.path.join(RESULTS, f'{tag}.json')) as f:
        return json.load(f)


def load_all():
    return {t: load(t) for t in TAGS if os.path.exists(os.path.join(RESULTS, f'{t}.json'))}


def load_window(tag):
    """Points spanning the compression-ratio window (window_experiment.py)."""
    with open(os.path.join(RESULTS, f'{tag}_window.json')) as f:
        return json.load(f)


def per_image_window(dw, method, ref='Haar', fit='cubic'):
    out = []
    for im in dw['images']:
        v = bd_rate(im['curves'][ref], im['curves'][method], 1, fit, tuple(im['window_bpp']))
        out.append((im['patient'], v))
    return out


def ssim_db(s):
    return -10 * math.log10(max(1e-12, 1 - s))


def _points(curve, metric):
    r = np.array([p[0] for p in curve], float)
    q = np.array([p[1] if metric == 1 else ssim_db(p[2]) for p in curve], float)
    o = np.argsort(q)
    r, q = r[o], q[o]
    keep = np.concatenate([[True], np.diff(q) > 1e-9])
    return np.log10(r[keep]), q[keep]


def _select(lr, q, lo, hi):
    inside = np.nonzero((q >= lo) & (q <= hi))[0]
    if inside.size == 0:
        return lr[:0], q[:0]
    a, b = max(0, inside[0] - 1), min(q.size, inside[-1] + 2)
    return lr[a:b], q[a:b]


def _integral(lr, q, lo, hi, method):
    if method == 'pchip':
        f = PchipInterpolator(q, lr, extrapolate=True)
        return float(f.integrate(lo, hi))
    deg = min(3, q.size - 1)
    p = np.polyint(np.polyfit(q, lr, deg))
    return float(np.polyval(p, hi) - np.polyval(p, lo))


def quality_at_rate(curve, rate, metric=1):
    """Quality of a curve at a given rate, by interpolation in log rate."""
    pts = sorted(curve)
    lr = np.log10([p[0] for p in pts])
    q = np.array([p[1] if metric == 1 else ssim_db(p[2]) for p in pts])
    x = math.log10(rate)
    if x < lr[0] or x > lr[-1]:
        return float('nan')
    return float(np.interp(x, lr, q))


def bd_rate(ref, test, metric=1, method='cubic', window=None):
    """Average rate difference of `test` against `ref`, in percent; negative is a saving.
    `window` = (rate_lo, rate_hi) restricts the quality interval to the qualities the
    reference curve reaches between those rates."""
    lr1, q1 = _points(ref, metric)
    lr2, q2 = _points(test, metric)
    if q1.size < 4 or q2.size < 4:
        return float('nan')
    lo, hi = max(q1[0], q2[0]), min(q1[-1], q2[-1])
    if window is not None:
        qa, qb = quality_at_rate(ref, window[0], metric), quality_at_rate(ref, window[1], metric)
        if math.isnan(qa) or math.isnan(qb):
            return float('nan')
        lo, hi = max(lo, qa), min(hi, qb)
        span = 0.5
    else:
        span = MIN_SPAN[metric]
    if hi - lo < span:
        return float('nan')
    a1, b1 = _select(lr1, q1, lo, hi)
    a2, b2 = _select(lr2, q2, lo, hi)
    if b1.size < 2 or b2.size < 2:
        return float('nan')
    d = (_integral(a2, b2, lo, hi, method) - _integral(a1, b1, lo, hi, method)) / (hi - lo)
    return float((10 ** d - 1) * 100)


def per_image(d, method, ref='Haar', metric=1, fit='cubic', window=None):
    out = []
    for im in d['images']:
        w = window(im) if callable(window) else window
        v = bd_rate(im['curves'][ref], im['curves'][method], metric, fit, w)
        out.append((im['patient'], v))
    return out


def summarise(pairs, seed=SEED):
    """Mean over images with a patient-level bootstrap interval, the Wilcoxon test on
    patient means, and the number of patients whose mean is a saving."""
    pairs = [(p, v) for p, v in pairs if not math.isnan(v)]
    pats = sorted({p for p, _ in pairs})
    by = {p: np.array([v for q, v in pairs if q == p]) for p in pats}
    vals = np.array([v for _, v in pairs])
    rng = np.random.default_rng(seed)
    boots = np.empty(BOOT)
    for b in range(BOOT):
        pick = rng.integers(len(pats), size=len(pats))
        boots[b] = np.concatenate([by[pats[i]] for i in pick]).mean()
    pm = np.array([by[p].mean() for p in pats])
    p_value = float(wilcoxon(pm).pvalue) if np.any(pm != 0) else 1.0
    return dict(mean=float(vals.mean()), sd=float(vals.std(ddof=1)), lo=float(np.percentile(boots, 2.5)),
                hi=float(np.percentile(boots, 97.5)), n_images=int(vals.size), n_patients=len(pats),
                improved=int(np.sum(pm < 0)), worse_images=int(np.sum(vals > 0)), p=p_value)


def holm(pvalues):
    """Holm step-down adjusted p-values for a dict of name -> p."""
    names = sorted(pvalues, key=lambda k: pvalues[k])
    m = len(names)
    out, run = {}, 0.0
    for i, k in enumerate(names):
        run = max(run, min(1.0, (m - i) * pvalues[k]))
        out[k] = run
    return out


def mean_curve(images, method, metric, grid):
    ys = []
    for im in images:
        pts = sorted(im['curves'][method])
        lr = np.log10([p[0] for p in pts])
        q = np.array([p[1] if metric == 1 else p[2] for p in pts])
        ys.append(np.interp(np.log10(grid), lr, q, left=np.nan, right=np.nan))
    Y = np.array(ys)
    with np.errstate(invalid='ignore'):
        m = np.nanmean(Y, axis=0)
    m[np.sum(~np.isnan(Y), axis=0) < 0.5 * len(ys)] = np.nan
    return m


def lossless_means(d):
    keys = d['images'][0]['lossless'].keys()
    return {k: float(np.mean([im['lossless'][k] for im in d['images']])) for k in keys}
