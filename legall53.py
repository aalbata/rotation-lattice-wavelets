"""Reversible LeGall 5/3 integer wavelet transform of JPEG 2000 Part 1, by lifting
with whole-sample symmetric extension, in the same Mallat layout as the lattice."""
import numpy as np


def _next(a):
    return np.concatenate([a[..., 1:], a[..., -1:]], axis=-1)


def _prev(a):
    return np.concatenate([a[..., :1], a[..., :-1]], axis=-1)


def fwd1d(x):
    s = x[..., 0::2].astype(np.int64).copy()
    d = x[..., 1::2].astype(np.int64).copy()
    d -= (s + _next(s)) >> 1
    s += (_prev(d) + d + 2) >> 2
    return np.concatenate([s, d], axis=-1)


def inv1d(y):
    n = y.shape[-1] // 2
    s = y[..., :n].astype(np.int64).copy()
    d = y[..., n:].astype(np.int64).copy()
    s -= (_prev(d) + d + 2) >> 2
    d += (s + _next(s)) >> 1
    out = np.empty(y.shape[:-1] + (2 * n,), dtype=np.int64)
    out[..., 0::2], out[..., 1::2] = s, d
    return out


def fwd2d(img, levels):
    out = np.array(img, dtype=np.int64)
    h, w = out.shape
    for _ in range(levels):
        sub = out[:h, :w]
        sub = fwd1d(sub)
        sub = fwd1d(sub.T).T
        out[:h, :w] = sub
        h, w = h // 2, w // 2
    return out


def inv2d(coef, levels):
    out = np.array(coef, dtype=np.int64)
    shapes = []
    h, w = out.shape
    for _ in range(levels):
        shapes.append((h, w))
        h, w = h // 2, w // 2
    for (h, w) in reversed(shapes):
        sub = out[:h, :w]
        sub = inv1d(sub.T).T
        sub = inv1d(sub)
        out[:h, :w] = sub
    return out
