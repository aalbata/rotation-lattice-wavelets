"""Haar transform with an arbitrary scalar gain per one-dimensional stage.

alpha = 1/sqrt(2) is the orthonormal Haar transform. alpha = 1 gives the
unnormalised Haar functions, which is the per-stage gain of the prefactor
sin(2 pi / log2 2) + cos(2 pi / log2 2) = 1 (1 - 2^-52 in double precision).
"""
import math

import numpy as np


def fwd2d(img, alpha, levels):
    out = np.array(img, dtype=np.float64)
    h, w = out.shape
    for _ in range(levels):
        sub = out[:h, :w]
        for _axis in range(2):
            e, o = sub[:, 0::2], sub[:, 1::2]
            sub = np.concatenate([alpha * (e + o), alpha * (o - e)], 1).T
        out[:h, :w] = sub
        h, w = h // 2, w // 2
    return out


def inv2d(coef, alpha, levels):
    out = np.array(coef, dtype=np.float64)
    shapes = []
    h, w = out.shape
    for _ in range(levels):
        shapes.append((h, w))
        h, w = h // 2, w // 2
    for (h, w) in reversed(shapes):
        sub = out[:h, :w]
        for _axis in range(2):
            sub = sub.T
            n = sub.shape[1] // 2
            lo, d = sub[:, :n], sub[:, n:]
            e, o = (lo - d) / (2 * alpha), (lo + d) / (2 * alpha)
            z = np.empty_like(sub)
            z[:, 0::2], z[:, 1::2] = e, o
            sub = z
        out[:h, :w] = sub
    return out


def level_map(shape, levels):
    """Decomposition level j of every coefficient in the Mallat layout:
    j for the detail subbands of level j (finest is 1), and J for LL."""
    H, W = shape
    m = np.full(shape, levels, dtype=np.int64)
    h, w = H, W
    for j in range(1, levels + 1):
        m[:h, w // 2:w] = j
        m[h // 2:h, :w] = j
        h, w = h // 2, w // 2
    m[:h, :w] = levels
    return m


def level_gain(shape, alpha, levels):
    """Factor (alpha sqrt 2)^(2j) carried by each coefficient of the per-stage
    gain transform relative to the orthonormal transform (Corollary 1)."""
    return (alpha * math.sqrt(2)) ** (2 * level_map(shape, levels))
