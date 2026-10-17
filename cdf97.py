"""CDF 9/7 analysis and synthesis by lifting with whole-sample symmetric extension.

The transform is driven through the same quantiser and entropy coder as the
lattice, with per-subband steps divided by the synthesis-basis norms, so that the
contribution of the transform can be separated from that of the coding engine.
"""
import numpy as np

A = -1.586134342059924
B = -0.052980118572961
G = 0.882911075530934
D = 0.443506852043971
K = 1.230174104914001

def _next(a):
    """a[n + 1] with whole-sample symmetric extension of the full signal at the
    right edge: for an even-length signal x[0..2M-1], s[M] = x[2M] = x[2M-2] = s[M-1]."""
    return np.concatenate([a[..., 1:], a[..., -1:]], axis=-1)


def _prev(a):
    """a[n - 1] with whole-sample symmetric extension at the left edge:
    d[-1] = x[-1] = x[1] = d[0]."""
    return np.concatenate([a[..., :1], a[..., :-1]], axis=-1)


def fwd1d(x):
    s = x[..., 0::2].astype(np.float64).copy()
    d = x[..., 1::2].astype(np.float64).copy()
    d += A * (s + _next(s))
    s += B * (d + _prev(d))
    d += G * (s + _next(s))
    s += D * (d + _prev(d))
    return np.concatenate([s / K, d * K], axis=-1)


def inv1d(y):
    n = y.shape[-1] // 2
    s = y[..., :n] * K
    d = y[..., n:] / K
    s = s - D * (d + _prev(d))
    d = d - G * (s + _next(s))
    s = s - B * (d + _prev(d))
    d = d - A * (s + _next(s))
    out = np.empty(y.shape[:-1] + (2 * n,), dtype=np.float64)
    out[..., 0::2], out[..., 1::2] = s, d
    return out


def fwd2d(img, levels):
    out = np.array(img, dtype=np.float64)
    h, w = out.shape
    for _ in range(levels):
        sub = out[:h, :w]
        sub = fwd1d(sub)
        sub = fwd1d(sub.T).T
        out[:h, :w] = sub
        h, w = h // 2, w // 2
    return out

def inv2d(coef, levels):
    out = np.array(coef, dtype=np.float64)
    H, W = out.shape
    shapes = []
    h, w = H, W
    for _ in range(levels):
        shapes.append((h, w)); h, w = h // 2, w // 2
    for (h, w) in reversed(shapes):
        sub = out[:h, :w]
        sub = inv1d(sub.T).T
        sub = inv1d(sub)
        out[:h, :w] = sub
    return out

def band_slices(H, W, levels):
    out = []
    h, w = H, W
    for l in range(levels):
        hh, hw = h // 2, w // 2
        out.append(((slice(0, hh), slice(hw, w)), l, 'HL'))
        out.append(((slice(hh, h), slice(0, hw)), l, 'LH'))
        out.append(((slice(hh, h), slice(hw, w)), l, 'HH'))
        h, w = hh, hw
    out.append(((slice(0, h), slice(0, w)), levels, 'LL'))
    return out

def synthesis_norms(H, W, levels):
    """L2 norm of the synthesis basis function of each subband, used as the
    quantiser step weight so that transform-domain quantisation is MSE-fair."""
    norms = {}
    for sl, l, name in band_slices(H, W, levels):
        z = np.zeros((H, W))
        ys, xs = sl
        z[(ys.start + ys.stop) // 2, (xs.start + xs.stop) // 2] = 1.0
        r = inv2d(z, levels)
        norms[(l, name)] = float(np.sqrt((r ** 2).sum()))
    return norms
