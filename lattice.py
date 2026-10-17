"""Paraunitary rotation-lattice wavelet transform.

The analysis polyphase matrix is a cascade of plane rotations separated by unit
delays on the lower branch,

    E(z) = R(t_{N-1}) L(z) R(t_{N-2}) ... L(z) R(t_0),   L(z) = diag(1, z^-1),

with R(t) = [[cos t, sin t], [-sin t, cos t]]. Every angle vector gives an
orthonormal transform. N = 1 with t_0 = pi/4 is the orthonormal Haar stage, and
N = 2 with (t_0, t_1) = (pi/6, pi/12) is the four-tap Daubechies filter bank. The
low-pass filter has a zero at the Nyquist frequency exactly when the angles sum to
pi/4 modulo pi. Signals
are extended periodically, so the transform of a length-2M row is an M + M
split. The integer mode factors each rotation into three shears with rounding and
applies R(pi) = -I as a sign change, so it is exactly invertible for any angle
vector.
"""
import math
import numpy as np

HAAR = [math.pi / 4]
DAUB4 = [math.pi / 6, math.pi / 12]
# Lattice angles of the six- and eight-tap Daubechies scaling filters, obtained by
# least-squares fitting of filters() to the minimum-phase spectral factors; the
# last angle completes the sum pi/4.
DAUB6 = [0.391053192372, 0.499841268202, math.pi / 4 - 0.391053192372 - 0.499841268202]
DAUB8 = [0.311766063405, 0.681370944936, -0.253706537650,
         math.pi / 4 - 0.311766063405 - 0.681370944936 + 0.253706537650]


# ------------------------------------------------------------- 1-D lattice
def fwd1d(x, th):
    """Analysis along the last axis: returns concat([low, high], axis=-1)."""
    a, b = x[..., 0::2].copy(), x[..., 1::2].copy()
    for k, t in enumerate(th):
        c, s = math.cos(t), math.sin(t)
        a, b = c * a + s * b, -s * a + c * b
        if k < len(th) - 1:
            b = np.roll(b, 1, axis=-1)
    return np.concatenate([a, b], axis=-1)


def inv1d(y, th):
    n = y.shape[-1] // 2
    a, b = y[..., :n].copy(), y[..., n:].copy()
    for k in range(len(th) - 1, -1, -1):
        if k < len(th) - 1:
            b = np.roll(b, -1, axis=-1)
        c, s = math.cos(th[k]), math.sin(th[k])
        a, b = c * a - s * b, s * a + c * b
    out = np.empty(y.shape[:-1] + (2 * n,), dtype=y.dtype)
    out[..., 0::2], out[..., 1::2] = a, b
    return out


def daubechies(p):
    """Minimum-phase Daubechies scaling filter with p vanishing moments (2p taps),
    normalized to sum sqrt(2), by spectral factorization."""
    deg = p - 1
    Q = np.zeros(2 * deg + 1)
    base = np.array([-0.25, 0.5, -0.25])           # (2 - z - 1/z) / 4
    term = np.array([1.0])
    for k in range(p):
        t = np.zeros(2 * deg + 1)
        t[deg - k:deg - k + term.size] = math.comb(p - 1 + k, k) * term
        Q += t
        term = np.convolve(term, base)
    r = np.roots(Q[::-1])
    r = r[np.abs(r) < 1 - 1e-9]
    h = np.real(np.poly(np.concatenate([r, -np.ones(p)])))
    return h / h.sum() * math.sqrt(2)


def filters(th, L=64, tol=1e-12):
    """Analysis low-pass and high-pass impulse responses, read off the
    transform matrix so that both polyphase components are covered."""
    A = fwd1d(np.eye(L), th)          # row i = transform of delta_i
    n = L // 2
    m = n // 2
    lo = A[:, m][::-1]
    hi = A[:, n + m][::-1]

    def trim(v):
        nz = np.nonzero(np.abs(v) > tol)[0]
        return v[nz.min():nz.max() + 1]
    return trim(lo), trim(hi)


# ------------------------------------------------------------- 2-D, multilevel
def fwd2d(img, th, levels):
    """Separable multilevel analysis, rows then columns at each level."""
    out = np.array(img, dtype=np.float64)
    h, w = out.shape
    for _ in range(levels):
        sub = out[:h, :w]
        sub = fwd1d(sub, th)
        sub = fwd1d(sub.T, th).T
        out[:h, :w] = sub
        h, w = h // 2, w // 2
    return out


def inv2d(coef, th, levels):
    out = np.array(coef, dtype=np.float64)
    shapes = []
    h, w = out.shape
    for _ in range(levels):
        shapes.append((h, w))
        h, w = h // 2, w // 2
    for (h, w) in reversed(shapes):
        sub = out[:h, :w]
        sub = inv1d(sub.T, th).T
        sub = inv1d(sub, th)
        out[:h, :w] = sub
    return out


# ---------------------------------------------------- one vanishing moment
def constrained(free):
    """Angle vector (t_0, ..., t_{N-2}, pi/4 - sum) from N - 1 free angles.

    A constant input reaches the output of the cascade through R(sum of angles),
    so its detail coefficients are proportional to cos(sum) - sin(sum). They vanish,
    and the low-pass filter has a zero at the Nyquist frequency, exactly when the
    angles sum to pi/4 modulo pi. N = 1 is then the orthonormal Haar stage."""
    free = [float(t) for t in free]
    return free + [math.pi / 4 - sum(free)]


# ---------------------------------------------------- angle side information
ANGLE_BITS = 8


def quantise_angles(th, bits=ANGLE_BITS):
    """Uniform quantisation of each angle modulo pi on [-pi/2, pi/2).
    R(t + pi) = -R(t), so the reduction changes only the signs of the
    subband coefficients, which the quantiser and coder treat symmetrically."""
    step = math.pi / (1 << bits)
    out = []
    for t in th:
        u = (t + math.pi / 2) % math.pi - math.pi / 2
        k = int(math.floor(u / step + 0.5))
        k = min(max(k, -(1 << (bits - 1))), (1 << (bits - 1)) - 1)
        out.append(k * step)
    return out


# ---------------------------------------------------- reversible integer mode
def shear_pair(t):
    """R(t) = [[1,p],[0,1]] [[1,0],[q,1]] [[1,p],[0,1]] with p=tan(t/2), q=-sin(t)."""
    return math.tan(t / 2.0), -math.sin(t)


def _round(v):
    return np.floor(v + 0.5).astype(np.int64)


def _is_pi(t):
    """True when t equals pi modulo 2 pi, where tan(t / 2) has a pole and R(t) = -I."""
    return abs(math.remainder(t - math.pi, 2 * math.pi)) < 1e-12


def fwd1d_int(x, th):
    a = x[..., 0::2].astype(np.int64).copy()
    b = x[..., 1::2].astype(np.int64).copy()
    for k, t in enumerate(th):
        if _is_pi(t):
            a, b = -a, -b
        else:
            p, q = shear_pair(t)
            a = a + _round(p * b)
            b = b + _round(q * a)
            a = a + _round(p * b)
        if k < len(th) - 1:
            b = np.roll(b, 1, axis=-1)
    return np.concatenate([a, b], axis=-1)


def inv1d_int(y, th):
    n = y.shape[-1] // 2
    a, b = y[..., :n].astype(np.int64).copy(), y[..., n:].astype(np.int64).copy()
    for k in range(len(th) - 1, -1, -1):
        if k < len(th) - 1:
            b = np.roll(b, -1, axis=-1)
        if _is_pi(th[k]):
            a, b = -a, -b
            continue
        p, q = shear_pair(th[k])
        a = a - _round(p * b)
        b = b - _round(q * a)
        a = a - _round(p * b)
    out = np.empty(y.shape[:-1] + (2 * n,), dtype=np.int64)
    out[..., 0::2], out[..., 1::2] = a, b
    return out


def fwd2d_int(img, th, levels):
    out = np.array(img, dtype=np.int64)
    h, w = out.shape
    for _ in range(levels):
        sub = out[:h, :w]
        sub = fwd1d_int(sub, th)
        sub = fwd1d_int(sub.T, th).T
        out[:h, :w] = sub
        h, w = h // 2, w // 2
    return out


def inv2d_int(coef, th, levels):
    out = np.array(coef, dtype=np.int64)
    shapes = []
    h, w = out.shape
    for _ in range(levels):
        shapes.append((h, w))
        h, w = h // 2, w // 2
    for (h, w) in reversed(shapes):
        sub = out[:h, :w]
        sub = inv1d_int(sub.T, th).T
        sub = inv1d_int(sub, th)
        out[:h, :w] = sub
    return out
