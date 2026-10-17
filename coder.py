"""Uniform deadzone quantiser and ctypes bindings to the context-adaptive
binary range coder in codec/codec.c (built by build_codec.py)."""
import ctypes
import os

import numpy as np

from build_codec import library_path

_path = library_path()
if not os.path.exists(_path):
    raise ImportError(f'{_path} not found; run "python build_codec.py" first')
_lib = ctypes.CDLL(_path)
_I32 = np.ctypeslib.ndpointer(np.int32, flags='C_CONTIGUOUS')
_U8 = np.ctypeslib.ndpointer(np.uint8, flags='C_CONTIGUOUS')
for _name, _res in (('encode', ctypes.c_int64), ('decode', ctypes.c_int),
                    ('encode_k', ctypes.c_int64), ('decode_k', ctypes.c_int)):
    _f = getattr(_lib, _name)
    _f.restype = _res
    _f.argtypes = [_I32, ctypes.c_int, ctypes.c_int, ctypes.c_int, _U8, ctypes.c_int64]

# Reconstruction offset of the deadzone quantiser: c_hat = sgn(q) (|q| + DELTA_OFFSET) step.
DELTA_OFFSET = 0.5


def quantise(coef, step):
    """Q(c) = sgn(c) floor(|c| / step)."""
    return np.sign(coef) * np.floor(np.abs(coef) / step)


def dequantise(q, step):
    return np.where(q == 0, 0.0, np.sign(q) * (np.abs(q) + DELTA_OFFSET) * step)


def _run(enc, q, levels):
    q = np.ascontiguousarray(q.astype(np.int32))
    H, W = q.shape
    cap = max(1 << 16, H * W * 8)
    buf = np.zeros(cap, np.uint8)
    if not 1 <= levels <= 16:
        raise ValueError('levels must be between 1 and 16')
    n = int(enc(q.ravel(), W, H, levels, buf, cap))
    if n < 0:
        raise RuntimeError('bitstream exceeds the output buffer')
    return n, buf[:n].copy()


def _unrun(dec, stream, W, H, levels):
    q = np.zeros(H * W, np.int32)
    dec(q, W, H, levels, np.ascontiguousarray(stream), stream.size)
    return q.reshape(H, W)


def code(q, levels):
    """Lossy-mode coder: order-0 Exp-Golomb tail. Returns (bytes, stream)."""
    return _run(_lib.encode, q, levels)


def uncode(stream, W, H, levels):
    return _unrun(_lib.decode, stream, W, H, levels)


def code_k(q, levels):
    """Reversible-mode coder: the neighbourhood magnitude sum is divided by four
    before bucketing, and the Exp-Golomb order of the tail is the bucket index
    minus two (at least zero), which suits unquantised integer coefficients."""
    return _run(_lib.encode_k, q, levels)


def uncode_k(stream, W, H, levels):
    return _unrun(_lib.decode_k, stream, W, H, levels)
