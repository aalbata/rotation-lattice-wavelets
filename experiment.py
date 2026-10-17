"""Rate-distortion, reversible-mode and verification experiment for one modality.

    python experiment.py CT          (likewise MR and MG)

Every lossy transform is followed by the same uniform deadzone quantiser and the
same context-adaptive range coder, so differences between their rate-distortion
curves are attributable to the transform. JPEG 2000 (OpenJPEG) is the external
anchor. Lattice angles are trained on the middle slice of every training patient,
and every reported number comes from images of the disjoint test patients. The
output is results/<tag>.json.
"""
import json
import math
import multiprocessing as mp
import os
import platform
import sys
import time

import imagecodecs
import jpeg_ls
import numpy as np
from scipy.optimize import minimize
from skimage.metrics import structural_similarity

import cdf97
import coder
import haar
import images
import lattice as L
import legall53

LEVELS = images.LEVELS
STEP_DIVISORS = (0.35, 0.7, 1.4, 2.8, 5.6, 11.2, 22.4, 44.8, 89.6)  # step = std / divisor
MIN_STEP = 1.0               # steps finer than one stored level are not used
TRAIN_DIVISOR = 2.8          # training step Delta_0 = std / 2.8
LAMBDA = math.log(2) / 6     # cost MSE / Delta_0^2 + LAMBDA * rate; high-rate slope of D(R)
ORDERS = (1, 2, 3, 4)         # N = 1 is Haar; N - 1 angles are free under sum = pi/4
MAXITER = 200                # Nelder-Mead iterations per start
SIMPLEX_STEP = 0.05          # initial simplex offset from the start along each free angle, radians
HEADER_BYTES = 9             # image size (2 x 16 bit), bit depth (8 bit), quantiser step (32 bit)
J2K_TARGETS = tuple(range(20, 80, 4))   # dB, relative to the image peak
J2K_RESOLUTIONS = LEVELS + 1
RESULTS = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'results')


# ------------------------------------------------------------------ metrics
def psnr(img, rec, peak):
    mse = float(np.mean((img - rec) ** 2))
    return math.inf if mse == 0 else 10 * math.log10(peak * peak / mse)


def ssim(img, rec, peak):
    """SSIM with the settings of Wang et al. (2004): 11 x 11 Gaussian window with
    sigma 1.5, K1 = 0.01, K2 = 0.03, population covariances, and the image
    maximum as the dynamic range L."""
    return float(structural_similarity(img, rec, data_range=peak, gaussian_weights=True,
                                       sigma=1.5, use_sample_covariance=False))


def finish(rec, maxval):
    """Decoder output: rounded to the integer grid and clipped to the declared bit depth."""
    return np.clip(np.round(rec), 0, maxval)


# ------------------------------------------------------------------ own codec
def encode_bytes(q, check=None):
    n, stream = coder.code(q, LEVELS)
    if check is not None:
        H, W = q.shape
        check.append(bool(np.array_equal(coder.uncode(stream, W, H, LEVELS), q.astype(np.int32))))
    return n + HEADER_BYTES


class Transform:
    """Forward coefficients of one image, the inverse map and a per-coefficient
    step weight, so that a whole rate-distortion curve needs one forward transform."""

    def __init__(self, img, fwd, inv, weight=1.0):
        self.img, self.coef, self.inv, self.weight = img, fwd(img), inv, weight

    def point(self, step, maxval, check=None):
        s = step * self.weight
        q = coder.quantise(self.coef, s)
        nb = encode_bytes(q, check)
        return nb * 8 / self.img.size, finish(self.inv(coder.dequantise(q, s)), maxval)


def lattice_t(img, th):
    return Transform(img, lambda a: L.fwd2d(a, th, LEVELS), lambda c: L.inv2d(c, th, LEVELS))


def scaled_haar_t(img, alpha):
    return Transform(img, lambda a: haar.fwd2d(a, alpha, LEVELS), lambda c: haar.inv2d(c, alpha, LEVELS))


_NORMS = {}


def cdf97_t(img):
    """CDF 9/7 with each subband step divided by the l2 norm of its synthesis basis function."""
    if img.shape not in _NORMS:
        nrm = cdf97.synthesis_norms(img.shape[0], img.shape[1], LEVELS)
        w = np.ones(img.shape)
        for sl, lev, name in cdf97.band_slices(img.shape[0], img.shape[1], LEVELS):
            w[sl] = 1.0 / nrm[(lev, name)]
        _NORMS[img.shape] = w
    return Transform(img, lambda a: cdf97.fwd2d(a, LEVELS), lambda c: cdf97.inv2d(c, LEVELS), _NORMS[img.shape])


def run_lattice(img, th, step, maxval):
    return lattice_t(img, th).point(step, maxval)


# ------------------------------------------------------------------ reference codecs
def container(img, bits):
    """Sample array handed to the reference codecs and the bit depth declared to
    them: 8-bit samples for images of at most 8 bits, 16-bit samples otherwise."""
    if bits <= 8:
        return img.astype(np.uint8), 8
    return img.astype(np.uint16), bits


def strip_comment(b):
    """Remove the optional COM marker segment (0xFF64) that OpenJPEG writes into
    the main header; it carries no image data."""
    b = bytes(b)
    i = b.find(b'\xff\x64', 0, b.find(b'\xff\x90'))
    if i < 0:
        return b
    n = int.from_bytes(b[i + 2:i + 4], 'big')
    return b[:i] + b[i + 2 + n:]


def j2k_curve(x):
    img, peak = x['img'], x['peak']
    a, bits = container(img, x['bits'])
    offset = 20 * math.log10(((1 << bits) - 1) / peak)
    pts, header = [], []
    for target in J2K_TARGETS:
        b = strip_comment(imagecodecs.jpeg2k_encode(a, level=target + offset, codecformat='J2K',
                                                    reversible=False, resolutions=J2K_RESOLUTIONS,
                                                    bitspersample=bits))
        rec = imagecodecs.jpeg2k_decode(b).astype(np.float64)
        p = psnr(img, rec, peak)
        if math.isinf(p):
            continue
        pts.append([len(b) * 8 / img.size, p, ssim(img, rec, peak)])
        header.append(b.find(b'\xff\x90'))             # bytes before the first SOT marker
    pts.sort()
    return pts, int(np.median(header)) if header else None


def lossless(x, th3):
    img = x['img'].astype(np.int64)
    H, W = img.shape
    bits = x['bits']
    out, exact = {}, []
    for key, fwd, inv in (
            ('lattice1', lambda a: L.fwd2d_int(a, L.HAAR, LEVELS), lambda c: L.inv2d_int(c, L.HAAR, LEVELS)),
            ('lattice3', lambda a: L.fwd2d_int(a, th3, LEVELS), lambda c: L.inv2d_int(c, th3, LEVELS)),
            ('legall53', lambda a: legall53.fwd2d(a, LEVELS), lambda c: legall53.inv2d(c, LEVELS))):
        c = fwd(img)
        exact.append(bool(np.array_equal(inv(c), img)))
        n, s = coder.code_k(c, LEVELS)
        exact.append(bool(np.array_equal(coder.uncode_k(s, W, H, LEVELS), c.astype(np.int32))))
        out[key] = (n + HEADER_BYTES) * 8 / img.size
    u, cbits = container(img, bits)
    b = jpeg_ls.encode_buffer(u.tobytes(), H, W, 1, bits)
    r = np.frombuffer(bytes(jpeg_ls.decode_buffer(b)[0]), u.dtype).reshape(H, W)
    exact.append(bool(np.array_equal(r, u)))
    out['jpegls'] = len(b) * 8 / img.size
    b = imagecodecs.jpegxl_encode(u, lossless=True, bitspersample=cbits)
    exact.append(bool(np.array_equal(imagecodecs.jpegxl_decode(b), u)))
    out['jpegxl'] = len(b) * 8 / img.size
    b = strip_comment(imagecodecs.jpeg2k_encode(u, codecformat='J2K', reversible=True,
                                                resolutions=J2K_RESOLUTIONS, bitspersample=cbits))
    exact.append(bool(np.array_equal(imagecodecs.jpeg2k_decode(b), u)))
    out['jpeg2000'] = len(b) * 8 / img.size
    return out, all(exact), cbits


# ------------------------------------------------------------------ angle training
def lagrangian(img, th, step, maxval):
    b, rec = run_lattice(img, th, step, maxval)
    return float(np.mean((img - rec) ** 2)) / step ** 2 + LAMBDA * b


_TRAIN = []


def _init_train(ids, tag):
    global _TRAIN
    rows = {f"{r['tag']}-{r['patient']}-{r['slice_rank']}": r for r in images.manifest(tag)}
    _TRAIN = [images.load_row(rows[i]) for i in ids]


def _train_term(args):
    k, th = args
    x = _TRAIN[k]
    return lagrangian(x['img'], th, x['img'].std() / TRAIN_DIVISOR, (1 << x['bits']) - 1)


def simplex(x0):
    x0 = np.asarray(x0, float)
    return np.vstack([x0] + [x0 + SIMPLEX_STEP * e for e in np.eye(x0.size)])


def nelder_mead(cost, x0):
    r = minimize(cost, np.asarray(x0, float), method='Nelder-Mead',
                 options=dict(maxiter=MAXITER, xatol=1e-3, fatol=1e-6,
                              initial_simplex=simplex(x0)))
    return [float(v) for v in r.x], float(r.fun), int(r.nfev), int(r.nit), bool(r.success)


DAUB = {2: L.DAUB4, 3: L.DAUB6, 4: L.DAUB8}


def starts(N, trained):
    """Free angles (t_0, ..., t_{N-2}) of three starting points: Haar embedded at
    order N, the Daubechies member of order N, and the order N - 1 solution
    extended by a zero angle. Each lies on the constraint sum = pi/4."""
    return [([math.pi / 4] + [0.0] * (N - 1))[:N - 1], DAUB[N][:N - 1], list(trained[N - 1])]


def train(tag, pool, ids):
    n = len(ids)
    trained, log = {1: L.constrained([])}, {1: []}
    for N in ORDERS[1:]:
        def cost(free):
            th = L.constrained(free)
            return float(np.mean(pool.map(_train_term, [(k, th) for k in range(n)])))
        runs = []
        for s in starts(N, trained):
            free, J, nfev, nit, ok = nelder_mead(cost, s)
            runs.append(dict(start=L.constrained(s), start_cost=cost(s), angles=L.constrained(free),
                             cost=J, nfev=nfev, nit=nit, converged=ok))
        best = min(runs, key=lambda r: r['cost'])
        trained[N] = best['angles']
        log[N] = runs
        print(f'  {tag} N={N}: {np.round(best["angles"], 4).tolist()} cost {best["cost"]:.5f}', flush=True)
    return trained, log


# ------------------------------------------------------------------ test images
def evaluate(args):
    row, trained = args
    x = images.load_row(row)
    img, peak = x['img'], x['peak']
    maxval = (1 << x['bits']) - 1
    steps = [s for s in (img.std() / k for k in STEP_DIVISORS) if s >= MIN_STEP]
    check = []

    free, J_img, nfev, nit, ok = nelder_mead(
        lambda t: lagrangian(img, L.constrained(t), img.std() / TRAIN_DIVISOR, maxval), trained[3][:2])
    th_img = L.constrained(free)
    th_img_q = L.constrained(L.quantise_angles(free))
    side = 2 * L.ANGLE_BITS / img.size

    methods = {
        'Haar': lambda: lattice_t(img, L.HAAR),
        'ScalarGain': lambda: scaled_haar_t(img, 1.0),
        'Daub4': lambda: lattice_t(img, L.DAUB4),
        'Daub6': lambda: lattice_t(img, L.DAUB6),
        'Daub8': lambda: lattice_t(img, L.DAUB8),
        'Lat2': lambda: lattice_t(img, trained[2]),
        'Lat3': lambda: lattice_t(img, trained[3]),
        'Lat4': lambda: lattice_t(img, trained[4]),
        'Lat3img': lambda: lattice_t(img, th_img_q),
        'CDF97': lambda: cdf97_t(img),
    }
    curves = {}
    for name, make in methods.items():
        t = make()
        pts = []
        for s in steps:
            b, rec = t.point(s, maxval, check)
            if name == 'Lat3img':
                b += side
            p = psnr(img, rec, peak)
            if not math.isinf(p):
                pts.append([b, p, ssim(img, rec, peak)])
        curves[name] = sorted(pts)
    curves['JPEG2000'], j2k_header = j2k_curve(x)

    # Proposition 1: a global gain g, here the square root of the number of pixels,
    # with step g * Delta gives the same integers and bits.
    g = math.sqrt(img.size)
    c_h = L.fwd2d(img, L.HAAR, LEVELS)
    # Corollary 1: per-stage gain 1 with level-dependent steps 2^j Delta equals orthonormal Haar.
    c_sg = haar.fwd2d(img, 1.0, LEVELS)
    w = haar.level_gain(img.shape, 1.0, LEVELS)
    prop1, cor1 = [], []
    for s in steps:
        q = coder.quantise(c_h, s)
        q_g = coder.quantise(g * c_h, g * s)
        q_c = coder.quantise(c_sg, s * w)
        same = np.array_equal(q, q_g) and coder.code(q, LEVELS)[1].tobytes() == coder.code(q_g, LEVELS)[1].tobytes()
        prop1.append(bool(same))
        cor1.append(bool(np.array_equal(q, q_c)))

    ll, ll_exact, cbits = lossless(x, trained[3])
    return dict(id=x['id'], patient=x['patient'], slice_rank=x['slice_rank'], sop_uid=x['sop_uid'],
                shape=list(img.shape), peak=peak, bits=x['bits'], levels_used=x['levels_used'],
                stored_bits=x['stored_bits'], std=float(img.std()), steps=steps, curves=curves,
                j2k_header_bytes=j2k_header,
                adaptive=dict(angles=th_img, angles_quantised=th_img_q, cost=J_img, nfev=nfev,
                              nit=nit, converged=ok),
                lossless=ll, lossless_exact=ll_exact, declared_bits=cbits,
                decoder_checks=len(check), decoder_exact=int(sum(check)),
                prop1_cases=len(prop1), prop1_identical=int(sum(prop1)),
                cor1_cases=len(cor1), cor1_identical=int(sum(cor1)))


def environment():
    import importlib.metadata as md
    import scipy
    import skimage
    import re
    import _CharLS
    v = imagecodecs.version()
    libs = {k: v.split(f'{k}-')[1].split(',')[0].strip() for k in ('openjpeg', 'libjxl') if f'{k}-' in v}
    with open(_CharLS.__file__, 'rb') as fh:
        m = re.search(rb'charls (\d+\.\d+\.\d+)', fh.read())
    libs['charls_pyjpegls'] = m.group(1).decode() if m else 'unknown'
    return dict(python=platform.python_version(), platform=platform.platform(),
                processor=platform.processor(), numpy=np.__version__, scipy=scipy.__version__,
                scikit_image=skimage.__version__, imagecodecs=imagecodecs.__version__,
                pyjpegls=md.version('pyjpegls'), pydicom=md.version('pydicom'), **libs)


def main(tag, workers):
    rows = images.manifest(tag)
    test = [r for r in rows if r['split'] == 'test']
    seen, train_ids = set(), []
    middle = 1 if tag in ('CT', 'MR') else 0
    for r in rows:
        if r['split'] == 'train' and int(r['slice_rank']) == middle and r['patient'] not in seen:
            seen.add(r['patient'])
            train_ids.append(f"{r['tag']}-{r['patient']}-{r['slice_rank']}")
    t0 = time.time()
    print(f'{tag}: {len(rows)} images, {len(test)} test, training on {len(train_ids)}', flush=True)
    with mp.Pool(min(workers, len(train_ids)), initializer=_init_train, initargs=(train_ids, tag)) as pool:
        trained, train_log = train(tag, pool, train_ids)
    t1 = time.time()
    out = []
    with mp.Pool(workers) as pool:
        for k, res in enumerate(pool.imap(evaluate, [(r, trained) for r in test])):
            out.append(res)
            print(f'  [{k + 1}/{len(test)}] {res["id"]}', flush=True)
    t2 = time.time()
    config = dict(levels=LEVELS, step_divisors=STEP_DIVISORS, min_step=MIN_STEP,
                  train_divisor=TRAIN_DIVISOR, lam=LAMBDA, orders=ORDERS, train_images=train_ids,
                  maxiter=MAXITER, simplex_step=SIMPLEX_STEP, header_bytes=HEADER_BYTES,
                  angle_bits=L.ANGLE_BITS, quantiser_offset=coder.DELTA_OFFSET,
                  ssim='Gaussian window, sigma 1.5, population covariances',
                  j2k_targets=J2K_TARGETS, j2k_resolutions=J2K_RESOLUTIONS, workers=workers)
    res = dict(tag=tag, config=config, environment=environment(),
               trained={str(k): v for k, v in trained.items()},
               training_runs={str(k): v for k, v in train_log.items()},
               runtime_s=dict(training=round(t1 - t0, 1), evaluation=round(t2 - t1, 1)),
               images=out)
    os.makedirs(RESULTS, exist_ok=True)
    with open(os.path.join(RESULTS, f'{tag}.json'), 'w', newline='\n') as f:
        json.dump(res, f, indent=1)
    print(f'{tag}: written results/{tag}.json ({t2 - t0:.0f} s)', flush=True)


if __name__ == '__main__':
    tag = sys.argv[1]
    workers = int(os.environ.get('WORKERS', max(1, min(12, (os.cpu_count() or 2) - 2))))
    main(tag, workers)
