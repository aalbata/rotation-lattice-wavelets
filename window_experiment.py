"""Rate-distortion points spanning the range of national maximum compression
ratios of each modality, widened by a factor of 1.4 in rate at each end.

    python window_experiment.py CT          (likewise MR and MG)

For every test image and transform, the quantizer step (or, for JPEG 2000, the
PSNR target) whose rate is 1.4 times the upper end of the rate window and the one
whose rate is the lower end divided by 1.4 are found by bisection, and eight points
are coded from one to the other, both included, at geometrically spaced steps (for
JPEG 2000, linearly spaced PSNR targets). The lattice angles are the ones trained
in experiment.py. Writes results/<tag>_window.json.
"""
import json
import math
import multiprocessing as mp
import os
import sys

import imagecodecs
import numpy as np

import experiment as E
import haar
import images
import lattice as L

WINDOWS = {'CT': (5, 15), 'MR': (5, 24), 'MG': (15, 25)}   # national maximum ratios (ESR 2011)
MARGIN = 1.4
POINTS = 8
METHODS = ('Haar', 'ScalarGain', 'Daub4', 'Daub6', 'Daub8', 'Lat3', 'Lat4', 'CDF97')


def bisect(rate_of, target, lo, hi, increasing):
    """Parameter in [lo, hi] (searched on a log scale) whose rate is closest to target."""
    for _ in range(22):
        mid = math.sqrt(lo * hi)
        r = rate_of(mid)
        if (r > target) == increasing:
            hi = mid
        else:
            lo = mid
    return math.sqrt(lo * hi)


def evaluate(args):
    row, trained, tag = args
    x = images.load_row(row)
    img, peak = x['img'], x['peak']
    maxval = (1 << x['bits']) - 1
    r_lo, r_hi = x['bits'] / WINDOWS[tag][1], x['bits'] / WINDOWS[tag][0]
    make = {'Haar': lambda: E.lattice_t(img, L.HAAR), 'ScalarGain': lambda: E.scaled_haar_t(img, 1.0),
            'Daub4': lambda: E.lattice_t(img, L.DAUB4), 'Daub6': lambda: E.lattice_t(img, L.DAUB6),
            'Daub8': lambda: E.lattice_t(img, L.DAUB8), 'Lat3': lambda: E.lattice_t(img, trained['3']),
            'Lat4': lambda: E.lattice_t(img, trained['4']), 'CDF97': lambda: E.cdf97_t(img)}
    curves = {}
    for name in METHODS:
        t = make[name]()
        rate = lambda s: t.point(s, maxval)[0]
        s_fine = bisect(rate, r_hi * MARGIN, 1e-3, 1e7, increasing=False)
        s_coarse = bisect(rate, r_lo / MARGIN, 1e-3, 1e7, increasing=False)
        pts = []
        for s in np.geomspace(s_fine, s_coarse, POINTS):
            b, rec = t.point(float(s), maxval)
            p = E.psnr(img, rec, peak)
            if not math.isinf(p):
                pts.append([b, p])
        curves[name] = sorted(pts)
    a, bits = E.container(img, x['bits'])
    offset = 20 * math.log10(((1 << bits) - 1) / peak)

    def j2k(target):
        b = E.strip_comment(imagecodecs.jpeg2k_encode(a, level=target + offset, codecformat='J2K',
                                                      reversible=False, resolutions=E.J2K_RESOLUTIONS,
                                                      bitspersample=bits))
        return b
    rate_j = lambda q: len(j2k(q)) * 8 / img.size
    q_hi = bisect(rate_j, r_hi * MARGIN, 5.0, 120.0, increasing=True)
    q_lo = bisect(rate_j, r_lo / MARGIN, 5.0, 120.0, increasing=True)
    pts = []
    for q in np.linspace(q_lo, q_hi, POINTS):
        b = j2k(float(q))
        rec = imagecodecs.jpeg2k_decode(b).astype(np.float64)
        p = E.psnr(img, rec, peak)
        if not math.isinf(p):
            pts.append([len(b) * 8 / img.size, p])
    curves['JPEG2000'] = sorted(pts)
    return dict(id=x['id'], patient=x['patient'], bits=x['bits'], window_bpp=[r_lo, r_hi], curves=curves)


def main(tag, workers):
    with open(os.path.join(E.RESULTS, f'{tag}.json')) as f:
        trained = json.load(f)['trained']
    test = [r for r in images.manifest(tag) if r['split'] == 'test']
    out = []
    with mp.Pool(workers) as pool:
        for k, res in enumerate(pool.imap(evaluate, [(r, trained, tag) for r in test])):
            out.append(res)
            print(f'  [{k + 1}/{len(test)}] {res["id"]}', flush=True)
    res = dict(tag=tag, ratios=WINDOWS[tag], margin=MARGIN, points=POINTS, images=out)
    with open(os.path.join(E.RESULTS, f'{tag}_window.json'), 'w', newline='\n') as f:
        json.dump(res, f, indent=1)
    print(f'{tag}: written results/{tag}_window.json', flush=True)


if __name__ == '__main__':
    tag = sys.argv[1]
    workers = int(os.environ.get('WORKERS', max(1, min(12, (os.cpu_count() or 2) - 2))))
    main(tag, workers)
