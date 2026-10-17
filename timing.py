"""Forward and inverse transform times and entropy-coder times on one CT test slice.

Each figure is the median of 15 repetitions after one warm-up run, measured in a
single process. The lattice is timed in the direct rotation form used by
experiment.py. Writes results/timing.json.
"""
import json
import os
import platform
import time

import numpy as np

import analysis as A
import cdf97
import coder
import images
import lattice as L

HERE = os.path.dirname(os.path.abspath(__file__))
REPEATS = 15
LEVELS = images.LEVELS


def clock(fn, arg):
    fn(arg)
    t = []
    for _ in range(REPEATS):
        t0 = time.perf_counter()
        out = fn(arg)
        t.append(time.perf_counter() - t0)
    return 1e3 * float(np.median(t)), out


def main():
    ct = A.load('CT')
    row = [r for r in images.manifest('CT') if r['split'] == 'test'][0]
    img = images.load_row(row)['img']
    th = {'Haar': L.HAAR, 'Daub4': L.DAUB4, 'Lat3': ct['trained']['3'], 'Lat4': ct['trained']['4']}
    res = {}
    for k, t in th.items():
        f, c = clock(lambda a: L.fwd2d(a, t, LEVELS), img)
        i, _ = clock(lambda a: L.inv2d(a, t, LEVELS), c)
        res[k] = dict(forward_ms=f, inverse_ms=i, N=len(t))
    f, c = clock(lambda a: cdf97.fwd2d(a, LEVELS), img)
    i, _ = clock(lambda a: cdf97.inv2d(a, LEVELS), c)
    res['CDF97'] = dict(forward_ms=f, inverse_ms=i)
    q = coder.quantise(L.fwd2d(img, th['Lat3'], LEVELS), img.std() / 2.8).astype(np.int32)
    e, (n, s) = clock(lambda a: coder.code(a, LEVELS), q)
    d, q2 = clock(lambda b: coder.uncode(b, img.shape[1], img.shape[0], LEVELS), s)
    assert np.array_equal(q, q2)
    res['coder'] = dict(encode_ms=e, decode_ms=d, bpp=n * 8 / img.size)
    res['image'] = dict(id=f"{row['tag']}-{row['patient']}-{row['slice_rank']}", shape=list(img.shape))
    res['machine'] = dict(platform=platform.platform(), processor=platform.processor(),
                          python=platform.python_version(), repeats=REPEATS)
    with open(os.path.join(HERE, 'results', 'timing.json'), 'w', newline='\n') as fh:
        json.dump(res, fh, indent=1)
    print(json.dumps(res, indent=1))


if __name__ == '__main__':
    main()
