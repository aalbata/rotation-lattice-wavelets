"""Reconstructions for the visual comparison and the example images of the figures.

The MR example is the test image ranked len // 2 by its N = 3 lattice BD-rate
against Haar (one of the two middle images of the modality); the CT example for the
scalar-gain figure is ranked len // 2 by its per-stage unit-gain BD-rate. Haar and
the N = 3 lattice are run at the smallest quantiser step whose rate does not exceed
the target (the largest rate not above it), and JPEG 2000 at the largest codestream
that does not exceed it. The target is a compression ratio of 16:1 relative to the
occupied bit depth. Writes
results/visual.npz, results/visual.json and results/figure_choices.json.
"""
import json
import math
import os

import imagecodecs
import numpy as np

import analysis as A
import experiment as E
import images
import lattice as L

HERE = os.path.dirname(os.path.abspath(__file__))
RATIO = 16
CROP = 128


def median_image(d, method):
    v = [(x, im) for (p, x), im in zip(A.per_image(d, method), d['images'])]
    v.sort(key=lambda t: t[0])
    return v[len(v) // 2][1]


def at_rate(transform, target, maxval):
    lo, hi = 1e-3, 1e6
    best = None
    for _ in range(60):
        mid = math.sqrt(lo * hi)
        b, rec = transform.point(mid, maxval)
        if b <= target:
            best = (b, rec)
            hi = mid
        else:
            lo = mid
    return best


def j2k_at_rate(x, target):
    a, bits = E.container(x['img'], x['bits'])
    offset = 20 * math.log10(((1 << bits) - 1) / x['peak'])
    lo, hi = 0.0, 120.0
    best = None
    for _ in range(40):
        mid = 0.5 * (lo + hi)
        b = E.strip_comment(imagecodecs.jpeg2k_encode(a, level=mid + offset, codecformat='J2K',
                                                      reversible=False, resolutions=E.J2K_RESOLUTIONS,
                                                      bitspersample=bits))
        r = len(b) * 8 / x['img'].size
        if r <= target:
            best = (r, imagecodecs.jpeg2k_decode(b).astype(np.float64))
            lo = mid
        else:
            hi = mid
    return best


def main():
    data = A.load_all()
    ct = median_image(data['CT'], 'ScalarGain')
    mr = median_image(data['MR'], 'Lat3')
    rows = {f"{r['tag']}-{r['patient']}-{r['slice_rank']}": r for r in images.manifest('MR')}
    x = images.load_row(rows[mr['id']])
    img, peak = x['img'], x['peak']
    maxval = (1 << x['bits']) - 1
    target = x['bits'] / RATIO
    th3 = data['MR']['trained']['3']
    out = {'Haar': at_rate(E.lattice_t(img, L.HAAR), target, maxval),
           'Lat3': at_rate(E.lattice_t(img, th3), target, maxval),
           'JPEG2000': j2k_at_rate(x, target)}
    ys, xs = np.nonzero(img > 0.25 * img.max())
    H, W = img.shape
    cy = int(np.clip(ys.mean(), CROP // 2, H - CROP // 2))
    cx = int(np.clip(xs.mean(), CROP // 2, W - CROP // 2))
    sl = (slice(cy - CROP // 2, cy + CROP // 2), slice(cx - CROP // 2, cx + CROP // 2))
    arrays = {'original': img[sl].astype(np.uint16)}
    info = dict(image=mr['id'], sop_uid=mr['sop_uid'], ratio=RATIO, target_bpp=target,
                crop=[cy - CROP // 2, cx - CROP // 2, CROP], order=['Haar', 'Lat3', 'JPEG2000'],
                labels={'Haar': 'Orthonormal Haar', 'Lat3': 'Lattice $N=3$', 'JPEG2000': 'JPEG 2000'},
                bpp={}, psnr={})
    for k, (b, rec) in out.items():
        arrays[k] = rec[sl].astype(np.uint16)
        info['bpp'][k] = b
        info['psnr'][k] = E.psnr(img, rec, peak)
    info['error_scale'] = float(np.percentile(np.abs(img[sl] - out['Haar'][1][sl]), 99.5))
    np.savez_compressed(os.path.join(HERE, 'results', 'visual.npz'), **arrays)
    with open(os.path.join(HERE, 'results', 'visual.json'), 'w', newline='\n') as f:
        json.dump(info, f, indent=1)
    with open(os.path.join(HERE, 'results', 'figure_choices.json'), 'w', newline='\n') as f:
        json.dump({'scalar_example': ct['id'], 'visual_example': mr['id']}, f, indent=1)
    print(json.dumps(info, indent=1))


if __name__ == '__main__':
    main()
