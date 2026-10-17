"""Lossless rate of JPEG-LS after the distinct values of each mammogram are
renumbered consecutively, with the table of distinct values counted in the rate.

The renumbering is invertible given the table, so the result is a lossless rate
for the original image. It separates the effect of the sparse value histogram of
the digitized-film mammograms from the coding method. Writes
results/histogram_check.json.
"""
import json
import math
import os

import jpeg_ls
import numpy as np

import images

HERE = os.path.dirname(os.path.abspath(__file__))


def jpegls_bpp(a, bits):
    u = a.astype(np.uint8 if bits <= 8 else np.uint16)
    b = jpeg_ls.encode_buffer(u.tobytes(), a.shape[0], a.shape[1], 1, bits)
    r = np.frombuffer(bytes(jpeg_ls.decode_buffer(b)[0]), u.dtype).reshape(a.shape)
    assert np.array_equal(r, u)
    return len(b) * 8 / a.size


def main():
    out = []
    for r in images.manifest('MG'):
        if r['split'] != 'test':
            continue
        x = images.load_row(r)
        a = x['img'].astype(np.int64)
        values, ranks = np.unique(a, return_inverse=True)
        ranks = ranks.reshape(a.shape)
        rbits = max(2, int(math.ceil(math.log2(values.size))))
        table_bits = values.size * x['bits']
        out.append(dict(id=x['id'], distinct=int(values.size), bits=x['bits'],
                        jpegls=jpegls_bpp(a, x['bits']),
                        jpegls_renumbered=jpegls_bpp(ranks, rbits) + table_bits / a.size))
    res = dict(images=out,
               mean_jpegls=float(np.mean([o['jpegls'] for o in out])),
               mean_jpegls_renumbered=float(np.mean([o['jpegls_renumbered'] for o in out])))
    with open(os.path.join(HERE, 'results', 'histogram_check.json'), 'w', newline='\n') as f:
        json.dump(res, f, indent=1)
    print(res['mean_jpegls'], res['mean_jpegls_renumbered'])


if __name__ == '__main__':
    main()
