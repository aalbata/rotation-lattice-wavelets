"""Write every table of the paper (LaTeX, booktabs) and every reported number
(out/numbers.json) from results/*.json. Nothing in the paper is typed by hand.

Compression-ratio windows span the lowest and the highest of the national maximum
ratios listed as medically acceptable by the European Society of Radiology (2011,
Table 1) for chest CT, MR and mammography, applied to the bit depth occupied by the
image maximum.
"""
import json
import math
import os

import numpy as np

import analysis as A
import images

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, 'out')
WINDOWS = {'CT': (5, 15), 'MR': (5, 24), 'MG': (15, 25)}   # acceptable compression ratios
ROWS = ['ScalarGain', 'Daub4', 'Daub6', 'Daub8', 'Lat2', 'Lat3', 'Lat4', 'Lat3img', 'CDF97', 'JPEG2000']
WROWS = ['ScalarGain', 'Daub4', 'Daub6', 'Daub8', 'Lat3', 'Lat4', 'CDF97', 'JPEG2000']
PAIRS = [('Lat2', 'Daub4'), ('Lat3', 'Daub6'), ('Lat4', 'Daub8'), ('Lat3img', 'Lat3'), ('Lat4', 'Lat3'),
         ('Lat3', 'Lat2'), ('CDF97', 'Lat3'), ('CDF97', 'Lat4'), ('JPEG2000', 'CDF97')]
NAME = {'ScalarGain': 'Per-stage unit gain, Eq.~\\eqref{eq:mod}',
        'Daub4': 'D4, untrained ($N=2$)', 'Daub6': 'D6, untrained ($N=3$)', 'Daub8': 'D8, untrained ($N=4$)',
        'Lat2': 'Lattice $N=2$, trained', 'Lat3': 'Lattice $N=3$, trained', 'Lat4': 'Lattice $N=4$, trained',
        'Lat3img': 'Lattice $N=3$, per image', 'CDF97': 'CDF 9/7, same coder',
        'JPEG2000': 'JPEG 2000 (OpenJPEG)'}
MODNAME = {'CT': ('Chest CT', 'LIDC-IDRI'), 'MR': ('Prostate MR', 'PROSTATE-DIAGNOSIS'),
           'MG': ('Film mammography', 'CBIS-DDSM')}


def f(x, n=1):
    return '--' if x is None or (isinstance(x, float) and math.isnan(x)) else f'{x:.{n}f}'


def m(x, n=1):
    """Number in math mode with a proper minus sign."""
    s = f(x, n)
    return s if s == '--' else f'${s}$'


def write(name, lines):
    os.makedirs(os.path.join(OUT, 'tables'), exist_ok=True)
    with open(os.path.join(OUT, 'tables', name + '.tex'), 'w', newline='\n', encoding='utf-8') as fh:
        fh.write('\n'.join(lines) + '\n')


def bd_block(data, rows, metric, fit='cubic'):
    res = {}
    for t, d in data.items():
        res[t] = {}
        for r in rows:
            res[t][r] = A.summarise(A.per_image(d, r, metric=metric, fit=fit))
    return res


def variants(d):
    """Results of one modality with two derived curves per image: the per-image
    lattice without its angle side information, and JPEG 2000 with its main header
    replaced by the header length of our coder."""
    out = dict(d)
    out['images'] = []
    for im in d['images']:
        npix = im['shape'][0] * im['shape'][1]
        side = 2 * d['config']['angle_bits'] / npix
        hdr = 8 * (im['j2k_header_bytes'] - d['config']['header_bytes']) / npix
        c = dict(im['curves'])
        c['Lat3imgNoSide'] = [[p[0] - side] + p[1:] for p in im['curves']['Lat3img']]
        c['JPEG2000Hdr'] = [[p[0] - hdr] + p[1:] for p in im['curves']['JPEG2000']]
        out['images'].append(dict(im, curves=c))
    return out


def bd_table(name, res, rows, tags, caption, label, header_extra=None):
    L = [r'\begin{table}', r'\tbl{' + caption + '}{%', r'\footnotesize', r'\setlength{\tabcolsep}{4pt}',
         r'\begin{tabular}{l' + 'c' * len(tags) + '}', r'\toprule']
    hdr = [t if header_extra is None else f'{t} ({header_extra[t]})' for t in tags]
    L.append('Transform & ' + ' & '.join(hdr) + r' \\')
    L.append(r'\midrule')
    for r in rows:
        cells = []
        for t in tags:
            s = res[t][r]
            cells.append(f"{m(s['mean'])} [{m(s['lo'])}, {m(s['hi'])}]")
        L.append(NAME[r] + ' & ' + ' & '.join(cells) + r' \\')
    L += [r'\bottomrule', r'\end{tabular}}', r'\label{' + label + '}', r'\end{table}']
    write(name, L)


def main():
    data = A.load_all()
    tags = [t for t in A.TAGS if t in data]
    N = {}

    # ---------------------------------------------------------------- Table 1
    L = [r'\begin{table}', r'\tbl{Test material. Images are stored integer values shifted to a zero minimum and '
         r'cropped to a multiple of 32 pixels (mammograms to $1024\times1024$ around the breast). Bit depth is the '
         r'number of bits occupied by the image maximum, given as the range over the test images; distinct levels is the '
         r'median number of distinct values per test image. Training and test patients are disjoint. '
         r'The angles are trained on the middle slice of each training patient (@@TRAINUSED@@ images).}{%',
         r'\footnotesize', r'\setlength{\tabcolsep}{3.5pt}', r'\begin{tabular}{llcccccc}', r'\toprule',
         r'Modality & Collection & \multicolumn{2}{c}{Patients} & \multicolumn{2}{c}{Images} & Bit depth & Distinct \\',
         r'\cmidrule(lr){3-4}\cmidrule(lr){5-6}',
         r' & & train & test & train & test & (bits) & levels \\', r'\midrule']
    for t in tags:
        d = data[t]
        test = d['images']
        man = images.manifest(t)
        n_train_img = sum(r['split'] == 'train' for r in man)
        n_train_pat = len({r['patient'] for r in man if r['split'] == 'train'})
        n_test_pat = len({im['patient'] for im in test})
        bits = [im['bits'] for im in test]
        lv = float(np.median([im['levels_used'] for im in test]))
        shapes = sorted({tuple(im['shape']) for im in test})
        N[f'nTestImg{t}'] = len(test)
        N[f'nTestPat{t}'] = n_test_pat
        N[f'nTrainPat{t}'] = n_train_pat
        N[f'nTrainImg{t}'] = n_train_img
        assert n_train_pat == len(d['config']['train_images'])
        N[f'bitsMin{t}'] = min(bits)
        N[f'bitsMax{t}'] = max(bits)
        N[f'levelsMedian{t}'] = lv
        N[f'nTrainUsed{t}'] = len(d['config']['train_images'])
        N[f'shapes{t}'] = [list(s) for s in shapes]
        L.append(f"{MODNAME[t][0]} & {MODNAME[t][1]} & {n_train_pat} & {n_test_pat} & "
                 f"{n_train_img} & {len(test)} & {min(bits)}--{max(bits)} & {lv:,.1f} \\\\")
    L += [r'\bottomrule', r'\end{tabular}}', r'\label{tab:data}', r'\end{table}']
    used = [str(N[f'nTrainUsed{t}']) for t in tags]
    L[1] = L[1].replace('@@TRAINUSED@@', ', '.join(used[:-1]) + ' and ' + used[-1])
    write('tab_data', L)

    # ---------------------------------------------------------------- BD tables
    psnr = bd_block(data, ROWS, 1)
    ssim = bd_block(data, ROWS, 2)
    wdata = {t: A.load_window(t) for t in tags}
    win = {t: {r: A.summarise(A.per_image_window(wdata[t], r)) for r in WROWS} for t in tags}
    pchip = bd_block(data, ROWS, 1, fit='pchip')
    bd_table('tab_bd_psnr', psnr, ROWS, tags,
             r'BD-rate against orthonormal Haar on PSNR, in percent, as the mean over the test images of each modality '
             r'with a 95\% bootstrap interval that resamples patients. Negative values are rate savings at equal '
             r'PSNR. Every row except the last uses the same quantizer and entropy coder. CT, chest computed '
             r'tomography; MR, prostate magnetic resonance imaging; MG, film mammography; BD-rate, '
             r'Bj{\o}ntegaard delta rate; PSNR, peak signal-to-noise ratio; $N$, number of lattice stages; '
             r'D4, D6 and D8, the four-, six- and eight-tap Daubechies members of the lattice; CDF, '
             r'Cohen--Daubechies--Feauveau.', 'tab:bdpsnr')
    bd_table('tab_bd_ssim', ssim, ROWS, tags,
             r'BD-rate against orthonormal Haar on SSIM expressed in decibels, $-10\log_{10}(1-\mathrm{SSIM})$, in '
             r'percent, as the mean over the test images with a 95\% bootstrap interval that resamples patients. Negative '
             r'values are rate savings at equal SSIM. SSIM, structural similarity index; other abbreviations '
             r'as in Table~\ref{tab:bdpsnr}.', 'tab:bdssim')
    hdr = {t: f'{WINDOWS[t][0]}:1--{WINDOWS[t][1]}:1' for t in tags}
    bd_table('tab_bd_window', win, WROWS, tags,
             r'BD-rate against orthonormal Haar on PSNR restricted to the quality range that Haar reaches between '
             r'the lowest and the highest national maximum compression ratio of each modality (in parentheses; '
             r'ratios relative to the occupied bit depth), computed from eight separate points per image and '
             r'transform that span the rate range of the window widened by a factor of 1.4 at each end, in '
             r'percent, with 95\% patient bootstrap intervals. Abbreviations as in Table~\ref{tab:bdpsnr}.',
             'tab:bdwindow', hdr)

    # ---------------------------------------------------------------- direct comparisons
    direct = {}
    for t in tags:
        dv = variants(data[t])
        direct[f'psnr:{t}:Lat3imgNoSide-vs-Lat3'] = A.summarise(A.per_image(dv, 'Lat3imgNoSide', ref='Lat3'))
        direct[f'psnr:{t}:JPEG2000Hdr-vs-CDF97'] = A.summarise(A.per_image(dv, 'JPEG2000Hdr', ref='CDF97'))
        for a, b in PAIRS:
            direct[f'psnr:{t}:{a}-vs-{b}'] = A.summarise(A.per_image(data[t], a, ref=b))
            if a in wdata[t]['images'][0]['curves'] and b in wdata[t]['images'][0]['curves']:
                direct[f'window:{t}:{a}-vs-{b}'] = A.summarise(A.per_image_window(wdata[t], a, ref=b))
            direct[f'ssim:{t}:{a}-vs-{b}'] = A.summarise(A.per_image(data[t], a, ref=b, metric=2))
    N['direct'] = direct

    # ---------------------------------------------------------------- statistics
    pv = {}
    for metric, res in (('psnr', psnr), ('ssim', ssim)):
        for t in tags:
            for r in ROWS:
                pv[f'{metric}:{t}:{r}'] = res[t][r]['p']
    adj = A.holm(pv)
    stats = {}
    for metric, res in (('psnr', psnr), ('ssim', ssim), ('window', win), ('pchip', pchip)):
        for t in tags:
            for r in res[t]:
                s = dict(res[t][r])
                key = f'{metric}:{t}:{r}'
                if key in adj:
                    s['p_holm'] = adj[key]
                stats[key] = s
    N['stats'] = stats
    N['holmFamily'] = len(pv)
    N['holmNotSignificant'] = sorted(k for k, v in adj.items() if v >= 0.05)
    N['holmMaxSignificant'] = max([v for v in adj.values() if v < 0.05], default=None)
    # sensitivity of the PSNR BD-rates to the interpolation: monotone piecewise-cubic against the cubic fit
    dif = {f'{t}:{r}': pchip[t][r]['mean'] - psnr[t][r]['mean'] for t in tags for r in ROWS}
    N['pchipMaxDiff'] = dict(coder=max(abs(v) for k, v in dif.items() if not k.endswith('JPEG2000')),
                             jpeg2000=max(abs(v) for k, v in dif.items() if k.endswith('JPEG2000')))

    # ---------------------------------------------------------------- angles
    L = [r'\begin{table}', r'\tbl{Lattice angles in radians from \eqref{eq:lagr} on the training patients of each '
         r'modality, and the untrained Daubechies members of the same lattice. The last column gives the angle '
         r'sum, which equals $\pi/4=0.7854$ when the low-pass filter vanishes at $\omega=\pi$.}{%',
         r'\footnotesize', r'\setlength{\tabcolsep}{3.5pt}', r'\begin{tabular}{lllll}', r'\toprule',
         r'Modality & $N=2$ & $N=3$ & $N=4$ & Sum ($N=3$) \\', r'\midrule']
    import lattice as LT
    lo4, _ = LT.filters(LT.DAUB4, 64)
    exact4 = np.array([1 + 3 ** 0.5, 3 + 3 ** 0.5, 3 - 3 ** 0.5, 1 - 3 ** 0.5]) / (4 * 2 ** 0.5)
    N['daubResidual'] = float(np.abs(lo4 - exact4).sum())
    fit = []
    for p, th in ((3, LT.DAUB6), (4, LT.DAUB8)):
        lo, _ = LT.filters(th, 64)
        fit.append(float(np.abs(lo - LT.daubechies(p)).sum()))
    N['daubFit'] = fit
    for t in tags:
        tr = data[t]['trained']
        cells = ['$(' + ',\\,'.join(f'{v:.3f}' for v in tr[n]) + ')$' for n in ('2', '3', '4')]
        L.append(f"{t} & " + ' & '.join(cells) + f" & {sum(tr['3']):.4f} \\\\")
        N[f'angles{t}'] = tr
        N[f'angleSum{t}'] = {n: sum(v) for n, v in tr.items()}
    L.append(r'\midrule')
    L.append('Daubechies & $(' + ',\\,'.join(f'{v:.3f}' for v in LT.DAUB4) + ')$ & $(' +
             ',\\,'.join(f'{v:.3f}' for v in LT.DAUB6) + ')$ & $(' + ',\\,'.join(f'{v:.3f}' for v in LT.DAUB8) +
             f')$ & {sum(LT.DAUB6):.4f} \\\\')
    L += [r'\bottomrule', r'\end{tabular}}', r'\label{tab:angles}', r'\end{table}']
    write('tab_angles', L)

    # ---------------------------------------------------------------- lossless
    keys = [('lattice1', 'Reversible lattice $N=1$ (Haar), our coder'), ('lattice3', 'Reversible lattice $N=3$, our coder'),
            ('legall53', 'Reversible LeGall 5/3, our coder'), ('jpeg2000', 'JPEG 2000 reversible 5/3 (OpenJPEG)'),
            ('jpegls', 'JPEG-LS (CharLS)'), ('jpegxl', 'JPEG XL (libjxl)')]
    L = [r'\begin{table}', r'\tbl{Lossless rate in bit per pixel, mean over the test images of each modality, at '
         r'the stored precision. The first three rows share the reversible-mode entropy coder. Abbreviations as '
         r'in Table~\ref{tab:bdpsnr}.}{%',
         r'\footnotesize', r'\begin{tabular}{l' + 'c' * len(tags) + '}', r'\toprule',
         'Method & ' + ' & '.join(tags) + r' \\', r'\midrule']
    ll = {t: A.lossless_means(data[t]) for t in tags}
    for k, nm in keys:
        L.append(nm + ' & ' + ' & '.join(f(ll[t][k], 2) for t in tags) + r' \\')
    L += [r'\bottomrule', r'\end{tabular}}', r'\label{tab:lossless}', r'\end{table}']
    write('tab_lossless', L)
    N['lossless'] = ll
    N['losslessAllExact'] = all(im['lossless_exact'] for t in tags for im in data[t]['images'])
    N['losslessLat3Worse'] = {t: int(sum(im['lossless']['lattice3'] > im['lossless']['lattice1']
                                         for im in data[t]['images'])) for t in tags}

    # ---------------------------------------------------------------- verification counts
    for k in ('decoder_checks', 'decoder_exact', 'prop1_cases', 'prop1_identical', 'cor1_cases', 'cor1_identical'):
        N[k] = int(sum(im[k] for t in tags for im in data[t]['images']))
    N['j2kHeaderMedian'] = {t: float(np.median([im['j2k_header_bytes'] for im in data[t]['images']])) for t in tags}
    N['environment'] = data[tags[0]]['environment']
    N['runtime'] = {t: data[t]['runtime_s'] for t in tags}
    N['trainingRuns'] = {t: data[t]['training_runs'] for t in tags}
    N['adaptiveConverged'] = {t: int(sum(im['adaptive']['converged'] for im in data[t]['images'])) for t in tags}

    # ---------------------------------------------------------------- timing
    tp = os.path.join(A.RESULTS, 'timing.json')
    if os.path.exists(tp):
        with open(tp) as fh:
            tm = json.load(fh)
        # multiplications per sample per one-dimensional pass: an N-stage lattice needs four per
        # sample pair and stage in the direct form and three in the shear form; CDF 9/7 needs one
        # per sample pair for each of its four lifting steps and two for the scaling constants
        names = {'Haar': 'Lattice $N=1$ (Haar)', 'Daub4': 'Lattice $N=2$ (D4)', 'Lat3': 'Lattice $N=3$',
                 'Lat4': 'Lattice $N=4$'}
        rows = [(k, names[k], 2 * tm[k]['N'], 1.5 * tm[k]['N']) for k in ('Haar', 'Daub4', 'Lat3', 'Lat4')]
        rows.append(('CDF97', 'CDF 9/7 (lifting)', (4 + 2) / 2, (4 + 2) / 2))
        L = [r'\begin{table}', r'\tbl{Transform and entropy-coder times for one $512\times512$ CT slice with five '
             r'decomposition levels (median of 15 runs, single process, NumPy transforms and the compiled coder), '
             r'and multiplications per sample per one-dimensional pass as timed here (direct rotation form for '
             r'the lattice) and in the three-shear form of \eqref{eq:shear}. The coder times are for the '
             r'coefficients of the trained $N=3$ lattice of the same slice quantized at the training step '
             r'$\Delta_x$ of \eqref{eq:lagr} (' + f"{tm['coder']['bpp']:.2f}" + r' bit per pixel).}{%',
             r'\footnotesize',
             r'\begin{tabular}{lcccc}', r'\toprule',
             r'Stage & Forward (ms) & Inverse (ms) & Mult., as timed & Mult., shear form \\', r'\midrule']
        for k, nm, a, b in rows:
            L.append(f"{nm} & {tm[k]['forward_ms']:.1f} & {tm[k]['inverse_ms']:.1f} & {a:g} & {b:g} \\\\")
        L.append(r'\midrule')
        L.append(f"Entropy coder & {tm['coder']['encode_ms']:.1f} & {tm['coder']['decode_ms']:.1f} & -- & -- \\\\")
        L += [r'\bottomrule', r'\end{tabular}}', r'\label{tab:time}', r'\end{table}']
        write('tab_time', L)
        N['timing'] = tm

    for p in ('visual.json', 'figure_choices.json', 'histogram_check.json'):
        fp = os.path.join(A.RESULTS, p)
        if os.path.exists(fp):
            with open(fp) as fh:
                N[p.split('.')[0]] = json.load(fh)
    with open(os.path.join(OUT, 'numbers.json'), 'w', newline='\n') as fh:
        json.dump(N, fh, indent=1, sort_keys=True)
    readme_results(tags, psnr, ssim, win, ll)
    print('tables and numbers written to', OUT)


PLAIN = {'ScalarGain': 'Per-stage unit gain', 'Daub4': 'D4, untrained (N=2)', 'Daub6': 'D6, untrained (N=3)',
         'Daub8': 'D8, untrained (N=4)', 'Lat2': 'Lattice N=2, trained', 'Lat3': 'Lattice N=3, trained',
         'Lat4': 'Lattice N=4, trained', 'Lat3img': 'Lattice N=3, per image', 'CDF97': 'CDF 9/7, same coder',
         'JPEG2000': 'JPEG 2000 (OpenJPEG)'}


def readme_results(tags, psnr, ssim, win, ll):
    """Write the results tables of README.md between its results markers."""
    def cell(s):
        return f"{s['mean']:.1f} [{s['lo']:.1f}, {s['hi']:.1f}]"
    L = ['BD-rate against orthonormal Haar in percent (negative is a saving), mean over the test images with a '
         '95 % patient bootstrap interval. Window columns restrict the comparison to the range between the '
         'lowest and the highest national maximum compression ratio of each modality.', '',
         '| Transform | ' + ' | '.join(f'{t} PSNR' for t in tags) + ' | ' + ' | '.join(f'{t} SSIM' for t in tags) +
         ' | ' + ' | '.join(f'{t} window' for t in tags) + ' |',
         '|---' * (1 + 3 * len(tags)) + '|']
    for r in ROWS:
        row = [PLAIN[r]] + [cell(psnr[t][r]) for t in tags] + [cell(ssim[t][r]) for t in tags]
        row += [cell(win[t][r]) if r in win[t] else '' for t in tags]
        L.append('| ' + ' | '.join(row) + ' |')
    L += ['', 'Lossless rate in bit per pixel, mean over the test images:', '',
          '| Method | ' + ' | '.join(tags) + ' |', '|---' * (1 + len(tags)) + '|']
    names = [('lattice1', 'Reversible lattice N=1 (Haar)'), ('lattice3', 'Reversible lattice N=3'),
             ('legall53', 'Reversible LeGall 5/3, same coder'), ('jpeg2000', 'JPEG 2000 reversible'),
             ('jpegls', 'JPEG-LS'), ('jpegxl', 'JPEG XL')]
    for k, nm in names:
        L.append(f'| {nm} | ' + ' | '.join(f'{ll[t][k]:.2f}' for t in tags) + ' |')
    path = os.path.join(HERE, 'README.md')
    with open(path, encoding='utf-8') as fh:
        text = fh.read()
    a, b = '<!-- results:start -->', '<!-- results:end -->'
    assert text.count(a) == 1 and text.count(b) == 1
    head, rest = text.split(a)
    _, tail = rest.split(b)
    with open(path, 'w', encoding='utf-8', newline='\n') as fh:
        fh.write(head + a + '\n' + '\n'.join(L) + '\n' + b + tail)


if __name__ == '__main__':
    main()
