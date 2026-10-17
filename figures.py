"""Draw every figure of the paper from results/*.json and results/visual.npz.

Figures are written to out/figures as PDF (vector), EPS (vector) and TIFF
(600 dpi). Widths match the 34 pc text width of the Taylor & Francis Interact
layout; text is set in Computer Modern at 7 pt.
"""
import json
import logging
import math
import os

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch
import numpy as np

import analysis as A
import lattice as L

logging.getLogger('fontTools').setLevel(logging.ERROR)

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, 'out', 'figures')
WIDTH = 34 / 6.0225          # 34 pc in inches
PT = 7

plt.rcParams.update({
    'font.family': 'serif', 'font.serif': ['cmr10'], 'mathtext.fontset': 'cm',
    'axes.formatter.use_mathtext': True, 'axes.unicode_minus': False,
    'font.size': PT, 'axes.labelsize': PT, 'axes.titlesize': PT, 'legend.fontsize': PT - 0.5,
    'xtick.labelsize': PT - 0.5, 'ytick.labelsize': PT - 0.5,
    'axes.linewidth': 0.5, 'lines.linewidth': 0.9, 'lines.markersize': 2.6,
    'xtick.major.width': 0.5, 'ytick.major.width': 0.5, 'xtick.minor.width': 0.4, 'ytick.minor.width': 0.4,
    'xtick.direction': 'in', 'ytick.direction': 'in', 'axes.grid': True,
    'grid.linewidth': 0.3, 'grid.color': '0.85', 'savefig.dpi': 600, 'pdf.fonttype': 42, 'ps.fonttype': 42,
})

STYLE = {
    'Haar':       dict(color='#000000', marker='o', ls='-',  label='Orthonormal Haar'),
    'ScalarGain': dict(color='#c0392b', marker='v', ls=':',  label='Per-stage unit gain'),
    'Daub4':      dict(color='#7f8c8d', marker='s', ls='--', label='D4 (untrained)'),
    'Daub6':      dict(color='#95a5a6', marker='h', ls='--', label='D6 (untrained)'),
    'Daub8':      dict(color='#bdc3c7', marker='8', ls='--', label='D8 (untrained)'),
    'Lat2':       dict(color='#5dade2', marker='<', ls='-.', label='Lattice $N=2$'),
    'Lat3':       dict(color='#1f4e79', marker='D', ls='-',  label='Lattice $N=3$'),
    'Lat4':       dict(color='#2874a6', marker='>', ls='-.', label='Lattice $N=4$'),
    'Lat3img':    dict(color='#7d3c98', marker='^', ls='-',  label='Lattice $N=3$, per image'),
    'CDF97':      dict(color='#148f77', marker='P', ls='--', label='CDF 9/7 (same coder)'),
    'JPEG2000':   dict(color='#b9770e', marker='X', ls='-',  label='JPEG 2000 (OpenJPEG)'),
}
SHORT = {'ScalarGain': 'Unit gain', 'Daub4': 'D4', 'Daub6': 'D6', 'Daub8': 'D8', 'Lat2': '$N=2$',
         'Lat3': '$N=3$', 'Lat4': '$N=4$', 'Lat3img': '$N=3$ img.', 'CDF97': 'CDF 9/7',
         'JPEG2000': 'JPEG 2000'}
TITLE = {'CT': 'CT (LIDC-IDRI)', 'MR': 'MR (PROSTATE-DIAGNOSIS)', 'MG': 'MG (CBIS-DDSM)'}


def save(fig, name):
    os.makedirs(OUT, exist_ok=True)
    fig.savefig(os.path.join(OUT, name + '.pdf'), metadata={'Creator': 'Matplotlib', 'CreationDate': None})
    eps = os.path.join(OUT, name + '.eps')
    fig.savefig(eps)
    with open(eps, 'rb') as fh:
        lines = fh.read().split(b'\n')
    with open(eps, 'wb') as fh:
        fh.write(b'\n'.join(l for l in lines if not l.startswith(b'%%CreationDate')))
    fig.savefig(os.path.join(OUT, name + '.tif'), dpi=600, pil_kwargs={'compression': 'tiff_lzw'})
    fig.savefig(os.path.join(OUT, name + '.png'), dpi=200, metadata={'Software': 'Matplotlib'})
    plt.close(fig)


def tidy(ax):
    ax.spines[['top', 'right']].set_visible(False)


# ------------------------------------------------------------------ Fig. 1
def fig_lattice():
    fig, ax = plt.subplots(figsize=(WIDTH, 1.2))
    ax.set_axis_off()
    ax.set_xlim(0, 100)
    ax.set_ylim(2, 28)

    def box(x, y, w, h, txt, fc='#eef3f8'):
        ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle='round,pad=0.4', fc=fc, ec='#123', lw=0.6))
        ax.text(x + w / 2, y + h / 2, txt, ha='center', va='center')

    def arr(x0, y0, x1, y1):
        ax.add_patch(FancyArrowPatch((x0, y0), (x1, y1), arrowstyle='-|>', mutation_scale=6,
                                     lw=0.6, color='#123', shrinkA=0, shrinkB=0))
    top, bot = 22.5, 7.5
    ax.text(0.5, top, r'$x[2n]$', va='center')
    ax.text(0.5, bot, r'$x[2n+1]$', va='center')
    arr(9.5, top, 16, top)
    arr(9.5, bot, 16, bot)
    box(16, 4, 10, 22, r'$\mathbf{R}(\theta_0)$')
    arr(26.4, top, 46, top)
    arr(26.4, bot, 32, bot)
    box(32, 3.5, 8, 8, r'$z^{-1}$', fc='#f7f1e3')
    arr(40.4, bot, 46, bot)
    box(46, 4, 10, 22, r'$\mathbf{R}(\theta_1)$')
    arr(56.4, top, 63, top)
    arr(56.4, bot, 63, bot)
    ax.text(66.5, 15, r'$\cdots$', ha='center', va='center', fontsize=PT + 3)
    arr(70, top, 76, top)
    arr(70, bot, 76, bot)
    box(76, 4, 12, 22, r'$\mathbf{R}(\theta_{N-1})$')
    arr(88.4, top, 94, top)
    arr(88.4, bot, 94, bot)
    ax.text(94.5, top, r'$s[n]$', va='center')
    ax.text(94.5, bot, r'$d[n]$', va='center')
    save(fig, 'fig1_lattice')


# ------------------------------------------------------------------ Fig. 2
def fig_filters(data):
    fig, ax = plt.subplots(1, 2, figsize=(WIDTH, 1.9))
    sets = [(L.HAAR, 'Haar, $N=1$', '#000000', 'o'),
            (L.DAUB4, 'D4, $N=2$', '#7f8c8d', 's')]
    colours = {'CT': '#1f4e79', 'MR': '#c0392b', 'MG': '#148f77'}
    for t in ('CT', 'MR', 'MG'):
        if t in data:
            sets.append((data[t]['trained']['3'], f'trained $N=3$, {t}', colours[t], 'D'))
    for th, lab, c, mk in sets:
        lo, _ = L.filters(th, 64)
        ax[0].plot(np.arange(lo.size), lo, marker=mk, color=c, label=lab)
        H = np.fft.rfft(lo, 2048)
        w = np.linspace(0, 1, H.size)
        ax[1].plot(w, 20 * np.log10(np.maximum(np.abs(H) / math.sqrt(2), 1e-6)), color=c, label=lab)
    ax[0].set_xlabel('tap index $n$')
    ax[0].set_ylabel(r'$h_0[n]$')
    ax[0].set_title('(a) analysis low-pass impulse response')
    ax[1].set_xlabel(r'normalized frequency $\omega/\pi$')
    ax[1].set_ylabel(r'$|H_0(e^{j\omega})|/\sqrt{2}$ (dB)')
    ax[1].set_title('(b) analysis low-pass magnitude response')
    ax[1].set_ylim(-60, 5)
    ax[1].legend(frameon=False, loc='lower left')
    for a in ax:
        tidy(a)
    fig.tight_layout(w_pad=1.5)
    save(fig, 'fig2_filters')


# ------------------------------------------------------------------ Fig. 3
def fig_scalar(data, info):
    im = next(i for i in data['CT']['images'] if i['id'] == info['scalar_example'])
    h = np.array(im['curves']['Haar'])
    e = np.array(im['curves']['ScalarGain'])
    fig, ax = plt.subplots(1, 3, figsize=(WIDTH, 1.75))
    n = min(len(h), len(e))
    k = np.arange(n)
    for arr_, key in ((h, 'Haar'), (e, 'ScalarGain')):
        st = {kk: v for kk, v in STYLE[key].items() if kk != 'label'}
        ax[0].plot(k, arr_[:n, 1], **st, label=STYLE[key]['label'])
        ax[1].plot(k, arr_[:n, 0], **st)
        ax[2].plot(arr_[:, 0], arr_[:, 1], **st)
    ax[0].set_xlabel(r'quantizer step index (same $\Delta$)')
    ax[0].set_ylabel('PSNR (dB)')
    ax[0].set_title(r'(a) quality at equal $\Delta$')
    ax[0].legend(frameon=False, loc='upper left')
    ax[1].set_xlabel(r'quantizer step index (same $\Delta$)')
    ax[1].set_ylabel('rate (bit/pixel)')
    ax[1].set_title(r'(b) rate at equal $\Delta$')
    ax[2].set_xscale('log')
    ax[2].set_xlabel('rate (bit/pixel)')
    ax[2].set_ylabel('PSNR (dB)')
    ax[2].set_title('(c) quality against rate')
    for a in ax:
        tidy(a)
    fig.tight_layout(w_pad=1.0)
    save(fig, 'fig3_scalar')


# ------------------------------------------------------------------ Fig. 4
FIG4 = ['Haar', 'ScalarGain', 'Daub4', 'Lat4', 'CDF97', 'JPEG2000']


def fig_rd(data):
    tags = [t for t in A.TAGS if t in data]
    fig, axes = plt.subplots(2, len(tags), figsize=(WIDTH, 3.5), squeeze=False)
    grid = np.logspace(math.log10(0.03), math.log10(8.0), 80)
    for j, t in enumerate(tags):
        ims = data[t]['images']
        for metric, row in ((1, 0), (2, 1)):
            ax = axes[row][j]
            for k in FIG4:
                m = A.mean_curve(ims, k, metric, grid)
                ax.plot(grid, m, **{**STYLE[k], 'markevery': 10})
            ax.set_xscale('log')
            ax.set_xlabel('rate (bit/pixel)')
            if j == 0:
                ax.set_ylabel('PSNR (dB)' if metric == 1 else 'SSIM')
            if row == 0:
                ax.set_title(TITLE[t])
            tidy(ax)
    h, lab = axes[0][0].get_legend_handles_labels()
    fig.legend(h, lab, frameon=False, ncol=3, loc='lower center', bbox_to_anchor=(0.5, 0.0))
    fig.tight_layout(rect=(0, 0.09, 1, 1), h_pad=0.6, w_pad=0.6)
    save(fig, 'fig4_rd')


# ------------------------------------------------------------------ Fig. 5
FIG5 = ['ScalarGain', 'Daub4', 'Daub6', 'Daub8', 'Lat2', 'Lat3', 'Lat4', 'Lat3img', 'CDF97', 'JPEG2000']


def fig_bd(data):
    tags = [t for t in A.TAGS if t in data]
    fig, axes = plt.subplots(1, len(tags), figsize=(WIDTH, 2.2), squeeze=False)
    for j, t in enumerate(tags):
        ax = axes[0][j]
        vals = []
        for m in FIG5:
            v = [x for _, x in A.per_image(data[t], m) if not math.isnan(x)]
            vals.append(v)
        bp = ax.boxplot(vals, patch_artist=True, widths=0.6, whis=1.5,
                        medianprops=dict(color='black', lw=0.8),
                        whiskerprops=dict(lw=0.5), capprops=dict(lw=0.5),
                        boxprops=dict(lw=0.5),
                        flierprops=dict(marker='.', ms=2, mfc='0.3', mec='none'))
        for patch, m in zip(bp['boxes'], FIG5):
            c = matplotlib.colors.to_rgb(STYLE[m]['color'])
            patch.set_facecolor(tuple(0.55 + 0.45 * x for x in c))
            patch.set_edgecolor('#222')
        ax.axhline(0, color='0.3', lw=0.5, ls='--')
        ax.set_xticks(range(1, len(FIG5) + 1))
        ax.set_xticklabels([SHORT[m] for m in FIG5], rotation=60, ha='right')
        ax.set_title(TITLE[t])
        if j == 0:
            ax.set_ylabel('BD-rate against Haar (%)')
        ax.grid(axis='x', visible=False)
        tidy(ax)
    fig.tight_layout(w_pad=0.8)
    save(fig, 'fig5_bdrate')


# ------------------------------------------------------------------ Fig. 6
def fig_visual():
    z = np.load(os.path.join(HERE, 'results', 'visual.npz'))
    with open(os.path.join(HERE, 'results', 'visual.json')) as f:
        info = json.load(f)
    orig = z['original'].astype(np.float64)
    panels = [('Original', orig, None)] + [(info['labels'][k], z[k].astype(np.float64), orig)
                                           for k in info['order']]
    fig, ax = plt.subplots(2, 4, figsize=(WIDTH, 3.05))
    vmin, vmax = np.percentile(orig, [1, 99.5])
    emax = float(info['error_scale'])
    im_ = None
    for i, (title, a, ref) in enumerate(panels):
        ax[0][i].imshow(a, cmap='gray', vmin=vmin, vmax=vmax, interpolation='nearest')
        if ref is None:
            ax[0][i].set_title('Original detail')
            ax[1][i].axis('off')
        else:
            k = info['order'][i - 1]
            ax[0][i].set_title(f"{title}\n{info['bpp'][k]:.3f} bit/pixel, {info['psnr'][k]:.2f} dB")
            im_ = ax[1][i].imshow(np.abs(ref - a), cmap='inferno', vmin=0, vmax=emax, interpolation='nearest')
            ax[1][i].set_title('absolute error')
    for a in ax.ravel():
        a.set_xticks([])
        a.set_yticks([])
        a.grid(False)
    fig.subplots_adjust(left=0.01, right=0.9, top=0.88, bottom=0.02, wspace=0.06, hspace=0.32)
    cax = fig.add_axes([0.915, 0.04, 0.015, 0.38])
    cb = fig.colorbar(im_, cax=cax)
    cb.set_label('absolute error (stored levels)')
    cb.outline.set_linewidth(0.4)
    save(fig, 'fig6_visual')


def main():
    data = A.load_all()
    with open(os.path.join(HERE, 'results', 'figure_choices.json')) as f:
        info = json.load(f)
    fig_lattice()
    fig_filters(data)
    fig_scalar(data, info)
    fig_rd(data)
    fig_bd(data)
    fig_visual()
    print('figures written to', OUT)


if __name__ == '__main__':
    main()
