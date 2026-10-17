# Paraunitary rotation-lattice wavelets for medical image compression

Code, image manifest and results for the article *Paraunitary rotation-lattice wavelets for medical image compression under rate-matched evaluation* by Ahmmad Rasheed,
Snwr Mohammad and Ali Al Bataineh.

The package implements a two-channel paraunitary rotation lattice whose members include
the orthonormal Haar transform and the four-, six- and eight-tap Daubechies filter banks,
trains its angles per imaging modality through a real codec, and compares it with Haar,
gain-type Haar modifications, CDF 9/7 and JPEG 2000 under a rate-matched protocol: one
deadzone quantizer and one context-adaptive range coder for every transform, rates measured
in compressed bytes, patient-disjoint training and test sets, and images at their stored
precision. Every table, every number in the text and every figure of the article is
generated from the files in `results/` by the scripts here.

## Results

<!-- results:start -->
BD-rate against orthonormal Haar in percent (negative is a saving), mean over the test images with a 95 % patient bootstrap interval. Window columns restrict the comparison to the range between the lowest and the highest national maximum compression ratio of each modality.

| Transform | CT PSNR | MR PSNR | MG PSNR | CT SSIM | MR SSIM | MG SSIM | CT window | MR window | MG window |
|---|---|---|---|---|---|---|---|---|---|
| Per-stage unit gain | 20.4 [18.9, 22.0] | 17.0 [16.7, 17.2] | 15.6 [15.0, 16.2] | 11.9 [11.1, 12.6] | 13.3 [12.8, 13.7] | 17.5 [16.8, 18.1] | 15.6 [15.1, 16.2] | 23.7 [23.4, 24.0] | 22.9 [22.7, 23.2] |
| D4, untrained (N=2) | -12.0 [-15.3, -8.7] | -14.3 [-14.9, -13.9] | -13.0 [-14.8, -11.1] | -7.6 [-11.1, -4.1] | -13.3 [-13.8, -12.8] | -14.3 [-16.5, -12.1] | -10.4 [-13.9, -6.9] | -19.0 [-19.9, -18.3] | -21.6 [-24.1, -19.2] |
| D6, untrained (N=3) | -15.6 [-19.8, -11.3] | -17.7 [-18.5, -17.1] | -15.1 [-17.3, -12.8] | -9.8 [-14.4, -5.2] | -16.8 [-17.6, -16.2] | -17.2 [-19.9, -14.6] | -14.2 [-18.8, -9.5] | -23.7 [-24.9, -22.7] | -26.5 [-29.4, -23.5] |
| D8, untrained (N=4) | -16.7 [-21.4, -11.9] | -19.4 [-20.3, -18.7] | -16.5 [-18.9, -14.0] | -9.9 [-15.1, -4.7] | -18.5 [-19.5, -17.8] | -18.8 [-21.7, -15.9] | -15.5 [-20.8, -10.2] | -25.7 [-27.1, -24.6] | -28.8 [-31.8, -25.7] |
| Lattice N=2, trained | -12.1 [-15.3, -8.9] | -14.4 [-14.9, -14.0] | -13.0 [-14.8, -11.2] | -7.8 [-11.3, -4.4] | -13.3 [-13.9, -12.9] | -14.3 [-16.6, -12.2] |  |  |  |
| Lattice N=3, trained | -16.4 [-20.8, -12.0] | -19.0 [-19.9, -18.3] | -16.2 [-18.5, -13.8] | -10.5 [-15.2, -5.8] | -18.0 [-19.0, -17.3] | -18.4 [-21.2, -15.6] | -14.7 [-19.5, -9.9] | -25.2 [-26.6, -24.1] | -28.4 [-31.4, -25.4] |
| Lattice N=4, trained | -17.9 [-22.7, -13.1] | -20.5 [-21.6, -19.7] | -17.5 [-20.1, -14.8] | -11.3 [-16.6, -6.0] | -19.7 [-20.9, -18.9] | -19.9 [-23.0, -16.8] | -16.3 [-21.6, -11.0] | -27.2 [-28.7, -26.0] | -30.7 [-33.9, -27.5] |
| Lattice N=3, per image | -16.3 [-20.7, -11.8] | -18.9 [-19.8, -18.2] | -16.2 [-18.5, -13.8] | -10.6 [-15.2, -6.0] | -18.0 [-19.0, -17.3] | -18.4 [-21.2, -15.6] |  |  |  |
| CDF 9/7, same coder | -24.3 [-29.7, -18.7] | -24.4 [-25.5, -23.6] | -23.9 [-27.1, -20.7] | -19.6 [-25.5, -13.6] | -23.3 [-24.4, -22.5] | -24.3 [-27.9, -20.8] | -21.2 [-27.3, -15.1] | -30.9 [-32.4, -29.7] | -34.1 [-37.4, -30.8] |
| JPEG 2000 (OpenJPEG) | -29.4 [-35.2, -23.4] | -25.9 [-27.3, -24.9] | -31.9 [-35.7, -28.2] | -25.7 [-31.9, -19.3] | -25.9 [-27.3, -24.7] | -36.9 [-40.2, -33.7] | -24.4 [-31.1, -17.6] | -29.7 [-31.4, -28.4] | -33.8 [-37.5, -30.0] |

Lossless rate in bit per pixel, mean over the test images:

| Method | CT | MR | MG |
|---|---|---|---|
| Reversible lattice N=1 (Haar) | 6.60 | 4.86 | 11.88 |
| Reversible lattice N=3 | 6.44 | 4.62 | 11.53 |
| Reversible LeGall 5/3, same coder | 6.26 | 4.44 | 11.26 |
| JPEG 2000 reversible | 6.09 | 4.25 | 10.79 |
| JPEG-LS | 5.85 | 4.38 | 10.56 |
| JPEG XL | 5.68 | 4.16 | 6.08 |
<!-- results:end -->

## Contents

| File | Purpose |
|---|---|
| `lattice.py` | rotation lattice: floating-point and reversible integer forms, angle constraint, filters |
| `haar.py` | Haar transform with a per-stage gain, level map and level gains |
| `cdf97.py` | CDF 9/7 by lifting with whole-sample symmetric extension, synthesis-basis norms |
| `legall53.py` | reversible LeGall 5/3 integer transform |
| `codec/codec.c` | context-adaptive binary range coder with matching decoders |
| `coder.py` | deadzone quantizer and ctypes bindings to the compiled coder |
| `build_codec.py` | compiles `codec/codec.c` with the C compiler of the `ziglang` package |
| `fetch_data.py` | downloads the images listed in `data/manifest.csv` from TCIA and verifies them |
| `images.py` | loads and crops the images at their stored precision |
| `experiment.py` | angle training, rate-distortion curves, lossless coding and equivalence checks |
| `window_experiment.py` | rate-distortion points spanning the compression-ratio windows widened by a factor of 1.4 at each end |
| `make_visual.py` | reconstructions for the visual comparison figure |
| `histogram_check.py` | JPEG-LS on mammograms with their distinct values renumbered |
| `timing.py` | transform and coder timings |
| `analysis.py` | BD-rate, patient bootstrap intervals, Wilcoxon tests, Holm correction |
| `tables.py` | LaTeX tables, `out/numbers.json` and the results block of this README |
| `figures.py` | every figure (PDF, EPS, TIFF, PNG) |
| `reproduce.py` | fast path and full path |
| `data/manifest.csv` | the 320 DICOM instances selected (240 of them enter the experiments): patient, series and instance UIDs, split, and the SHA-256 of the decoded pixel array as little-endian 32-bit integers |
| `data/selection_log.csv` | patients skipped by the inclusion rules (none) |
| `results/` | per-image results for CT, MR and MG, the window points, timings, and the data of the visual figure |

## Reproducing

The commands are for Windows PowerShell; on Linux or macOS use `python3` and the same
arguments. Python 3.12 is required. On Windows, unpack the package into a short folder path
(for example `C:\rlw`) or enable long-path support, because the `ziglang` compiler package
installs files with long paths.

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
python reproduce.py            # fast path: tables, numbers and figures from results/
python reproduce.py --full     # full path: build the coder, download, rerun everything
```

The fast path needs neither the images nor a compiled coder and takes about a minute.
The full path downloads 320 DICOM instances (about 2.6 GB, mostly mammograms) from the public TCIA API,
compiles the coder, and reruns all experiments; on the machine below the computation takes
about 2 hours with 12 worker processes, plus the download. Steps can be run one at a time:

```powershell
python build_codec.py
python fetch_data.py
python experiment.py CT
python experiment.py MR
python experiment.py MG
python window_experiment.py CT
python window_experiment.py MR
python window_experiment.py MG
python make_visual.py
python histogram_check.py
python timing.py
python tables.py
python figures.py
```

`python fetch_data.py --select` rebuilds the selection itself from the TCIA series lists.
Outputs are written to `out/`.

The reported results were produced on Windows 11 with an Intel Core Ultra 7 155U processor
and Python 3.12.10, with the package versions pinned in `requirements.txt` (direct
dependencies and their dependencies). The bundled
codec libraries were OpenJPEG 2.5.4 and libjxl 0.12.0 (through imagecodecs 2026.8.16) and
CharLS 2.4.2 (through pyjpegls 1.5.1).

## Column definitions

- *BD-rate*: Bjontegaard average rate difference against orthonormal Haar at equal quality,
  in percent, computed per image from least-squares cubic fits of log10 rate against quality
  (of lower degree where fewer than four points fall in the fitting range) over the common
  quality interval; negative values are savings. Mean over the test images of the modality,
  with a 95 % bootstrap interval that resamples patients in brackets.
- *PSNR*: peak signal-to-noise ratio with the image maximum as peak.
- *SSIM*: structural similarity with an 11 x 11 Gaussian window (sigma 1.5), K1 = 0.01,
  K2 = 0.03, population covariances and the image maximum as dynamic range L, used on the
  scale -10 log10(1 - SSIM).
- *Window*: BD-rate restricted to the quality range that Haar reaches between the lowest and
  the highest of the national maximum compression ratios listed as medically acceptable by the
  European Society of Radiology (CT 5:1 and 15:1, MR 5:1 and 24:1, mammography 15:1 and 25:1),
  relative to the occupied bit depth, from eight separate points per image and transform that
  span the rate range of the window widened by a factor of 1.4 at each end.
- *Lossless*: mean rate in bit per pixel at the stored precision.

## Data

The images come from The Cancer Imaging Archive (TCIA) and are not redistributed here,
except a 128 x 128 crop of one PROSTATE-DIAGNOSIS slice and its reconstructions in
`results/visual.npz`. All three collections are distributed under the Creative Commons
Attribution 3.0 license and must be cited when used:

- Armato III, S. G., et al. (2015). Data from LIDC-IDRI (Version 4) [Data set]. The Cancer
  Imaging Archive. https://doi.org/10.7937/K9/TCIA.2015.LO9QL9SX
- Bloch, B. N., Jain, A., & Jaffe, C. C. (2015). Data from PROSTATE-DIAGNOSIS (Version 2)
  [Data set]. The Cancer Imaging Archive. https://doi.org/10.7937/K9/TCIA.2015.FOQEUJVT
- Sawyer-Lee, R., Gimenez, F., Hoogi, A., & Rubin, D. (2016). Curated Breast Imaging Subset
  of Digital Database for Screening Mammography (CBIS-DDSM) (Version 1) [Data set]. The
  Cancer Imaging Archive. https://doi.org/10.7937/K9/TCIA.2016.7O02S9CY
- Clark, K., et al. (2013). The Cancer Imaging Archive (TCIA): Maintaining and operating a
  public information repository. Journal of Digital Imaging, 26(6), 1045-1057.
  https://doi.org/10.1007/s10278-013-9622-7

The authors acknowledge the National Cancer Institute and the Foundation for the National
Institutes of Health, and their critical role in the creation of the free publicly
available LIDC/IDRI Database used in this study.

## License

The code is released under the MIT License (see `LICENSE`). The license does not extend to
the TCIA images or to the image-derived arrays in `results/visual.npz`.
