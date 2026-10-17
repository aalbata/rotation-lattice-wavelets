"""Select and download the evaluation images from The Cancer Imaging Archive (TCIA).

Usage
    python fetch_data.py             download the instances listed in data/manifest.csv
    python fetch_data.py --select    rebuild the selection from the TCIA series lists

The selection is deterministic given the TCIA series lists. Candidate series are
filtered by modality and series description, patients are visited in the order of
a seeded permutation, one series is kept per patient, and a patient is skipped when
any of its selected images fails the inclusion rules (two-dimensional frame, minimum
side, dynamic range of at least 64 stored levels). Accepted patients alternate
between the training split and the test split, so no patient contributes images to
both. The selected instances and the reason for every skipped patient are written to
data/manifest.csv and data/selection_log.csv.

Downloading from the manifest fetches each instance individually and checks the
SHA-256 digest of its decoded pixel array against the manifest.
"""
import argparse
import concurrent.futures as cf
import csv
import hashlib
import io
import json
import os
import re
import sys
import time
import urllib.parse
import urllib.request
import zipfile

import numpy as np
import pydicom

API = 'https://services.cancerimagingarchive.net/nbia-api/services/v1'
HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, 'data')
DICOM = os.path.join(DATA, 'dicom')
MANIFEST = os.path.join(DATA, 'manifest.csv')
SELLOG = os.path.join(DATA, 'selection_log.csv')
SEED = 20261016
MIN_RANGE = 64

RULES = {
    'CT': dict(collection='LIDC-IDRI', modality='CT', description=None,
               count=(65, 400), patients=40, per_patient=3, spacing=6, min_side=256),
    'MR': dict(collection='PROSTATE-DIAGNOSIS', modality='MR', description='T2W_TSE_AX',
               count=(13, 400), patients=40, per_patient=3, spacing=6, min_side=256),
    'MG': dict(collection='CBIS-DDSM', modality='MG', description='full mammogram images',
               count=(1, 1), patients=80, per_patient=1, spacing=0, min_side=1024),
}
FIELDS = ['tag', 'split', 'patient', 'collection', 'series_uid', 'sop_uid', 'slice_rank',
          'rows', 'cols', 'bits_stored', 'pixel_sha256', 'license']


def _get(url, timeout=600, tries=4):
    for k in range(tries):
        try:
            with urllib.request.urlopen(url, timeout=timeout) as r:
                return r.read()
        except Exception:
            if k == tries - 1:
                raise
            time.sleep(5 * (k + 1))


def api(path, **params):
    return API + '/' + path + '?' + urllib.parse.urlencode(params)


def patient_key(tag, pid):
    if tag == 'MG':
        return re.search(r'P_\d+', pid).group(0)
    return pid


def pixel_digest(a):
    return hashlib.sha256(np.ascontiguousarray(a.astype('<i4')).tobytes()).hexdigest()


def slice_order(datasets):
    """Order the instances of a series along the slice normal."""
    def pos(ds):
        try:
            o = np.array(ds.ImageOrientationPatient, float)
            n = np.cross(o[:3], o[3:])
            return float(np.dot(n, np.array(ds.ImagePositionPatient, float)))
        except Exception:
            return float(getattr(ds, 'InstanceNumber', 0))
    return sorted(range(len(datasets)), key=lambda i: (pos(datasets[i]), str(datasets[i].SOPInstanceUID)))


def check(a, min_side):
    if a.ndim != 2:
        return 'not a single two-dimensional frame'
    if min(a.shape) < min_side:
        return f'smaller than {min_side} pixels'
    if int(a.max()) - int(a.min()) < MIN_RANGE:
        return f'dynamic range below {MIN_RANGE} levels'
    return ''


def candidates(tag):
    r = RULES[tag]
    series = json.loads(_get(api('getSeries', Collection=r['collection'], Modality=r['modality'])))
    lo, hi = r['count']
    series = [s for s in series if lo <= int(s.get('ImageCount', 0)) <= hi
              and (r['description'] is None or s.get('SeriesDescription') == r['description'])]
    series.sort(key=lambda s: s['SeriesInstanceUID'])
    by_patient = {}
    for s in series:
        by_patient.setdefault(patient_key(tag, s['PatientID']), []).append(s)
    patients = sorted(by_patient)
    rng = np.random.default_rng(SEED + list(RULES).index(tag))
    order = [patients[i] for i in rng.permutation(len(patients))]
    out = []
    for p in order:
        group = by_patient[p]
        out.append((p, group[int(rng.integers(len(group)))]))
    return out


def evaluate(tag, patient, s):
    """Download one series and return the accepted instances or a rejection reason."""
    r = RULES[tag]
    raw = _get(api('getImage', SeriesInstanceUID=s['SeriesInstanceUID']))
    z = zipfile.ZipFile(io.BytesIO(raw))
    files = [n for n in z.namelist() if not n.endswith('/') and not n.lower().endswith('.json')]
    datasets, blobs = [], []
    for n in files:
        b = z.read(n)
        try:
            datasets.append(pydicom.dcmread(io.BytesIO(b), stop_before_pixels=True))
            blobs.append(b)
        except Exception:
            continue
    if len(datasets) < max(1, 2 * r['spacing'] + 1):
        return None, 'too few instances'
    order = slice_order(datasets)
    mid = len(order) // 2
    ranks = [mid - r['spacing'], mid, mid + r['spacing']] if r['per_patient'] == 3 else [mid]
    rows = []
    for k, rk in enumerate(ranks):
        b = blobs[order[rk]]
        ds = pydicom.dcmread(io.BytesIO(b))
        try:
            a = ds.pixel_array
        except Exception as e:
            return None, f'pixel data not decodable ({type(e).__name__})'
        why = check(a, r['min_side'])
        if why:
            return None, why
        rows.append(dict(tag=tag, patient=patient, collection=r['collection'],
                         series_uid=s['SeriesInstanceUID'], sop_uid=str(ds.SOPInstanceUID),
                         slice_rank=k, rows=int(a.shape[0]), cols=int(a.shape[1]),
                         bits_stored=int(ds.BitsStored), pixel_sha256=pixel_digest(a),
                         license=s.get('LicenseName', ''), blob=b))
    return rows, ''


def save_blob(tag, sop, b):
    d = os.path.join(DICOM, tag)
    os.makedirs(d, exist_ok=True)
    with open(os.path.join(d, sop + '.dcm'), 'wb') as f:
        f.write(b)


def select(tags, workers=6):
    manifest, log = [], []
    for tag in tags:
        need = RULES[tag]['patients']
        cand = candidates(tag)
        accepted = 0
        pos = 0
        print(f'{tag}: {len(cand)} candidate patients', flush=True)
        with cf.ThreadPoolExecutor(workers) as ex:
            while accepted < need and pos < len(cand):
                batch = cand[pos:pos + workers]
                pos += len(batch)
                futs = [ex.submit(evaluate, tag, p, s) for p, s in batch]
                for (p, s), f in zip(batch, futs):
                    if accepted >= need:
                        break
                    try:
                        rows, why = f.result()
                    except Exception as e:
                        rows, why = None, f'download failed ({type(e).__name__})'
                    if rows is None:
                        log.append(dict(tag=tag, patient=p, series_uid=s['SeriesInstanceUID'], reason=why))
                        continue
                    split = 'train' if accepted % 2 == 0 else 'test'
                    for row in rows:
                        save_blob(tag, row['sop_uid'], row.pop('blob'))
                        row['split'] = split
                        manifest.append(row)
                    accepted += 1
                    print(f'  {tag} {accepted}/{need} {p} ({split})', flush=True)
        if accepted < need:
            sys.exit(f'{tag}: only {accepted} patients satisfy the inclusion rules')
    os.makedirs(DATA, exist_ok=True)
    with open(MANIFEST, 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=FIELDS)
        w.writeheader()
        w.writerows(manifest)
    with open(SELLOG, 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=['tag', 'patient', 'series_uid', 'reason'])
        w.writeheader()
        w.writerows(log)
    print(f'wrote {len(manifest)} instances to {MANIFEST} and {len(log)} skipped patients to {SELLOG}')


def read_manifest():
    with open(MANIFEST, newline='') as f:
        return list(csv.DictReader(f))


def download(workers=6):
    rows = read_manifest()

    def one(row):
        path = os.path.join(DICOM, row['tag'], row['sop_uid'] + '.dcm')
        if not os.path.exists(path):
            b = _get(api('getSingleImage', SeriesInstanceUID=row['series_uid'],
                         SOPInstanceUID=row['sop_uid']))
            save_blob(row['tag'], row['sop_uid'], b)
        a = pydicom.dcmread(path).pixel_array
        return pixel_digest(a) == row['pixel_sha256']

    with cf.ThreadPoolExecutor(workers) as ex:
        ok = list(ex.map(one, rows))
    bad = [r['sop_uid'] for r, g in zip(rows, ok) if not g]
    if bad:
        sys.exit(f'{len(bad)} instances do not match the manifest digest: {bad[:3]}')
    print(f'{len(rows)} instances present and verified')


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--select', action='store_true')
    ap.add_argument('--tags', nargs='*', default=list(RULES))
    a = ap.parse_args()
    if a.select:
        select(a.tags)
    else:
        download()
