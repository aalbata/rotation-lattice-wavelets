"""Regenerate the tables, numbers and figures of the paper.

    python reproduce.py            fast path: tables, numbers and figures from the
                                   shipped results/ (no images, no compiled coder)
    python reproduce.py --full     full path: build the coder, download the images
                                   listed in data/manifest.csv, rerun every
                                   experiment, then the fast path

Outputs are written to out/ (tables/*.tex, numbers.json, figures/*).
"""
import argparse
import subprocess
import sys


def run(*args):
    print('>', ' '.join(args), flush=True)
    subprocess.run([sys.executable, *args], check=True)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--full', action='store_true', help='rerun everything from the raw downloads')
    a = ap.parse_args()
    if a.full:
        run('build_codec.py')
        run('fetch_data.py')
        for tag in ('CT', 'MR', 'MG'):
            run('experiment.py', tag)
        for tag in ('CT', 'MR', 'MG'):
            run('window_experiment.py', tag)
        run('make_visual.py')
        run('histogram_check.py')
        run('timing.py')
    run('tables.py')
    run('figures.py')


if __name__ == '__main__':
    main()
