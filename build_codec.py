"""Compile codec/codec.c into the shared library loaded by coder.py.

The C compiler is the one shipped in the ziglang wheel (pinned in
requirements.txt), so no system compiler is needed on Windows, Linux or macOS.
Set CC to use another compiler instead, for example CC="gcc".
"""
import os
import shutil
import subprocess
import sys
import tempfile

HERE = os.path.dirname(os.path.abspath(__file__))
SRC = os.path.join(HERE, 'codec', 'codec.c')


def library_path():
    name = {'win32': 'rlcodec.dll', 'darwin': 'librlcodec.dylib'}.get(sys.platform, 'librlcodec.so')
    return os.path.join(HERE, 'codec', name)


def main():
    out = library_path()
    cc = os.environ.get('CC')
    with tempfile.TemporaryDirectory() as tmp:
        target = os.path.join(tmp, os.path.basename(out))
        if cc:
            cmd = cc.split() + ['-O2', '-shared', '-fPIC', '-o', target, SRC]
        else:
            cmd = [sys.executable, '-m', 'ziglang', 'cc', '-O2', '-shared', '-o', target, SRC]
            if sys.platform != 'win32':
                cmd.insert(5, '-fPIC')
        print(' '.join(cmd))
        subprocess.run(cmd, check=True)
        shutil.copyfile(target, out)
    print('built', out)


if __name__ == '__main__':
    main()
