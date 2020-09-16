#!/usr/bin/env python
import subprocess as sp
import shlex
import sys

from pathlib import Path

CONVERTER = sys.argv[1] if len(sys.argv) > 1 else "eran2onnx.py"


def main():
    for network in Path("original").iterdir():
        cmd = f"python {CONVERTER} {network} --input_shape 3 32 32 --check_cifar -o onnx/{network.stem}.onnx"
        print(cmd)
        sp.run(shlex.split(cmd))


if __name__ == "__main__":
    main()
