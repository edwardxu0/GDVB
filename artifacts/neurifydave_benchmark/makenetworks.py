#!/usr/bin/env python
import subprocess as sp
import shlex
import sys

from pathlib import Path

CONVERTER = sys.argv[1] if len(sys.argv) > 1 else "neurifynnet2onnx.py"


def main():
    for network in Path("original/networks").iterdir():
        name = network.stem
        cmd = f"python {CONVERTER} {network} -o onnx/{name}.onnx"
        print(cmd)
        sp.run(shlex.split(cmd))


if __name__ == "__main__":
    main()
