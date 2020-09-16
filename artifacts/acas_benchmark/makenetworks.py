#!/usr/bin/env python
import subprocess as sp
import shlex
import sys

from pathlib import Path

CONVERTER = sys.argv[1] if len(sys.argv) > 1 else "reluplexnnet2onnx.py"


def main():
    for network in Path("original").iterdir():
        name = "N_" + "_".join(network.stem.split("_")[2:4])
        cmd = (
            f"python {CONVERTER} {network} --drop_normalization -o onnx/{name}.onnx"
        )
        print(cmd)
        sp.run(shlex.split(cmd))


if __name__ == "__main__":
    main()
