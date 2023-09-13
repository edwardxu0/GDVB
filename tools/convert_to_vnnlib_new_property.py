#!/usr/bin/env python

import os
import argparse
import numpy as np
from torchvision.transforms import transforms, ToTensor
import torchvision.datasets as datasets
import pathlib


def _parse_args():
    parser = argparse.ArgumentParser(
        description="convert GDVB benchmark to vnn-lib format"
    )
    parser.add_argument("root", type=str, help="path to GDVB benchmark")
    parser.add_argument("artifact", type=str)
    parser.add_argument("eps", type=float)
    parser.add_argument("props", type=int)
    parser.add_argument("timeout", type=int)
    return parser.parse_args()


def main(args):
    if args.artifact in ["MNIST", "CIFAR10"]:
        if args.artifact == "CIFAR10":
            mean = [0.4914, 0.4822, 0.4465]
            std = [0.2023, 0.1994, 0.201]
        else:
            assert False
        transform = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize(mean=mean, std=std),
            ]
        )
        dataset = eval(f"datasets.{args.artifact}")(
            root="./data", train=False, download=True, transform=transform
        )
    else:
        assert False

    benchmark_csv = []

    models = os.listdir(os.path.join(args.root, "dis_model"))
    for m in models:
        for p in range(args.props):
            print(m, p)

            img_pil, label = dataset[p]
            img_npy = np.array(img_pil).flatten()

            # generate VNN-lib Property
            # 1) define input
            # vnn_lib_lines = [f"; Mnist property with label: {label}.", ""]
            vnn_lib_lines = []
            for x in range(len(img_npy)):
                vnn_lib_lines += [f"(declare-const X_{x} Real)"]

            # 2) define output
            vnn_lib_lines += [""]
            if args.artifact in ["MNIST", "CIFAR10"]:
                nb_output = 10
            elif args.artifact == "DAVE2":
                nb_output = 1
            else:
                assert False

            for x in range(nb_output):
                vnn_lib_lines += [f"(declare-const Y_{x} Real)"]

            # 3) define input constraints:
            vnn_lib_lines += ["", "; Input constraints:"]
            for i, x in enumerate(img_npy):
                vnn_lib_lines += [
                    f"(assert (<= X_{i} {x+args.eps}))",
                    f"(assert (>= X_{i} {x-args.eps}))",
                ]

            # 4) define output constraints:
            vnn_lib_lines += ["", f"; Output constraints:"]
            if args.artifact in ["MNIST", "CIFAR10"]:
                vnn_lib_lines += ["(assert (or"]
                for x in range(10):
                    vnn_lib_lines += [f"(and (>= Y_{x} Y_{label}))"]
                vnn_lib_lines += ["))"]
            else:
                assert False

            # save property file
            property_name = f"{m[:-5]}_{p}_{args.eps}.vnnlib"
            property_dir = os.path.join(args.root, "vnnlib")
            print(property_dir)
            pathlib.Path(property_dir).mkdir(parents=True, exist_ok=True)

            open(os.path.join(property_dir, property_name), "w").writelines(
                x + "\n" for x in vnn_lib_lines
            )
            # add to benchmark csv
            benchmark_csv += [f"onnx/{m},vnnlib/{property_name},{args.timeout}"]

    open(os.path.join(args.root, "instances.csv"), "w").writelines(
        x + "\n" for x in benchmark_csv
    )


if __name__ == "__main__":
    main(_parse_args())
