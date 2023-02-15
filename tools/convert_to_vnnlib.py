#!/usr/bin/env python

import os
import argparse
import numpy as np
import torch
import torchvision.datasets as datasets
import pathlib


def _parse_args():
    parser = argparse.ArgumentParser(
        description="convert GDVB benchmark to vnn-lib format"
    )
    parser.add_argument("root", type=str, help="path to GDVB benchmark")
    parser.add_argument("--artifact", type=str, default="MNIST", help="artifact")
    parser.add_argument("--timeout", type=int, default=300)
    return parser.parse_args()


def main(args):

    test = []
    prop_dir = os.path.join(args.root, "veri_slurm")
    for x in [x for x in os.listdir(prop_dir) if "deeppoly" in x]:
        lines = open(os.path.join(prop_dir, x), "r").readlines()
        test += [x for x in lines if "python" in x]

    if args.artifact == "MNIST":
        dataset = datasets.MNIST(
            root="./data", train=True, download=True, transform=None
        )
    elif args.artifact == "DAVE2":
        pass
    else:
        assert False

    benchmark_csv = []
    test = list(set(test))

    for x in test:
        tks = x.split()
        # print(tks)
        # model_file = tks[9].replace("results", args.root.split("/")[0])
        for x in tks:
            if "onnx" in x:
                model_file = x.replace("results", args.root.split("/")[0])
            if "robust" in x:
                property_file = x.replace("results", args.root.split("/")[0])

        assert model_file and property_file

        lines = open(property_file, "r").readlines()
        # print(lines)

        # calculate input constraints
        img = [x for x in lines if "Image" in x]
        assert len(img) == 1
        img_id = int(img[0].split("/")[-1].split(".")[0])

        img_path = img[0].split("(")[1].split(")")[0][1:-1]

        img_npy = np.load(img_path).flatten()

        eps = [x for x in lines if "epsilon" in x]
        assert len(eps) == 2
        eps = float(eps[0][9:-1])
        eps = float(f"{eps:.4f}")

        # generate VNN-lib Property
        # 1) define input
        # vnn_lib_lines = [f"; Mnist property with label: {label}.", ""]
        vnn_lib_lines = []
        for x in range(len(img_npy)):
            vnn_lib_lines += [f"(declare-const X_{x} Real)"]

        # 2) define output
        vnn_lib_lines += [""]
        if args.artifact == "MNIST":
            nb_output = 10
        elif args.artifact == "DAVE2":
            nb_output = 1
        else:
            assert False

        for x in range(10):
            vnn_lib_lines += [f"(declare-const Y_{x} Real)"]

        # 3) define input constraints:
        vnn_lib_lines += ["", "; Input constraints:"]
        for i, x in enumerate(img_npy):
            vnn_lib_lines += [
                f"(assert (<= X_{i} {x+eps}))",
                f"(assert (>= X_{i} {x-eps}))",
            ]

        # 4) define output constraints:
        vnn_lib_lines += ["", f"; Output constraints:"]
        if args.artifact == "MNIST":
            label = dataset[img_id][1]
            vnn_lib_lines += ["(assert (or"]
            for x in range(10):
                vnn_lib_lines += [f"(and (>= Y_{x} Y_{label}))"]
            vnn_lib_lines += ["))"]
        elif args.artifact == "DAVE2":
            lines = open(
                os.path.join(os.path.dirname(img_path), "properties.csv"), "r"
            ).readlines()

            for x in lines[1:]:
                tokens = x.split(",")
                if img_id == int(tokens[0]):
                    min_val = tokens[-2]
                    max_val = tokens[-1]
                    break
            assert min_val and max_val
            vnn_lib_lines += ["(assert (or"]
            vnn_lib_lines += [f"(and (>= Y_0 {min_val}))"]
            vnn_lib_lines += [f"(and (<= Y_0 {max_val})"]
            vnn_lib_lines += ["))"]
        else:
            assert False

        # save property file
        property_name = f'{model_file.split("/")[-1][:-5]}_{img_id}_{eps}.vnnlib'
        property_dir = os.path.join(args.root, "vnnlib")
        pathlib.Path(property_dir).mkdir(parents=True, exist_ok=True)

        open(os.path.join(property_dir, property_name), "w").writelines(
            x + "\n" for x in vnn_lib_lines
        )
        # add to benchmark csv
        benchmark_csv += [
            f"onnx/{model_file.split('/')[-1]},vnnlib/{property_name},{args.timeout}"
        ]

    open(os.path.join(args.root, "instances.csv"), "w").writelines(
        x + "\n" for x in benchmark_csv
    )


if __name__ == "__main__":
    main(_parse_args())
