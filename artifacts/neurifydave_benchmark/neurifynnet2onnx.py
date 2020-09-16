#!/usr/bin/env python
import argparse
import numpy as np
import torch
import torch.nn as nn

from functools import partial
from pathlib import Path
from typing import List, Optional


def _parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "nnet_network", type=Path, help="path to the NNET network to convert"
    )
    parser.add_argument(
        "-o",
        "--output",
        type=Path,
        default=Path("model.onnx"),
        help="path to save the ONNX model",
    )
    return parser.parse_args()


class TransposeFlatten(nn.Module):
    def forward(self, x):
        return x.permute(0, 2, 3, 1).flatten(1)
        # return x.permute(0, 1, 3, 2).flatten(1)


class DaveSuffix(nn.Module):
    def forward(self, x):
        return torch.atan(x) * 2


def build_linear(
    weights: np.ndarray,
    bias: np.ndarray,
    activation: str,
    input_shape: List[int],
    output_shape: List[int],
) -> nn.Module:
    flat_input_size = np.product(input_shape)
    output_shape.extend([1, bias.shape[0]])
    flat_output_size = np.product(output_shape)

    linear_layer = nn.Linear(flat_input_size, flat_output_size)
    linear_layer.weight.data = torch.from_numpy(weights).float()
    linear_layer.bias.data = torch.from_numpy(bias).float()

    activation_layer: Optional[nn.Module] = None
    if activation == "relu":
        activation_layer = nn.ReLU()
    elif activation == "affine":
        return linear_layer
    else:
        raise ValueError(f"Unknown activation type: {activation}")
    return nn.Sequential(linear_layer, activation_layer)


def build_conv(
    weights: np.ndarray,
    bias: np.ndarray,
    activation: str,
    input_shape: List[int],
    output_shape: List[int],
    **params,
):
    assert weights.shape[2] == weights.shape[3]

    conv_layer = nn.Conv2d(
        weights.shape[0],
        weights.shape[1],
        kernel_size=params["kernel_size"],
        stride=params["stride"],
        padding=params["padding"],
    )
    conv_layer.weight.data = torch.from_numpy(weights).float()
    conv_layer.bias.data = torch.from_numpy(bias).float()

    output_shape.extend(conv_layer(torch.ones(input_shape)).shape)

    activation_layer: Optional[nn.Module] = None
    if activation == "relu":
        activation_layer = nn.ReLU()
    elif activation == "affine":
        return conv_layer
    else:
        raise ValueError(f"Unknown activation type: {activation}")
    return nn.Sequential(conv_layer, activation_layer)


def next_line(network_file):
    while True:
        line = network_file.readline().strip().lower().strip(",")
        if line.startswith("//"):
            continue
        return line


def parse_conv_params(param_list):
    out_c, in_c, kernel_size, stride, padding = param_list
    return {
        "out_c": out_c,
        "in_c": in_c,
        "kernel_size": kernel_size,
        "stride": stride,
        "padding": padding,
    }


def main(args: argparse.Namespace):
    print(
        "WARNING!!! "
        "This script is only designed to work on the Neurify DAVE nnet networks."
    )
    layers: List[nn.Module] = []
    with open(args.nnet_network) as network_file:
        num_layers, input_size, output_size, max_layer_size = [
            int(v) for v in next_line(network_file).split(",")
        ]
        layer_sizes = [int(v) for v in next_line(network_file).split(",")]
        layer_types = [-1] + [int(v) for v in next_line(network_file).split(",")]
        conv_params = {
            i: parse_conv_params(int(v) for v in next_line(network_file).split(","))
            for i, t in enumerate(layer_types)
            if t == 1
        }

        layer = 1
        input_shape = [1, 3, 100, 100]
        while layer <= num_layers:
            output_shape: List[int] = []
            weights = []
            bias = []
            if layer_types[layer] == 0:
                build_layer = build_linear
                for _ in range(layer_sizes[layer]):
                    weights.append(
                        [float(v) for v in next_line(network_file).split(",")]
                    )
                for _ in range(layer_sizes[layer]):
                    bias.append(float(next_line(network_file)))
                if layer_types[layer - 1] == 1:
                    weights = (
                        np.asarray(weights)
                        .reshape([layer_sizes[layer]] + input_shape[1:])
                        .transpose(0, 3, 2, 1)
                        .reshape((layer_sizes[layer], -1))
                    )
                    layers.append(TransposeFlatten())
            elif layer_types[layer] == 1:
                build_layer = partial(build_conv, **conv_params[layer])
                for out_c in range(conv_params[layer]["out_c"]):
                    weights.append(
                        [float(v) for v in next_line(network_file).split(",")]
                    )
                weights = (
                    np.asarray(weights)
                    .reshape(
                        (
                            conv_params[layer]["out_c"],
                            conv_params[layer]["in_c"],
                            conv_params[layer]["kernel_size"],
                            conv_params[layer]["kernel_size"],
                        )
                    )
                    .transpose((0, 1, 3, 2))
                )
                for _ in range(conv_params[layer]["out_c"]):
                    bias.append(float(next_line(network_file)))
            else:
                raise RuntimeError(f"Unknown layer type: {layer_types[layer]}")
            W = np.asarray(weights)
            b = np.asarray(bias)
            activation = "relu"
            if layer == num_layers:
                activation = "affine"
            layers.append(build_layer(W, b, activation, input_shape, output_shape))
            input_shape = output_shape
            layer += 1
    pytorch_model = nn.Sequential(*layers, DaveSuffix())
    print(pytorch_model)
    dummy_input = torch.ones([1, 3, 100, 100])
    torch.onnx.export(pytorch_model, dummy_input, args.output)


if __name__ == "__main__":
    main(_parse_args())
