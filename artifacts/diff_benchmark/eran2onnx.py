#!/usr/bin/env python
import argparse
import ast
import numpy as np
import re
import torch
import torch.nn as nn
import torch.utils.data as data

from pathlib import Path
from torchvision import datasets, transforms
from typing import Dict, List, Optional, Tuple

ParamDict = Dict[str, np.ndarray]


class PytorchReshape(nn.Module):
    def __init__(self, shape):
        super().__init__()
        self.shape = (-1,) + tuple(shape)

    def forward(self, x):
        return x.contiguous().view(self.shape)


class PytorchTranspose(nn.Module):
    def __init__(self, *dims):
        super().__init__()
        self.dims = (0,) + tuple(d + 1 for d in dims)

    def forward(self, x):
        return x.permute(self.dims)


class PytorchParSum(nn.Module):
    def __init__(self, par_sum_1, par_sum_2):
        super().__init__()
        self.par_sum_1 = nn.Sequential(*par_sum_1)
        self.par_sum_2 = nn.Sequential(*par_sum_2)

    def forward(self, x):
        y_1 = self.par_sum_1(x)
        y_2 = self.par_sum_2(x)
        return y_1 + y_2


def _parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "eran_network", type=Path, help="path to the ERAN network to convert"
    )
    parser.add_argument(
        "-o",
        "--output",
        type=Path,
        default=Path("model.onnx"),
        help="path to save the ONNX model",
    )
    parser.add_argument(
        "--input_shape",
        type=int,
        nargs="+",
        default=[1, 28, 28],
        help="the input shape to the network (in CHW format)",
    )
    parser.add_argument(
        "--drop_normalization",
        action="store_true",
        help="do not include any input normalization in the converted model",
    )
    parser.add_argument(
        "--check_cifar_accuracy",
        action="store_true",
        help="evaluate the converted model on the CIFAR10 test set",
    )
    parser.add_argument(
        "--check_mnist_accuracy",
        action="store_true",
        help="evaluate the converted model on the MNIST test set",
    )
    return parser.parse_args()


def parse_layer_params(param_str: str) -> ParamDict:
    params = []
    pattern = re.compile(r"([a-zA-Z_]+?=.+?),? [a-zA-Z]")
    while True:
        param_match = re.match(pattern, param_str)
        if param_match is None:
            params.append(param_str)
            break
        params.append(param_match.group(1))
        param_str = param_str[param_match.end() - 1 :]

    param_dict = {}
    for param in params:
        key, value = param.split("=")
        param_dict[key] = np.array(ast.literal_eval(value))
    return param_dict


def build_normalize(
    parameters: ParamDict, input_shape: List[int], output_shape: List[int]
) -> nn.Module:
    output_shape.extend(input_shape)
    num_c = input_shape[0]
    weights = np.diag(1.0 / parameters["std"])
    bias = -parameters["mean"] / parameters["std"]
    norm_layer = nn.Conv2d(num_c, num_c, 1, 1)
    norm_layer.weight.data = torch.from_numpy(
        weights.reshape(num_c, num_c, 1, 1)
    ).float()
    norm_layer.bias.data = torch.from_numpy(bias).float()
    return norm_layer


def build_linear(
    weights: np.ndarray,
    bias: np.ndarray,
    activation: str,
    input_shape: List[int],
    output_shape: List[int],
) -> nn.Module:
    flat_input_size = np.product(input_shape)
    output_shape.append(bias.shape[0])
    flat_output_size = np.product(output_shape)

    linear_layer = nn.Linear(flat_input_size, flat_output_size)
    linear_layer.weight.data = torch.from_numpy(weights).float()
    linear_layer.bias.data = torch.from_numpy(bias).float()

    activation_layer: Optional[nn.Module] = None
    if activation == "relu":
        activation_layer = nn.ReLU()
    elif activation == "sigmoid":
        activation_layer = nn.Sigmoid()
    elif activation == "tanh":
        activation_layer = nn.Tanh()
    elif activation == "affine":
        return linear_layer
    else:
        raise ValueError(f"Unknown activation type: {activation}")
    return nn.Sequential(linear_layer, activation_layer)


def build_conv(
    weights: np.ndarray,
    bias: np.ndarray,
    activation: str,
    parameters: ParamDict,
    input_shape: List[int],
    output_shape: List[int],
) -> nn.Module:
    k = parameters["kernel_size"]
    if not k.shape or len(k) == 1:
        k_h = k_w = k.item()
    else:
        assert len(k) == 2
        k_h, k_w = k
    s = parameters.get("stride", np.array([1, 1]))
    if not s.shape or len(s) == 1:
        s_h = s_w = s.item()
    else:
        assert len(s) == 2
        s_h, s_w = s
    p = parameters.get("padding", np.array([0]))
    assert not p.shape or len(p) == 1
    if p.item() >= 1:
        _, in_height, in_width = input_shape
        out_height = round(np.ceil(float(in_height) / s_h).item())
        out_width = round(np.ceil(float(in_width) / s_w).item())
        pad_along_height = max((out_height - 1) * s_h + k_h - in_height, 0)
        pad_along_width = max((out_width - 1) * s_w + k_w - in_width, 0)
        p_top = pad_along_height // 2
        p_bottom = pad_along_height - p_top
        p_left = pad_along_width // 2
        p_right = pad_along_width - p_left
    else:
        p_left = p_right = p_top = p_bottom = 0

    in_c, in_h, in_w = input_shape
    out_c = parameters["filters"].item()
    out_h = int(np.floor(float(in_h - k_h + p_top + p_bottom) / s_h + 1))
    out_w = int(np.floor(float(in_w - k_w + p_left + p_right) / s_w + 1))
    output_shape.extend([out_c, out_h, out_w])

    pad_layer = nn.ZeroPad2d((p_left, p_right, p_top, p_bottom))
    conv_layer = nn.Conv2d(
        input_shape[0], output_shape[0], (k_h, k_w), (s_h, s_w), (0, 0),
    )
    conv_layer.weight.data = torch.from_numpy(weights).float().permute(3, 2, 0, 1)
    conv_layer.bias.data = torch.from_numpy(bias).float()

    pad_conv_layer = nn.Sequential(pad_layer, conv_layer)

    activation_layer: Optional[nn.Module] = None
    if activation == "relu":
        activation_layer = nn.ReLU()
    elif activation == "sigmoid":
        activation_layer = nn.Sigmoid()
    elif activation == "tanh":
        activation_layer = nn.Tanh()
    elif activation == "affine":
        return pad_conv_layer
    else:
        raise ValueError(f"Unknown activation type: {activation}")
    return nn.Sequential(pad_conv_layer, activation_layer)


def build_maxpool(
    parameters: ParamDict, input_shape: List[int], output_shape: List[int]
) -> nn.Module:
    k = parameters["pool_size"]
    if not k.shape or len(k) == 1:
        k_h = k_w = k.item()
    else:
        assert len(k) == 2
        k_h, k_w = k
    if "padding" in parameters:
        raise ValueError("Padding for MaxPool is not currently supported")
    p_top = p_left = p_bottom = p_right = 0
    s = parameters.get("stride", np.array([k_h, k_w]))
    if not s.shape or len(s) == 1:
        s_h = s_w = s.item()
    else:
        assert len(k) == 2
        s_h, s_w = s

    in_c, in_h, in_w = input_shape
    out_c = in_c
    out_h = int(np.floor(float(in_h - k_h + p_top + p_bottom) / s_h + 1))
    out_w = int(np.floor(float(in_w - k_w + p_left + p_right) / s_w + 1))
    output_shape.extend([out_c, out_h, out_w])

    pool_layer = nn.MaxPool2d((k_h, k_w), (s_h, s_w))
    return pool_layer


def build_parsum(
    par_sum_1: List[nn.Module],
    par_sum_2: List[nn.Module],
    activation: str,
    input_shape: List[int],
    output_shape: List[int],
) -> nn.Module:
    parsum_layer = PytorchParSum(par_sum_1, par_sum_2)

    activation_layer: Optional[nn.Module] = None
    if activation == "relu":
        activation_layer = nn.ReLU()
    elif activation == "sigmoid":
        activation_layer = nn.Sigmoid()
    elif activation == "tanh":
        activation_layer = nn.Tanh()
    elif activation == "affine":
        return parsum_layer
    else:
        raise ValueError(f"Unknown activation type: {activation}")
    return nn.Sequential(parsum_layer, activation_layer)


def main(args: argparse.Namespace):
    layers: List[nn.Module] = []
    shapes: List[Tuple[Tuple[int, ...], Tuple[int, ...]]] = []
    par_sum_start_index: Optional[int] = None
    par_sum_1: Optional[List[nn.Module]] = None
    par_sum_2: Optional[List[nn.Module]] = None
    last_layer: str = "input"
    input_shape: List[int] = args.input_shape
    output_shape: List[int] = []
    with open(args.eran_network) as network_file:
        while True:
            line = network_file.readline().strip().lower()
            if line.startswith("normalize"):
                if args.drop_normalization:
                    continue
                parameters = parse_layer_params(line.split(maxsplit=1)[1])
                layers.append(build_normalize(parameters, input_shape, output_shape))
                last_layer = "normalize"
            elif line in ("affine", "relu", "sigmoid", "tanh"):
                activation = line.strip(", \n").lower()
                W = np.array(ast.literal_eval(network_file.readline().strip()))
                b = np.array(ast.literal_eval(network_file.readline().strip()))
                if last_layer in ("conv", "normalize", "parsumrelu", "parsumcomplete"):
                    c, h, w = input_shape
                    m = np.zeros((h * w * c, h * w * c))
                    column = 0
                    for i in range(h * w):
                        for j in range(c):
                            m[i + j * h * w, column] = 1
                            column += 1
                    W = np.matmul(W, m)
                    layers.append(PytorchTranspose(1, 2, 0))
                    layers.append(PytorchReshape([np.product(input_shape)]))
                elif last_layer in ("input", "maxpool"):
                    layers.append(PytorchTranspose(1, 2, 0))
                    layers.append(PytorchReshape([np.product(input_shape)]))
                layers.append(build_linear(W, b, activation, input_shape, output_shape))
                last_layer = "fc"
            elif line == "conv2d":
                activation, param_string = (
                    network_file.readline().strip(", \n").split(maxsplit=1)
                )
                parameters = parse_layer_params(param_string)
                activation = activation.strip(", \n").lower()
                W = np.array(ast.literal_eval(network_file.readline().strip()))
                b = np.array(ast.literal_eval(network_file.readline().strip()))
                layers.append(
                    build_conv(W, b, activation, parameters, input_shape, output_shape)
                )
                last_layer = "conv"
            elif line == "maxpooling2d":
                parameters = parse_layer_params(network_file.readline().strip(", \n"))
                layers.append(build_maxpool(parameters, input_shape, output_shape))
                last_layer = "maxpool"
            elif line == "parsum1":
                assert par_sum_start_index is None
                par_sum_start_index = len(layers)
                last_layer = "parsum1"
                continue
            elif line == "parsum2":
                assert par_sum_start_index is not None
                par_sum_2 = layers[par_sum_start_index:]
                layers = layers[:par_sum_start_index]
                shapes = shapes[:par_sum_start_index]
                output_shape = list(shapes[-1][1])
                last_layer = "parsum2"
            elif line == "parsumrelu" or line == "parsumcomplete":
                assert par_sum_start_index is not None
                assert par_sum_2 is not None
                activation = "affine"
                if line == "parsumrelu":
                    activation = "relu"
                output_shape = list(shapes[-1][1])
                par_sum_1 = layers[par_sum_start_index:]
                layers = layers[:par_sum_start_index]
                shapes = shapes[:par_sum_start_index]
                input_shape = list(shapes[-1][1])
                layers.append(
                    build_parsum(
                        par_sum_2, par_sum_1, activation, input_shape, output_shape
                    )
                )
                par_sum_1 = None
                par_sum_2 = None
                par_sum_start_index = None
                last_layer = line
            elif line == "":
                break
            else:
                raise ValueError(f"Unknown layer type: {line}")
            shapes.append((tuple(input_shape), tuple(output_shape)))
            input_shape = output_shape
            output_shape = []
    pytorch_model = nn.Sequential(*layers)
    print(pytorch_model)
    dummy_input = torch.ones([1] + args.input_shape)
    torch.onnx.export(pytorch_model, dummy_input, args.output)

    if args.check_mnist_accuracy:
        data_loader = data.DataLoader(
            datasets.MNIST(
                "/tmp/data",
                train=False,
                download=True,
                transform=transforms.Compose([transforms.ToTensor()]),
            ),
            batch_size=1000,
        )
        pytorch_model.eval().cuda()
        num_correct = 0.0
        for i, (x, y) in enumerate(data_loader):
            y_ = pytorch_model(x.cuda()).argmax(dim=-1).cpu()
            num_correct += (y == y_).sum().item()
        accuracy = num_correct / len(data_loader.dataset)
        print("Accuracy:", accuracy)
    if args.check_cifar_accuracy:
        data_loader = data.DataLoader(
            datasets.CIFAR10(
                "/tmp/data",
                train=False,
                download=True,
                transform=transforms.Compose(
                    [
                        transforms.ToTensor(),
                        # transforms.Normalize(
                        #     mean=[0.4914, 0.4822, 0.4465], std=[0.2023, 0.1994, 0.201]
                        # ),
                    ]
                ),
            ),
            batch_size=100,
        )
        pytorch_model.eval().cuda()
        num_correct = 0.0
        for i, (x, y) in enumerate(data_loader):
            y_ = pytorch_model(x.cuda()).argmax(dim=-1).cpu()
            num_correct += (y == y_).sum().item()
        accuracy = num_correct / len(data_loader.dataset)
        print("Accuracy:", accuracy)


if __name__ == "__main__":
    main(_parse_args())
