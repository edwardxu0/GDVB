#!/usr/bin/env python3
import ast
import copy
import logging
import multiprocessing as mp
import numpy as np

logger = logging.getLogger()
logger.setLevel(logging.ERROR)

from itertools import combinations
from multiprocessing.pool import ThreadPool
from pathlib import Path
from r4v.nn import load_network

TEMPLATE = """[distillation]
maxmemory="32G"
threshold=1e-9
cuda=true
type="regression"
precompute_teacher=true

[distillation.parameters]
epochs=10
optimizer="adadelta"
rho=0.95
loss="MSE"
learning_rate=1.0

[distillation.data]
format="udacity-driving"
batchsize=256
presized=true

[distillation.data.transform]
bgr=true
mean=[103.939, 116.779, 123.68]
max_value=255.0

[distillation.data.train]
shuffle=true

[distillation.data.train.teacher]
path="artifacts/udacity.sdc.100/training"

[distillation.data.train.student]
path="artifacts/udacity.sdc.100/training"

[distillation.data.validation]
shuffle=false

[distillation.data.validation.teacher]
path="artifacts/udacity.sdc.100/validation"

[distillation.data.validation.student]
path="artifacts/udacity.sdc.100/validation"

[distillation.teacher]
framework="onnx"
input_shape=[1, 100, 100, 3]
input_format="NHWC"
model="networks/dave/model.onnx"

"""


def get_num_neurons(args):
    dave, transform = args
    drop_layers, scale_layers_1, scale_layers_2 = transform
    if len(drop_layers) > 0:
        dave.drop_layer(list(drop_layers))
    if len(scale_layers_1) > 0:
        dave.scale_layer(list(scale_layers_1), 0.5)
    if len(scale_layers_2) > 0:
        dave.scale_layer(list(scale_layers_2), 0.5)
    network = dave.as_pytorch()
    return transform, network.num_neurons()


def get_neuron_counts():
    droppable_layers = [0, 1, 2, 3, 4, 7, 8, 9, 10]
    drop_layers = []
    for k in range(len(droppable_layers)):
        for layers in combinations(droppable_layers, k):
            drop_layers.append(layers)
    transforms = []
    for dropped_layers in drop_layers:
        scalable_layers = [i for i in droppable_layers if i not in dropped_layers]
        for k_1 in range(len(scalable_layers) + 1):
            for scaled_layers_1 in combinations(scalable_layers, k_1):
                for k_2 in range(len(scaled_layers_1) + 1):
                    for scaled_layers_2 in combinations(scaled_layers_1, k_2):
                        transform = (dropped_layers, scaled_layers_1, scaled_layers_2)
                        print(transform)
                        transforms.append(transform)
    print(len(transforms))

    print("Measuring neuron counts.")
    dave = load_network(
        {
            "model": "networks/dave/model.onnx",
            "input_shape": [1, 100, 100, 3],
            "input_format": "NHWC",
        }
    )

    print("Collecting results.")
    configs = {}
    # for transform, num_neurons in neuron_counts:
    for transform, num_neurons in (
        get_num_neurons((copy.deepcopy(dave), t)) for t in transforms
    ):
        configs[transform] = num_neurons
        print(len(configs), transform, num_neurons)
    return configs


def main():
    neuron_count_limit = (
        load_network(
            {
                "model": "networks/dave/model.onnx",
                "input_shape": [1, 100, 100, 3],
                "input_format": "NHWC",
            }
        )
        .as_pytorch()
        .num_neurons()
    )
    if not Path("dave.sampling.log").exists():
        configs = get_neuron_counts()
    else:
        configs = {}
        with open("dave.sampling.log") as f:
            collecting_results = False
            last_index = None
            for line in f:
                if line.startswith("Collecting results."):
                    print(line.strip())
                    collecting_results = True
                    continue
                if not collecting_results:
                    continue
                print(line.strip())
                split_line = line.strip().split()
                index = int(split_line[0])
                if index == last_index:
                    collecting_results = False
                    continue
                last_index = index
                num_neurons = int(split_line[-1])
                transform = ast.literal_eval(" ".join(split_line[1:-1]))
                configs[transform] = num_neurons
    neuron_counts = list(configs.values())
    print(
        len(configs),
        np.min(neuron_counts),
        np.max(neuron_counts),
        np.mean(neuron_counts),
        np.median(neuron_counts),
    )
    min_neuron_count = np.min(neuron_counts)
    nbins = 10
    bin_width = (neuron_count_limit - min_neuron_count) / nbins
    bin_limits = [min_neuron_count + bin_width * i for i in range(nbins + 1)]
    bins = [[] for _ in range(nbins)]
    for transform, num_neurons in sorted(configs.items(), key=lambda kv: kv[1]):
        for i, (bin_min, bin_max) in enumerate(zip(bin_limits, bin_limits[1:])):
            if num_neurons >= bin_min and num_neurons < bin_max:
                bins[i].append(transform)
    for i, b in enumerate(bins):
        print(i, (bin_limits[i], bin_limits[i + 1]), len(b))
    print()
    selected = set()
    count = 0
    while len(selected) < 20:
        count += 1
        for i, b in enumerate(bins):
            if len(b) < count:
                continue
            while True:
                ci = np.random.choice(np.arange(len(b)))
                c = b[ci]
                if c not in selected:
                    break
            selected.add(c)
            print(len(selected), (bin_limits[i], bin_limits[i + 1]), c)
    config_dir = Path("configs/scenario.3/dave.tse.sample")
    config_dir.mkdir(exist_ok=True, parents=True)
    for transform in selected:
        drop_layers, scale_layers_1, scale_layers_2 = transform
        filename = config_dir / (
            "dave.D.%s.S.%s.S.%s.toml"
            % (
                ".".join(str(l) for l in drop_layers),
                ".".join(str(l) for l in scale_layers_1),
                ".".join(str(l) for l in scale_layers_2),
            )
        )
        with open(filename, "w+") as f:
            f.write(TEMPLATE)
            if len(drop_layers) > 0:
                f.write("[[distillation.strategies.drop_layer]]\n")
                f.write("layer_id=%s\n\n" % list(drop_layers))
            if len(scale_layers_1) > 0:
                f.write("[[distillation.strategies.scale_layer]]\n")
                f.write("factor=0.5\n")
                f.write("layer_id=%s\n\n" % list(scale_layers_1))
            if len(scale_layers_2) > 0:
                f.write("[[distillation.strategies.scale_layer]]\n")
                f.write("factor=0.5\n")
                f.write("layer_id=%s\n\n" % list(scale_layers_2))


if __name__ == "__main__":
    main()
