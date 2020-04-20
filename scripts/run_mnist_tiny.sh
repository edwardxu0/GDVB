#!/bin/bash

python -m gdvb configs/mnist_tiny.toml gen_ca 10
python -m gdvb configs/mnist_tiny.toml train 10
python -m gdvb configs/mnist_tiny.toml gen_props 10
python -m gdvb configs/mnist_tiny.toml verify 10
python -m gdvb configs/mnist_tiny.toml analyze 10
