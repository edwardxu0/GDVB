# R4V: Refactoring for Verification

## Installation

The following command should install the necessary python environment. Requires python3.6+ and [venv](https://docs.python.org/3/library/venv.html) to be installed on the system.

```bash
./install.sh
```

To load the environment, run:

```bash
. .env.d/openenv.sh
```

## Network Info

To show information about an onnx model (i.e., its layers), use the following command.

```
python -m r4v info MODEL
```

For example, to see info about the layers in the ResNet-34 model, run:

```
python -m r4v info networks/resnet34/model.onnx
```

## Distillation

The following command will run distillation using the configuration defined in the provided configuration file. The distillation process outputs a model in [ONNX](https://github.com/onnx/onnx) format.

```bash
python -m r4v distill CONFIG_FILE
```

The configuration file is specified in the [TOML](https://github.com/toml-lang/toml) format. Some example configuration files are provided in the `configs` directory.

The available strategies are `drop_layer`, `scale_layer`, and `scale_input`. A `drop_layer` strategy can be specified as:

```toml
[[distillation.strategies.drop_layer]]
layer_id=[0,1]
```

Where `layer_id` specifies a list of indices, where each index is the index of the layer (in the original network) to be dropped.

A `scale_layer` strategy can be specified as:

```toml
[[distillation.strategies.scale_layer]]
layer_id=[0,1]
factor=0.5
```

Where `layer_id` is specified the same way as for `drop_layer`, and `factor` specifies the scale factor for the layers.

A `scale_input` strategy can be defined as:

```toml
[[distillation.strategies.scale_input]]
factor=[1.0, 1.0, 0.5, 0.5]
```

Where `factor` is a list of scale factors, one for each dimension of the input.
