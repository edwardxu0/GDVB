import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from .. import logging


def _get_tf_pads(in_height, in_width, kernel_shape, strides):
    out_height = np.ceil(float(in_height) / float(strides[0]))
    out_width = np.ceil(float(in_width) / float(strides[1]))
    pad_along_height = max(
        (out_height - 1) * strides[0] + kernel_shape[0] - in_height, 0
    )
    pad_along_width = max((out_width - 1) * strides[1] + kernel_shape[1] - in_width, 0)
    pad_top = pad_along_height // 2
    pad_bottom = pad_along_height - pad_top
    pad_left = pad_along_width // 2
    pad_right = pad_along_width - pad_left
    return int(pad_top), int(pad_bottom), int(pad_left), int(pad_right)


class PytorchBatchNorm(nn.Module):
    def __init__(self, mean, var, weight, bias, momentum, eps):
        super().__init__()
        self.register_buffer("mean", torch.from_numpy(mean))
        self.register_buffer("var", torch.from_numpy(var))
        self.register_buffer("weight", torch.from_numpy(weight))
        self.register_buffer("bias", torch.from_numpy(bias))
        self.momentum = momentum
        self.eps = eps

    def infer_output_size(self, input_size):
        return input_size

    def forward(self, x):
        return F.batch_norm(
            x,
            self.mean,
            self.var,
            self.weight,
            self.bias,
            training=False,
            momentum=self.momentum,
            eps=self.eps,
        )


class PytorchReshape(nn.Module):
    def __init__(self, shape):
        super(PytorchReshape, self).__init__()
        self.shape = shape

    def forward(self, x):
        return x.contiguous().view(*self.shape)


class PytorchFlatten(nn.Module):
    def forward(self, x):
        return x.contiguous().view(-1, np.product(x.size()[1:]))


class PytorchResidualBlock(nn.Module):
    def __init__(self, layer, maintain_weights=False):
        super(PytorchResidualBlock, self).__init__()
        logger = logging.getLogger(__name__)

        self.dropped_operations = layer.dropped_operations

        i = 0
        padding = 0
        pads = layer.conv_pads[i]
        self.conv_padding_1 = None
        if layer.conv_padding[i] == "VALID":
            assert layer.pads == (0, 0, 0, 0)
            padding = (0, 0)
        elif maintain_weights and not layer.modified:
            if pads[0] == pads[1] and pads[2] == pads[3]:
                padding = (pads[0], pads[2])
            else:
                self.conv_padding_1 = nn.ZeroPad2d(pads)
        elif layer.conv_padding[i] == "SAME":
            tf_pads = _get_tf_pads(
                layer.input_shape[2],
                layer.input_shape[3],
                layer.conv_kernel_shape[i],
                layer.conv_strides[i],
            )
            if pads != tf_pads:
                logger.warning(
                    "Converting padding to tensorflow padding: %s -> %s", pads, tf_pads
                )
                pads = tf_pads
            if pads[0] == pads[1] and pads[2] == pads[3]:
                padding = (pads[0], pads[2])
            else:
                self.conv_padding_1 = nn.ZeroPad2d(pads)
        else:
            raise ValueError("Unknown padding type: %s", layer.conv_padding[i])
        self.conv_layer_1 = nn.Conv2d(
            layer.in_features[i],
            layer.out_features[i],
            layer.conv_kernel_shape[i],
            stride=tuple(layer.conv_strides[i]),
            padding=padding,
        )
        if not layer.modified and maintain_weights:
            self.conv_layer_1.weight.data = torch.from_numpy(layer.conv_weight[i])
            self.conv_layer_1.bias.data = torch.from_numpy(layer.conv_bias[i])
        elif maintain_weights:
            raise ValueError("Cannot maintain weights of modified layer.")
        if "batchnormalization" not in self.dropped_operations:
            self.bn_layer_1 = torch.nn.BatchNorm2d(
                layer.out_features[i],
                eps=layer.bn_attributes[i]["epsilon"],
                momentum=layer.bn_attributes[i]["momentum"],
            )
            if not layer.modified and maintain_weights:
                self.bn_layer_1 = PytorchBatchNorm(
                    layer.bn_mean[i],
                    layer.bn_var[i],
                    layer.bn_weight[i],
                    layer.bn_bias[i],
                    layer.bn_attributes[i]["momentum"],
                    layer.bn_attributes[i]["epsilon"],
                )
        self.relu_1 = nn.ReLU()

        i = 1
        if self.conv_padding_1 is not None:
            current_shape = nn.Sequential(self.conv_padding_1, self.conv_layer_1)(
                torch.ones(layer.input_shape)
            ).shape
        else:
            current_shape = nn.Sequential(self.conv_layer_1)(
                torch.ones(layer.input_shape)
            ).shape
        padding = 0
        pads = layer.conv_pads[i]
        self.conv_padding_2 = None
        if layer.conv_padding[i] == "VALID":
            assert layer.pads == (0, 0, 0, 0)
            padding = (0, 0)
        elif maintain_weights and not layer.modified:
            if pads[0] == pads[1] and pads[2] == pads[3]:
                padding = (pads[0], pads[2])
            else:
                self.conv_padding_2 = nn.ZeroPad2d(pads)
        elif layer.conv_padding[i] == "SAME":
            tf_pads = _get_tf_pads(
                current_shape[2],
                current_shape[3],
                layer.conv_kernel_shape[i],
                layer.conv_strides[i],
            )
            if pads != tf_pads:
                logger.warning(
                    "Converting padding to tensorflow padding: %s -> %s", pads, tf_pads
                )
                pads = tf_pads
            if pads[0] == pads[1] and pads[2] == pads[3]:
                padding = (pads[0], pads[2])
            else:
                self.conv_padding_2 = nn.ZeroPad2d(pads)
        else:
            raise ValueError("Unknown padding type: %s", layer.conv_padding[i])
        self.conv_layer_2 = nn.Conv2d(
            layer.in_features[i],
            layer.out_features[i],
            layer.conv_kernel_shape[i],
            stride=tuple(layer.conv_strides[i]),
            padding=padding,
        )
        if not layer.modified and maintain_weights:
            self.conv_layer_2.weight.data = torch.from_numpy(layer.conv_weight[i])
            self.conv_layer_2.bias.data = torch.from_numpy(layer.conv_bias[i])
        if "batchnormalization" not in self.dropped_operations:
            self.bn_layer_2 = torch.nn.BatchNorm2d(
                layer.out_features[i],
                eps=layer.bn_attributes[i]["epsilon"],
                momentum=layer.bn_attributes[i]["momentum"],
            )
            if not layer.modified and maintain_weights:
                self.bn_layer_2 = PytorchBatchNorm(
                    layer.bn_mean[i],
                    layer.bn_var[i],
                    layer.bn_weight[i],
                    layer.bn_bias[i],
                    layer.bn_attributes[i]["momentum"],
                    layer.bn_attributes[i]["epsilon"],
                )

        self.operations = []
        if self.conv_padding_1 is not None:
            self.operations.append(self.conv_padding_1)
        self.operations.append(self.conv_layer_1)
        if "batchnormalization" not in self.dropped_operations:
            self.operations.append(self.bn_layer_1)
        self.operations.append(self.relu_1)
        if self.conv_padding_2 is not None:
            self.operations.append(self.conv_padding_2)
        self.operations.append(self.conv_layer_2)
        if "batchnormalization" not in self.dropped_operations:
            self.operations.append(self.bn_layer_2)
        self.operations = tuple(self.operations)

        self.downsample = layer.downsample
        if self.downsample:
            padding = 0
            pads = layer.ds_conv_pads
            self.downsample_conv_padding = None
            if pads == (0, 0, 0, 0):
                padding = (0, 0)
            elif pads[0] == pads[1] and pads[2] == pads[3]:
                padding = (pads[0], pads[2])
            else:
                self.downsample_conv_padding = nn.ZeroPad2d(pads)
                self.operations += (self.downsample_conv_padding,)
            self.downsample_conv = nn.Conv2d(
                layer.in_features[0],
                layer.out_features[1],
                layer.ds_conv_kernel_shape,
                stride=tuple(layer.ds_conv_strides),
                padding=padding,
            )
            if not layer.modified and maintain_weights:
                self.downsample_conv.weight.data = torch.from_numpy(
                    layer.ds_conv_weight
                )
                self.downsample_conv.bias.data = torch.from_numpy(layer.ds_conv_bias)
            self.operations += (self.downsample_conv,)
            if "batchnormalization" not in self.dropped_operations:
                if not layer.modified and maintain_weights:
                    self.downsample_bn = PytorchBatchNorm(
                        layer.ds_bn_mean,
                        layer.ds_bn_var,
                        layer.ds_bn_weight,
                        layer.ds_bn_bias,
                        layer.ds_bn_attributes["momentum"],
                        layer.ds_bn_attributes["epsilon"],
                    )
                else:
                    self.downsample_bn = torch.nn.BatchNorm2d(
                        layer.out_features[i],
                        eps=layer.ds_bn_attributes["epsilon"],
                        momentum=layer.ds_bn_attributes["momentum"],
                    )
                self.operations += (self.downsample_bn,)

        self.input_names = layer.input_names
        self.output_names = layer.output_names
        self.input_shape = layer.input_shape
        self.output_shape = layer.output_shape

        self.use_residual = layer.use_residual

    def num_neurons(self, device=torch.device("cpu")):
        neuron_count = 0
        x = torch.ones(self.input_shape).to(device)
        residual = x

        if self.downsample:
            if self.downsample_conv_padding is not None:
                residual = self.downsample_conv_padding(residual)
            residual = self.downsample_conv(residual)

        if self.conv_padding_1 is not None:
            x = self.conv_padding_1(x)
        x = self.conv_layer_1(x)
        neuron_count += np.product(x.size())

        if self.conv_padding_2 is not None:
            x = self.conv_padding_2(x)
        x = self.conv_layer_2(x)
        neuron_count += np.product(x.size())

        if self.downsample and self.use_residual:
            neuron_count += np.product(residual.size())
        return neuron_count

    def infer_output_size(self, input_size):
        x = torch.zeros(input_size)
        if self.conv_padding_1 is not None:
            x = self.conv_padding_1(x)
        x = self.conv_layer_1(x)
        if self.conv_padding_2 is not None:
            x = self.conv_padding_2(x)
        x = self.conv_layer_2(x)
        return x.size()

    def forward(self, x):
        residual = x
        if self.downsample:
            if self.downsample_conv_padding is not None:
                residual = self.downsample_conv_padding(residual)
            residual = self.downsample_conv(residual)
            if "batchnormalization" not in self.dropped_operations:
                residual = self.downsample_bn(residual)

        if self.conv_padding_1 is not None:
            x = self.conv_padding_1(x)
        x = self.conv_layer_1(x)
        if "batchnormalization" not in self.dropped_operations:
            x = self.bn_layer_1(x)
        x = self.relu_1(x)

        if self.conv_padding_2 is not None:
            x = self.conv_padding_2(x)
        x = self.conv_layer_2(x)
        if "batchnormalization" not in self.dropped_operations:
            x = self.bn_layer_2(x)

        if self.use_residual:
            return x + residual
        else:
            return x


class PytorchResidualBlockV2(nn.Module):
    def __init__(self, layer, maintain_weights=False):
        super(PytorchResidualBlockV2, self).__init__()
        logger = logging.getLogger(__name__)

        self.dropped_operations = layer.dropped_operations

        i = 0
        if "batchnormalization" not in self.dropped_operations:
            self.bn_layer_1 = torch.nn.BatchNorm2d(
                layer.in_features[i],
                eps=layer.bn_attributes[i]["epsilon"],
                momentum=layer.bn_attributes[i]["momentum"],
            )
            if not layer.modified and maintain_weights:
                self.bn_layer_1 = PytorchBatchNorm(
                    layer.bn_mean[i],
                    layer.bn_var[i],
                    layer.bn_weight[i],
                    layer.bn_bias[i],
                    layer.bn_attributes[i]["momentum"],
                    layer.bn_attributes[i]["epsilon"],
                )
        self.relu_1 = nn.ReLU()
        padding = 0
        pads = layer.conv_pads[i]
        self.conv_padding_1 = None
        if layer.conv_padding[i] == "VALID":
            assert pads == (0, 0, 0, 0)
            padding = (0, 0)
        elif maintain_weights and not layer.modified:
            if pads[0] == pads[1] and pads[2] == pads[3]:
                padding = (pads[0], pads[2])
            else:
                self.conv_padding_1 = nn.ZeroPad2d(pads)
        elif layer.conv_padding[i] == "SAME":
            tf_pads = _get_tf_pads(
                layer.input_shape[2],
                layer.input_shape[3],
                layer.conv_kernel_shape[i],
                layer.conv_strides[i],
            )
            if pads != tf_pads:
                logger.warning(
                    "Converting padding to tensorflow padding: %s -> %s", pads, tf_pads
                )
                pads = tf_pads
            if pads[0] == pads[1] and pads[2] == pads[3]:
                padding = (pads[0], pads[2])
            else:
                self.conv_padding_1 = nn.ZeroPad2d(pads)
        else:
            raise ValueError("Unknown padding type: %s", layer.conv_padding[i])
        self.conv_layer_1 = nn.Conv2d(
            layer.in_features[i],
            layer.out_features[i],
            layer.conv_kernel_shape[i],
            stride=tuple(layer.conv_strides[i]),
            padding=padding,
        )
        if not layer.modified and maintain_weights:
            self.conv_layer_1.weight.data = torch.from_numpy(layer.conv_weight[i])
            self.conv_layer_1.bias.data = torch.from_numpy(layer.conv_bias[i])
        elif maintain_weights:
            raise ValueError("Cannot maintain weights of modified layer.")

        i = 1
        if self.conv_padding_1 is not None:
            current_shape = nn.Sequential(self.conv_padding_1, self.conv_layer_1)(
                torch.ones(layer.input_shape)
            ).shape
        else:
            current_shape = nn.Sequential(self.conv_layer_1)(
                torch.ones(layer.input_shape)
            ).shape
        if "batchnormalization" not in self.dropped_operations:
            self.bn_layer_2 = torch.nn.BatchNorm2d(
                layer.in_features[i],
                eps=layer.bn_attributes[i]["epsilon"],
                momentum=layer.bn_attributes[i]["momentum"],
            )
            if not layer.modified and maintain_weights:
                self.bn_layer_2 = PytorchBatchNorm(
                    layer.bn_mean[i],
                    layer.bn_var[i],
                    layer.bn_weight[i],
                    layer.bn_bias[i],
                    layer.bn_attributes[i]["momentum"],
                    layer.bn_attributes[i]["epsilon"],
                )
        self.relu_2 = nn.ReLU()
        padding = 0
        pads = layer.conv_pads[i]
        self.conv_padding_2 = None
        if layer.conv_padding[i] == "VALID":
            assert pads == (0, 0, 0, 0)
            padding = (0, 0)
        elif maintain_weights and not layer.modified:
            if pads[0] == pads[1] and pads[2] == pads[3]:
                padding = (pads[0], pads[2])
            else:
                self.conv_padding_2 = nn.ZeroPad2d(pads)
        elif layer.conv_padding[i] == "SAME":
            tf_pads = _get_tf_pads(
                current_shape[2],
                current_shape[3],
                layer.conv_kernel_shape[i],
                layer.conv_strides[i],
            )
            if pads != tf_pads:
                logger.warning(
                    "Converting padding to tensorflow padding: %s -> %s", pads, tf_pads
                )
                pads = tf_pads
            if pads[0] == pads[1] and pads[2] == pads[3]:
                padding = (pads[0], pads[2])
            else:
                self.conv_padding_2 = nn.ZeroPad2d(pads)
        else:
            raise ValueError("Unknown padding type: %s", layer.conv_padding[i])
        self.conv_layer_2 = nn.Conv2d(
            layer.in_features[i],
            layer.out_features[i],
            layer.conv_kernel_shape[i],
            stride=tuple(layer.conv_strides[i]),
            padding=padding,
        )
        if not layer.modified and maintain_weights:
            self.conv_layer_2.weight.data = torch.from_numpy(layer.conv_weight[i])
            self.conv_layer_2.bias.data = torch.from_numpy(layer.conv_bias[i])

        self.operations = []
        if "batchnormalization" not in self.dropped_operations:
            self.operations.append(self.bn_layer_1)
        self.operations.append(self.relu_1)
        if self.conv_padding_1 is not None:
            self.operations.append(self.conv_padding_1)
        self.operations.append(self.conv_layer_1)
        if "batchnormalization" not in self.dropped_operations:
            self.operations.append(self.bn_layer_2)
        self.operations.append(self.relu_2)
        if self.conv_padding_2 is not None:
            self.operations.append(self.conv_padding_2)
        self.operations.append(self.conv_layer_2)
        self.operations = tuple(self.operations)

        self.downsample = layer.downsample
        if self.downsample:
            self.downsample_op = self.downsample.as_pytorch(
                maintain_weights=maintain_weights
            )
            self.operations += (self.downsample_op,)

        self.input_names = layer.input_names
        self.output_names = layer.output_names
        self.input_shape = layer.input_shape
        self.output_shape = layer.output_shape

        self.use_residual = layer.use_residual

    def num_neurons(self, device=torch.device("cpu")):
        neuron_count = 0
        x = torch.ones(self.input_shape).to(device)
        residual = x

        if self.downsample:
            residual = self.downsample_conv(x)

        if self.conv_padding_1 is not None:
            x = self.conv_padding_1(x)
        x = self.conv_layer_1(x)
        neuron_count += np.product(x.size())

        if self.conv_padding_2 is not None:
            x = self.conv_padding_2(x)
        x = self.conv_layer_2(x)
        neuron_count += np.product(x.size())

        if self.downsample and self.use_residual:
            neuron_count += np.product(residual.size())
        return neuron_count

    def infer_output_size(self, input_size):
        x = torch.zeros(input_size)
        if self.conv_padding_1 is not None:
            x = self.conv_padding_1(x)
        x = self.conv_layer_1(x)
        if self.conv_padding_2 is not None:
            x = self.conv_padding_2(x)
        x = self.conv_layer_2(x)
        return x.size()

    def forward(self, x):
        residual = x
        if self.downsample and self.use_residual:
            residual = self.downsample_op(x)

        if "batchnormalization" not in self.dropped_operations:
            x = self.bn_layer_1(x)
        x = self.relu_1(x)
        if self.conv_padding_1 is not None:
            x = self.conv_padding_1(x)
        x = self.conv_layer_1(x)

        if "batchnormalization" not in self.dropped_operations:
            x = self.bn_layer_2(x)
        x = self.relu_2(x)
        if self.conv_padding_2 is not None:
            x = self.conv_padding_2(x)
        x = self.conv_layer_2(x)

        if self.use_residual:
            return x + residual
        else:
            return x


class PytorchSequential(nn.Module):
    def __init__(self, layer, *operations):
        super(PytorchSequential, self).__init__()
        self.operations = operations
        self.sequential = nn.Sequential(*operations)
        self.input_names = layer.input_names
        self.output_names = layer.output_names
        self.input_shape = layer.input_shape
        self.output_shape = layer.output_shape

    def infer_output_size(self, input_size):
        dummy_input = torch.zeros(input_size)
        return self.eval()(dummy_input).size()

    def num_neurons(self, device=torch.device("cpu")):
        neuron_count = 0
        x = torch.ones(self.input_shape).to(device)
        for layer in self.operations:
            if layer.__class__ in [
                PytorchFlatten,
                PytorchTranspose,
                nn.ReLU,
                nn.BatchNorm2d,
                PytorchBatchNorm,
                PytorchAtan,
                PytorchMultiply,
            ]:
                continue
            elif layer.__class__ in [
                PytorchResidualBlock,
                PytorchResidualBlockV2,
                PytorchSequential,
            ]:
                num_neurons = layer.num_neurons(device=device)
                neuron_count += num_neurons
                x = torch.ones(layer.output_shape).to(device)
            elif layer.__class__ in [nn.Sequential]:
                if len(layer) == 0:
                    continue
                x = layer(x)
                num_neurons = np.product(x.size())
                neuron_count += num_neurons
            else:
                x = layer(x)
                num_neurons = np.product(x.size())
                neuron_count += num_neurons
        return neuron_count

    def forward(self, x):
        return self.sequential(x)


class PytorchConcat(nn.Module):
    def __init__(self, layer, *operations):
        super(PytorchConcat, self).__init__()
        self.operations = operations
        self.module_list = nn.ModuleList(operations)
        self.axis = layer.concat_axis

        self.input_names = layer.input_names
        self.output_names = layer.output_names
        self.input_shape = layer.input_shape
        self.output_shape = layer.output_shape

    def infer_output_size(self, input_size):
        dummy_input = torch.zeros(input_size)
        return self.eval()(dummy_input).size()

    def num_neurons(self, device=torch.device("cpu")):
        neuron_count = 0
        x = torch.ones(self.input_shape).to(device)
        for layer in self.operations:
            if layer.__class__ in [
                PytorchFlatten,
                PytorchTranspose,
                nn.ReLU,
                nn.BatchNorm2d,
                PytorchBatchNorm,
            ]:
                continue
            elif layer.__class__ in [PytorchResidualBlock, PytorchSequential]:
                neuron_count += layer.num_neurons(device=device)
                x = torch.ones(layer.output_shape).to(device)
            else:
                x = layer(x)
                neuron_count += np.product(x.size())
        return neuron_count

    def forward(self, x):
        return torch.cat([module(x) for module in self.module_list], dim=self.axis)


class PytorchTranspose(nn.Module):
    def __init__(self, *dims):
        super(PytorchTranspose, self).__init__()
        self.dims = dims

    def forward(self, x):
        return x.permute(*self.dims)


class PytorchAtan(nn.Module):
    def forward(self, x):
        return torch.atan(x)


class PytorchMultiply(nn.Module):
    def __init__(self, value):
        super(PytorchMultiply, self).__init__()
        self.register_buffer("value", torch.from_numpy(value))

    def forward(self, x):
        return x * self.value
