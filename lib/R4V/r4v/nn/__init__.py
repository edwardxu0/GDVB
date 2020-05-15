import numpy as np
import onnx
import onnx.utils
import torch
import torch.nn as nn

from collections import defaultdict, Iterable
from functools import partial
from itertools import chain
from onnx import numpy_helper

from .. import logging
from .pytorch import (
    PytorchAtan,
    PytorchBatchNorm,
    PytorchConcat,
    PytorchFlatten,
    PytorchMultiply,
    PytorchReshape,
    PytorchResidualBlock,
    PytorchResidualBlockV2,
    PytorchSequential,
    PytorchTranspose,
)

ONNX_TO_NUMPY_DTYPE = {
    onnx.TensorProto.DOUBLE: np.float64,
    onnx.TensorProto.FLOAT16: np.float16,
    onnx.TensorProto.FLOAT: np.float32,
    onnx.TensorProto.INT16: np.int16,
    onnx.TensorProto.INT32: np.int32,
    onnx.TensorProto.INT64: np.int64,
}


def as_numpy(node):
    if isinstance(node, onnx.TensorProto):
        return numpy_helper.to_array(node)
    elif isinstance(node, onnx.NodeProto):
        return numpy_helper.to_array(node.attribute[0].t)
    elif isinstance(node, onnx.AttributeProto):
        if node.type == onnx.AttributeProto.FLOAT:
            return np.float(node.f)
        elif node.type == onnx.AttributeProto.INT:
            return np.int(node.i)
        elif node.type == onnx.AttributeProto.INTS:
            return np.asarray(node.ints)
        elif node.type == onnx.AttributeProto.STRING:
            return node.s.decode("utf-8")
        raise ValueError("Unknown attribute type: %s" % (node,))
    else:
        raise ValueError("Unknown node type: %s" % type(node))


def fix_padding(pads):
    # return padding as left, right, top, bottom
    logger = logging.getLogger(__name__)
    if len(pads) == 2:
        pads = (int(pads[0]), int(pads[0]), int(pads[1]), int(pads[1]))
    elif len(pads) == 4:
        pads = (int(pads[0]), int(pads[2]), int(pads[1]), int(pads[3]))
    elif len(pads) == 8:
        assert pads[0] == 0 and pads[1] == 0
        assert pads[4] == 0 and pads[5] == 0
        pads = (int(pads[2]), int(pads[6]), int(pads[3]), int(pads[7]))
    else:
        raise AssertionError(
            "Unsupported length for padding values (%s): %s" % (pads, len(pads))
        )
    return pads


def get_tf_pads(in_height, in_width, kernel_shape, strides):
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


def as_implicit_padding(pads):
    if tuple(pads) == (0, 0, 0, 0):
        return "VALID"
    else:
        return "SAME"


def get_conv_parameters(conv_op, pad_op=None):
    logger = logging.getLogger(__name__)

    conv_weight = as_numpy(conv_op[2])
    if len(conv_op) > 3:
        conv_bias = as_numpy(conv_op[3])
    else:
        conv_bias = np.zeros((conv_weight.shape[0],), dtype=np.float32)

    conv_attributes = {a.name: as_numpy(a) for a in conv_op[0].attribute}
    if "auto_pad" in conv_attributes:
        logger.warning("auto_pad is deprecated: (%s)", conv_attributes["auto_pad"])
    if "dilations" in conv_attributes:
        assert np.all(
            conv_attributes["dilations"] == 1
        ), "Only 1-dilation convolutions are currently supported."
    if "group" in conv_attributes:
        assert conv_attributes["group"] == 1

    kernel_shape = tuple(conv_attributes.get("kernel_shape", conv_weight.shape[2:]))
    assert kernel_shape == tuple(conv_weight.shape[2:])
    strides = tuple(conv_attributes.get("strides", (1, 1)))
    pads = fix_padding(tuple(conv_attributes.get("pads", (0, 0, 0, 0))))
    if pad_op is None:
        return conv_weight, conv_bias, kernel_shape, strides, pads
    assert pads == (0, 0, 0, 0)
    padding_attributes = {a.name: as_numpy(a) for a in pad_op[0].attribute}
    pads = fix_padding(tuple(padding_attributes["pads"]))
    assert padding_attributes.get("mode", "constant") == "constant"
    assert padding_attributes.get("value", 0.0) == 0.0

    return conv_weight, conv_bias, kernel_shape, strides, pads


def get_batchnorm_parameters(bn_op):
    attributes = {a.name: as_numpy(a) for a in bn_op[0].attribute}
    assert "epsilon" in attributes
    assert "momentum" in attributes
    scale = as_numpy(bn_op[2])
    bias = as_numpy(bn_op[3])
    mean = as_numpy(bn_op[4])
    var = as_numpy(bn_op[5])
    return attributes, scale, bias, mean, var


def load_network(config):
    return LayerConverter().convert(
        OnnxModel(config["model"]),
        input_shape=config["input_shape"],
        input_format=config.get("input_format", "NCHW"),
    )


class Layer:
    PATTERNS = []

    def __init__(self, node_list):
        self.node_list = node_list
        self.dropped = False
        self.modified = False
        self.shape_preserving = False
        self._inputs = []
        self._outputs = []
        self._input_shape = None
        self._output_shape = None
        self.input_names = [list(self.node_list[0][0].input)[0]]
        self.output_names = list(self.node_list[-1][0].output)

    def __eq__(self, other):
        return repr(self) == repr(other)

    def __hash__(self):
        return id(self)

    @property
    def inputs(self):
        return self._inputs

    @inputs.setter
    def inputs(self, value):
        if not isinstance(value, list):
            value = [value]
        self._inputs = value
        assert len(self._inputs) <= 1
        if len(self._inputs) == 0:
            assert self.dropped
            return
        self.input_shape = self._inputs[0].output_shape
        for layer in self._inputs:
            layer._outputs.append(self)

    @property
    def input_shape(self):
        return self._input_shape

    @input_shape.setter
    def input_shape(self, value):
        self._input_shape = value
        old_output_shape = np.array(self.output_shape)
        self.update_output_shape()
        if np.any(np.array(self.output_shape) != old_output_shape):
            for output in self.outputs:
                output.input_shape = self.output_shape

    @property
    def outputs(self):
        return self._outputs

    @outputs.setter
    def outputs(self, value):
        self._outputs = value

    @property
    def output_shape(self):
        return self._output_shape

    @output_shape.setter
    def output_shape(self, value):
        self._output_shape = value

    def update_output_shape(self):
        output_shape = self.as_pytorch().infer_output_size(self._input_shape)
        self.output_shape = tuple(output_shape)

    def scale(self, factor):
        raise ValueError(
            "The scale operation is not supported for layer type: %s"
            % self.__class__.__name__
        )

    def as_pytorch(self, maintain_weights=False):
        raise NotImplementedError(
            "The as_pytorch operation is not supported for layer type: %s"
            % self.__class__.__name__
        )


class Rescalable(Layer):
    def scale(self, factor):
        raise NotImplementedError()


class Droppable(Layer):
    def drop(self):
        self.dropped = True
        self.modified = True
        for layer in self.inputs:
            layer.outputs.remove(self)
        for output in self.outputs:
            output.inputs = self.inputs
        self.inputs = []
        self.outputs = []


class DroppableOperations(Layer):
    def drop_operation(self, op_type):
        raise NotImplementedError()


class Linearizable(Layer):
    def linearize(self):
        raise NotImplementedError()


class ResidualConnection(Linearizable):
    pass


class Input(Layer):
    def __init__(self, input_shape, input_format="NCHW"):
        self.dropped = False
        self.shape_preserving = False
        self._inputs = None
        self._outputs = []
        self.input_format = input_format
        self._input_shape = input_shape
        self._output_shape = self.input_shape

    @Layer.input_shape.setter
    def input_shape(self, value):
        self._input_shape = value
        self._output_shape = value
        for output in self.outputs:
            output.input_shape = self.output_shape


class FullyConnected(Rescalable, Droppable):
    PATTERNS = []

    def __init__(self, node_list):
        super().__init__(node_list)
        node_list = [
            node for node in node_list if node[0].op_type.lower() in ["relu", "dropout"]
        ]
        if not hasattr(self, "weight"):
            self.weight = None
            self.bias = None
        self.in_features = self.weight.shape[1]
        self.out_features = self.bias.shape[0]

        self.activation = None
        if len(node_list) > 0:
            assert (
                len(node_list[0][1:]) == 1
            ), "relu should not have more than one input"
            self.activation = node_list[0][0].op_type.lower()
        self.dropout = None
        if len(node_list) > 1:
            assert (
                len(node_list[1][1:]) == 1
            ), "dropout should not have more than one input"
            self.dropout = node_list[1][0].attribute[0].f

    def __repr__(self):
        return (
            f"FullyConnected({self.in_features}, "
            f"{self.out_features}, "
            f"activation={self.activation}, "
            f"dropout={self.dropout})"
        )

    @Layer.input_shape.setter
    def input_shape(self, value):
        assert value[0] == 1
        self.in_features = value[1]
        self._input_shape = value
        self._output_shape = (1, self.out_features)

    def scale(self, factor):
        self.modified = True
        #self.out_features = int(self.out_features * factor)
        self.out_features = int(round(self.out_features * factor))
        output_shape = list(self.output_shape)
        output_shape[1] = self.out_features
        self.output_shape = tuple(output_shape)
        for output in self._outputs:
            output.input_shape = self.output_shape
        return self

    def as_pytorch(self, maintain_weights=False):
        if maintain_weights and self.modified:
            raise ValueError("Cannot maintain weights of modified layer.")
        if not self.dropped:
            linear_layer = nn.Linear(self.in_features, self.out_features)
            if maintain_weights and not self.modified:
                linear_layer.weight.data = torch.from_numpy(self.weight)
                linear_layer.bias.data = torch.from_numpy(self.bias)
            elif maintain_weights:
                raise ValueError("Cannot maintain weights of modified layer.")
            if self.dropout is not None:
                assert self.activation == "relu"
                return PytorchSequential(
                    self, linear_layer, nn.ReLU(), nn.Dropout(self.dropout)
                )
            elif self.activation == "relu":
                return PytorchSequential(self, linear_layer, nn.ReLU())
            else:
                return PytorchSequential(self, linear_layer)


class MatmulFullyConnected(FullyConnected):
    PATTERNS = [
        ["matmul", "add", "relu", "dropout"],
        ["matmul", "add", "relu"],
        ["matmul", "add"],
        ["matmul"],
    ]

    def __init__(self, node_list):
        if isinstance(node_list[0][1], onnx.TensorProto):
            self.weight = as_numpy(node_list[0][1])
        elif isinstance(node_list[0][2], onnx.TensorProto):
            self.weight = as_numpy(node_list[0][2]).T
        else:
            raise ValueError(
                "A constant weight value was expected for the fully connected layer."
            )
        if len(node_list) > 1:
            assert node_list[1][0].op_type.lower() == "add"
            self.bias = as_numpy(node_list[1][2])
        else:
            self.bias = np.zeros((self.weight.shape[1],))

        super().__init__(node_list)


class GemmFullyConnected(FullyConnected):
    PATTERNS = [["gemm", "relu", "dropout"], ["gemm", "relu"], ["gemm"]]

    def __init__(self, node_list):
        if isinstance(node_list[0][1], onnx.TensorProto):
            self.weight = as_numpy(node_list[0][1])
        elif isinstance(node_list[0][2], onnx.TensorProto):
            self.weight = as_numpy(node_list[0][2])
        else:
            raise ValueError(
                "A constant weight value was expected for the fully connected layer."
            )
        self.bias = as_numpy(node_list[0][3])

        self.gemm_attributes = {"alpha": 1.0, "beta": 1.0, "transA": 0, "transB": 0}
        self.gemm_attributes.update(
            {a.name: as_numpy(a) for a in node_list[0][0].attribute}
        )
        assert self.gemm_attributes["alpha"] == 1.0
        assert self.gemm_attributes["beta"] == 1.0

        if self.gemm_attributes["transB"] == 0:
            self.weight = self.weight.T

        super().__init__(node_list)


class Dropout(Droppable):
    PATTERNS = [["dropout"]]

    def __init__(self, node_list):
        super().__init__(node_list)
        assert len(node_list) == 1
        assert len(node_list[0][1:]) == 1, "dropout should not have more than one input"
        self.zero_prob = node_list[0][0].attribute[0].f

    def __repr__(self):
        return f"Dropout({self.zero_prob})"

    def as_pytorch(self, maintain_weights=False):
        if not self.dropped:
            return PytorchSequential(self, nn.Dropout(self.zero_prob))


class Convolutional(Rescalable, Droppable):
    PATTERNS = [["conv", "relu"], ["pad", "conv", "relu"], ["conv"], ["pad", "conv"]]

    def __init__(self, node_list):
        super().__init__(node_list)
        logger = logging.getLogger(__name__)

        conv_op = node_list[0]
        pad_op = None
        if node_list[0][0].op_type.lower() == "pad":
            pad_op = node_list[0]
            assert len(pad_op) == 2
            conv_op = node_list[1]

        self.weight, self.bias, self.kernel_shape, self.strides, self.pads = get_conv_parameters(
            conv_op, pad_op=pad_op
        )
        self.padding = as_implicit_padding(self.pads)
        self.in_features = self.weight.shape[1]
        self.out_features = self.bias.shape[0]

        self.activation = node_list[-1][0].op_type.lower()
        if self.activation != "relu":
            self.activation = None

    def __repr__(self):
        return (
            f"Convolutional({self.in_features}, "
            f"{self.out_features}, "
            f"kernel_shape={self.kernel_shape}, "
            f"strides={self.strides}, "
            f"padding={self.padding}, "
            f"activation={self.activation})"
        )

    @Layer.input_shape.setter
    def input_shape(self, value):
        self.in_features = value[1]
        Layer.input_shape.fset(self, value)

    def scale(self, factor):
        self.modified = True
        #self.out_features = int(self.out_features * factor)
        self.out_features = int(round(self.out_features * factor))
        self.update_output_shape()
        for output in self._outputs:
            output.input_shape = self.output_shape
        return self

    def scale_stride(self, factor):
        self.modified = True
        self.strides = tuple(int(x * factor) for x in self.strides)
        self.update_output_shape()
        for output in self._outputs:
            output.input_shape = self.output_shape
        return self

    def replace_padding(self, padding):
        self.modified = True
        padding = padding.upper()
        if padding not in ["SAME", "VALID"]:
            raise ValueError(f"Unknown padding type: {padding}")
        self.padding = padding
        self.update_output_shape()
        for output in self._outputs:
            output.input_shape = self.output_shape
        return self

    def as_pytorch(self, maintain_weights=False):
        logger = logging.getLogger(__name__)
        if maintain_weights and self.modified:
            raise ValueError("Cannot maintain weights of modified layer.")
        if not self.dropped:
            padding = 0
            conv_padding_layer = nn.Sequential()
            if maintain_weights and not self.modified:
                if self.pads == (0, 0, 0, 0):
                    padding = (0, 0)
                elif self.pads[0] == self.pads[1] and self.pads[2] == self.pads[3]:
                    padding = (self.pads[0], self.pads[2])
                else:
                    conv_padding_layer = nn.ZeroPad2d(self.pads)
            elif self.padding == "VALID":
                padding = (0, 0)
            elif self.padding == "SAME":
                tf_pads = get_tf_pads(
                    self.input_shape[2],
                    self.input_shape[3],
                    self.kernel_shape,
                    self.strides,
                )
                pads = tf_pads
                if pads[0] == pads[1] and pads[2] == pads[3]:
                    padding = (pads[0], pads[2])
                else:
                    conv_padding_layer = nn.ZeroPad2d(pads)
            conv_layer = nn.Conv2d(
                self.in_features,
                self.out_features,
                self.kernel_shape,
                stride=tuple(self.strides),
                padding=padding,
            )
            if maintain_weights and not self.modified:
                conv_layer.weight.data = torch.from_numpy(self.weight)
                conv_layer.bias.data = torch.from_numpy(self.bias)
            elif maintain_weights:
                raise ValueError("Cannot maintain weights of modified layer.")
            if self.activation == "relu":
                return PytorchSequential(
                    self, conv_padding_layer, conv_layer, nn.ReLU()
                )
            else:
                return PytorchSequential(self, conv_padding_layer, conv_layer)
        return PytorchSequential(self)


class MaxPool(Droppable):
    PATTERNS = [["maxpool"]]

    def __init__(self, node_list):
        logger = logging.getLogger(__name__)
        super().__init__(node_list)
        maxpool_node = node_list[0][0]
        self.maxpool_attributes = {a.name: as_numpy(a) for a in maxpool_node.attribute}
        if "auto_pad" in self.maxpool_attributes:
            logger.warning(
                "auto_pad is deprecated: (%s)", self.maxpool_attributes["auto_pad"]
            )
            assert self.maxpool_attributes["auto_pad"] == "VALID"
        self.maxpool_kernel_shape = tuple(self.maxpool_attributes["kernel_shape"])
        self.maxpool_strides = tuple(self.maxpool_attributes["strides"])
        self.maxpool_pads = fix_padding(
            tuple(self.maxpool_attributes.get("pads", (0, 0, 0, 0)))
        )
        self.output_names = list(node_list[-1][0].output)

    def __repr__(self):
        return (
            f"MaxPool("
            f"kernel_shape={self.maxpool_kernel_shape}, "
            f"stride={self.maxpool_strides}, "
            f"pads={self.maxpool_pads})"
        )

    def as_pytorch(self, maintain_weights=False):
        if not self.dropped:
            assert self.maxpool_pads[0] == self.maxpool_pads[2]
            assert self.maxpool_pads[1] == self.maxpool_pads[3]
            maxpool_op = nn.MaxPool2d
            pads = (self.maxpool_pads[0], self.maxpool_pads[1])
            if len(self.input_shape) == 3:
                maxpool_op = nn.MaxPool1d
                pads = (self.maxpool_pads[0],)
            max_pool_layer = maxpool_op(
                self.maxpool_kernel_shape, stride=self.maxpool_strides, padding=pads
            )
            return PytorchSequential(self, max_pool_layer)
        return PytorchSequential(self)


class BatchNorm(Droppable):
    PATTERNS = [["batchnormalization"]]

    def __init__(self, node_list):
        super().__init__(node_list)
        assert len(node_list) == 1
        batchnorm_op = node_list[0]
        self.attributes, self.weight, self.bias, self.mean, self.var = get_batchnorm_parameters(
            batchnorm_op
        )

        self.in_features = self.out_features = self.weight.shape[0]

    def __repr__(self):
        return f"BatchNorm()"

    @Layer.input_shape.setter
    def input_shape(self, value):
        self.in_features = self.out_features = value[1]
        Layer.input_shape.fset(self, value)

    def as_pytorch(self, maintain_weights=False):
        if not self.dropped:
            bn_layer = torch.nn.BatchNorm2d(
                self.in_features,
                eps=self.attributes["epsilon"],
                momentum=self.attributes["momentum"],
            )
            if maintain_weights and not self.modified:
                bn_layer = PytorchBatchNorm(
                    self.mean,
                    self.var,
                    self.weight,
                    self.bias,
                    self.attributes["momentum"],
                    self.attributes["epsilon"],
                )
            elif maintain_weights:
                raise ValueError("Cannot maintain weights of modified layer.")
            return PytorchSequential(self, bn_layer)


class Relu(Droppable):
    PATTERNS = [["relu"]]

    def __init__(self, node_list):
        super().__init__(node_list)
        assert len(node_list) == 1

    def __repr__(self):
        return f"Relu()"

    def as_pytorch(self, maintain_weights=False):
        if not self.dropped:
            return PytorchSequential(self, nn.ReLU())


class ResidualBlock(ResidualConnection, Droppable, DroppableOperations):
    PATTERNS = [
        ["conv", "batchnormalization", "relu", "conv", "batchnormalization", "add"],
        [
            "conv",
            "batchnormalization",
            "relu",
            "conv",
            "batchnormalization",
            "conv",
            "batchnormalization",
            "add",
        ],
        [
            "pad",
            "conv",
            "batchnormalization",
            "relu",
            "pad",
            "conv",
            "batchnormalization",
            "add",
        ],
        [
            "pad",
            "conv",
            "batchnormalization",
            "relu",
            "pad",
            "conv",
            "batchnormalization",
            "pad",
            "conv",
            "batchnormalization",
            "add",
        ],
    ]

    def __init__(self, node_list):
        super().__init__(node_list)
        for i in range(1, 5):
            assert len(node_list[i - 1][0].output) == 1
            assert (
                node_list[i][0].input[0] == node_list[i - 1][0].output[0]
            ), "%s != %s" % (node_list[1][0].input[0], node_list[0][0].output[0])

        self.use_residual = True

        pad_op = [None, None, None]
        if node_list[0][0].op_type.lower() == "pad":
            if len(node_list) == 8:
                pad_op = [node_list[0], node_list[4]]
                node_list = node_list[1:4] + node_list[5:]
            else:
                pad_op = [node_list[0], node_list[4], node_list[7]]
                node_list = node_list[1:4] + node_list[5:7] + node_list[8:]

        conv_op = [node_list[0], node_list[3]]
        batchnorm_op = [node_list[1], node_list[4]]
        activation_op = node_list[2]
        if len(node_list) == 6:
            downsample = None
            add_op = node_list[5]
            assert add_op[0].input[0] == batchnorm_op[-1][0].output[0], "%s != %s" % (
                add_op[0].input[0],
                batchnorm_op[-1][0].output[0],
            )
            if pad_op[0] is not None:
                assert add_op[0].input[1] == pad_op[0][0].input[0], "%s != %s" % (
                    add_op[0].input[1],
                    pad_op[0][0].input[0],
                )
            else:
                assert add_op[0].input[1] == conv_op[0][0].input[0], "%s != %s" % (
                    add_op[0].input[1],
                    conv_op[0][0].input[0],
                )
        else:
            downsample = node_list[5:7]
            add_op = node_list[7]
            assert downsample[0][0].input[0] == node_list[0][0].input[0], "%s != %s" % (
                downsample[0][0].input[0],
                node_list[0][0].input[0],
            )
            assert (
                downsample[1][0].input[0] == downsample[0][0].output[0]
            ), "%s != %s" % (downsample[1][0].input[0], downsample[0][0].output[0])
            assert downsample[1][0].output[0] in add_op[0].input, "%s not in %s" % (
                downsample[1][0].output[0],
                add_op[0].input,
            )
            assert batchnorm_op[1][0].output[0] in add_op[0].input, "%s not in %s" % (
                batchnorm_op[1][0].output[0],
                add_op[0].input,
            )

        self.in_features = [None, None]
        self.out_features = [None, None]
        self.conv_weight = [None, None]
        self.conv_bias = [None, None]
        self.conv_kernel_shape = [(), ()]
        self.conv_strides = [(), ()]
        self.conv_pads = [(), ()]
        self.conv_padding = ["VALID", "VALID"]
        self.bn_attributes = [{}, {}]
        self.bn_weight = [None, None]
        self.bn_bias = [None, None]
        self.bn_mean = [None, None]
        self.bn_var = [None, None]
        for i in range(2):
            conv_params = get_conv_parameters(conv_op[i], pad_op=pad_op[i])
            self.conv_weight[i], self.conv_bias[i], self.conv_kernel_shape[
                i
            ], self.conv_strides[i], self.conv_pads[i] = conv_params
            self.conv_padding[i] = as_implicit_padding(self.conv_pads[i])
            self.in_features[i] = self.conv_weight[i].shape[1]
            self.out_features[i] = self.conv_bias[i].shape[0]
            self.bn_attributes[i], self.bn_weight[i], self.bn_bias[i], self.bn_mean[
                i
            ], self.bn_var[i] = get_batchnorm_parameters(batchnorm_op[i])
        if activation_op[0].op_type.lower() == "relu":
            self.activation = "relu"
        assert self.activation == "relu"

        self.downsample = False
        if downsample is not None:
            self.downsample = True
            w, b, k, s, p = get_conv_parameters(downsample[0], pad_op=pad_op[-1])
            self.ds_conv_weight = w
            self.ds_conv_bias = b
            self.ds_conv_kernel_shape = k
            self.ds_conv_strides = s
            self.ds_conv_pads = p
            if (
                tuple(self.ds_conv_kernel_shape) == (1, 1)
                and tuple(self.ds_conv_strides) == (1, 1)
                and tuple(self.ds_conv_pads) == (0, 0, 0, 0)
            ):
                self.downsample = "ResizingBottleneck"
            attrs, bn_w, bn_b, bn_m, bn_v = get_batchnorm_parameters(downsample[1])
            self.ds_bn_attributes = attrs
            self.ds_bn_weight = bn_w
            self.ds_bn_bias = bn_b
            self.ds_bn_mean = bn_m
            self.ds_bn_var = bn_v

        self.dropped_operations = set()

    def __repr__(self):
        conv_pads = [pads[:2] for pads in self.conv_pads]
        residual = (
            "None"
            if not self.use_residual
            else "Identity"
            if not self.downsample
            else self.downsample
            if isinstance(self.downsample, str)
            else "Downsample"
        )
        return (
            f"ResidualBlock(\n"
            f"  {self.in_features[0]},"
            f"{self.in_features[1]},"
            f"{self.out_features[1]},\n"
            f"  kernel_shape={self.conv_kernel_shape},\n"
            f"  strides={self.conv_strides},\n"
            f"  pads={conv_pads},\n"
            f"  residual={residual},\n"
            f"  dropped_ops={self.dropped_operations},\n"
            f")"
        )

    @Layer.input_shape.setter
    def input_shape(self, value):
        self.in_features[0] = value[1]
        Layer.input_shape.fset(self, value)
        if not self.downsample and self.out_features[-1] != self.in_features[0]:
            self.downsample = "ResizingBottleneck"
            self.ds_conv_kernel_shape = (1, 1)
            self.ds_conv_strides = (1, 1)
            self.ds_conv_pads = (0, 0, 0, 0)
            self.ds_bn_attributes = {"epsilon": 1e-5, "momentum": 0.1}

    def drop_operation(self, op_type):
        if op_type.lower() not in set(["batchnormalization"]):
            raise ValueError(
                "Cannot drop %s operations from layer type %s."
                % (op_type, self.__class__.__name__)
            )
        self.dropped_operations.add(op_type)
        self.modified = True

    def linearize(self):
        self.modified = True
        self.use_residual = False
        if self.downsample:
            self.downsample = False
        return self

    def as_pytorch(self, maintain_weights=False):
        if maintain_weights and self.modified:
            raise ValueError("Cannot maintain weights of modified layer.")
        if not self.dropped:
            return PytorchResidualBlock(self, maintain_weights=maintain_weights)


class _ResidualBlock(ResidualBlock):
    PATTERNS = [
        [
            "conv",
            "batchnormalization",
            "conv",
            "batchnormalization",
            "relu",
            "conv",
            "batchnormalization",
            "add",
        ],
        [
            "pad",
            "conv",
            "batchnormalization",
            "pad",
            "conv",
            "batchnormalization",
            "relu",
            "pad",
            "conv",
            "batchnormalization",
            "add",
        ],
    ]

    def __init__(self, node_list):
        if len(node_list) == 8:
            super().__init__(node_list[2:7] + node_list[:2] + node_list[7:])
        else:
            super().__init__(node_list[3:10] + node_list[:3] + node_list[10:])


class AveragePool(Droppable):
    PATTERNS = [["averagepool"]]

    def __init__(self, node_list):
        super().__init__(node_list)
        assert len(node_list) == 1
        avgpool_node = node_list[0]
        self.attributes = {a.name: as_numpy(a) for a in avgpool_node[0].attribute}
        self.kernel_shape = tuple(self.attributes["kernel_shape"])
        self.strides = tuple(self.attributes.get("strides", self.kernel_shape))
        self.pads = fix_padding(
            tuple(self.attributes.get("pads", (0, 0) * len(self.kernel_shape)))
        )
        self.count_include_pad = self.attributes.get("count_include_pad", False)

    def __repr__(self):
        return (
            f"AveragePool("
            f"kernel_shape={self.kernel_shape}, "
            f"strides={self.strides}, "
            f"pads={self.pads}, "
            f"count_include_pad={self.count_include_pad})"
        )

    def as_pytorch(self, maintain_weights=False):
        if not self.dropped:
            _, _, h, w = self.input_shape
            return PytorchSequential(
                self,
                nn.AvgPool2d(
                    self.kernel_shape,
                    self.strides,
                    (self.pads[0], self.pads[1]),
                    count_include_pad=self.count_include_pad,
                ),
            )


class GlobalAveragePool(Droppable):
    def __repr__(self):
        return f"GlobalAveragePool()"

    def as_pytorch(self, maintain_weights=False):
        if not self.dropped:
            _, _, h, w = self.input_shape
            return PytorchSequential(self, nn.AvgPool2d((h, w)))


class AdaptiveAveragePool(GlobalAveragePool):
    PATTERNS = [
        [
            "pad",
            "averagepool",
            "shape",
            "gather",
            "unsqueeze",
            "unsqueeze",
            "concat",
            "reshape",
        ]
    ]

    def __init__(self, node_list):
        super().__init__(node_list)
        assert node_list[0][0].op_type.lower() == "pad"
        assert as_numpy(node_list[0][0].attribute[0]) == "constant"
        assert np.all(as_numpy(node_list[0][0].attribute[1]) == 0)
        assert as_numpy(node_list[0][0].attribute[2]) == 0.0

        assert node_list[1][0].op_type.lower() == "averagepool"
        assert np.all(as_numpy(node_list[1][0].attribute[1]) == 0)
        assert np.all(as_numpy(node_list[1][0].attribute[2]) == 1)
        self.kernel_shape = as_numpy(node_list[1][0].attribute[0])
        self.checked = False

        assert node_list[2][0].op_type.lower() == "shape"
        assert node_list[2][0].input == node_list[1][0].output

        assert node_list[3][0].op_type.lower() == "gather"
        assert np.all(as_numpy(node_list[3][0].attribute[0]) == 0)
        assert node_list[3][0].input[0:1] == node_list[2][0].output

        assert node_list[4][0].op_type.lower() == "unsqueeze"
        assert np.all(as_numpy(node_list[4][0].attribute[0]) == 0)
        assert node_list[4][0].input == node_list[3][0].output

        assert node_list[5][0].op_type.lower() == "unsqueeze"
        assert np.all(as_numpy(node_list[5][0].attribute[0]) == 0)
        assert as_numpy(node_list[5][1]) == -1

        assert node_list[6][0].op_type.lower() == "concat"
        assert list(node_list[6][0].input) == list(node_list[4][0].output) + list(
            node_list[5][0].output
        )
        assert np.all(as_numpy(node_list[6][0].attribute[0]) == 0)

        assert node_list[7][0].op_type.lower() == "reshape"
        assert list(node_list[7][0].input) == list(node_list[1][0].output) + list(
            node_list[6][0].output
        )

    @Layer.input_shape.setter
    def input_shape(self, value):
        Layer.input_shape.fset(self, value)
        if not self.checked:
            assert tuple(self.input_shape[-2:]) == tuple(
                self.kernel_shape
            ), "%s != %s" % (tuple(self.input_shape[-2:]), tuple(self.kernel_shape))
            self.checked = True

    def as_pytorch(self, maintain_weights=False):
        if not self.dropped:
            _, _, h, w = self.input_shape
            module = PytorchSequential(self, nn.AvgPool2d((h, w)), PytorchFlatten())
            return module


class _AveragePool(GlobalAveragePool):
    PATTERNS = [["pad", "averagepool"]]

    def __init__(self, node_list):
        super().__init__(node_list)
        assert node_list[0][0].op_type.lower() == "pad"
        assert as_numpy(node_list[0][0].attribute[0]) == "constant"
        assert np.all(as_numpy(node_list[0][0].attribute[1]) == 0)
        assert as_numpy(node_list[0][0].attribute[2]) == 0.0

        assert node_list[1][0].op_type.lower() == "averagepool"
        assert np.all(as_numpy(node_list[1][0].attribute[1]) == 0)
        self.kernel_shape = as_numpy(node_list[1][0].attribute[0])
        assert np.all(as_numpy(node_list[1][0].attribute[2]) == 1) or np.all(
            as_numpy(node_list[1][0].attribute[2]) == self.kernel_shape[0]
        )
        self.checked = False

    @Layer.input_shape.setter
    def input_shape(self, value):
        Layer.input_shape.fset(self, value)
        if not self.checked:
            assert tuple(self.input_shape[-2:]) == tuple(self.kernel_shape)
            self.checked = True


class OnnxGlobalAveragePool(GlobalAveragePool):
    PATTERNS = [["globalaveragepool"]]


class ResnetV2GlobalAveragePool(GlobalAveragePool):
    PATTERNS = [["batchnormalization", "relu", "globalaveragepool"]]

    def __init__(self, node_list):
        super().__init__(node_list)

        batchnorm_op = node_list[0]
        relu_op = node_list[1]
        pool_op = node_list[2]

        self.bn_attributes, self.bn_weight, self.bn_bias, self.bn_mean, self.bn_var = get_batchnorm_parameters(
            batchnorm_op
        )
        if relu_op[0].op_type.lower() == "relu":
            self.activation = "relu"
        assert self.activation == "relu"

    def __repr__(self):
        return f"ResnetGlobalAveragePool(activation={self.activation})"

    def as_pytorch(self, maintain_weights=False):
        if not self.dropped:
            _, c, h, w = self.input_shape
            bn_layer = torch.nn.BatchNorm2d(
                c,
                eps=self.bn_attributes["epsilon"],
                momentum=self.bn_attributes["momentum"],
            )
            if maintain_weights and not self.modified:
                bn_layer.weight.data = torch.from_numpy(self.bn_weight)
                bn_layer.bias.data = torch.from_numpy(self.bn_bias)
                bn_layer.running_mean.data = torch.from_numpy(self.bn_mean)
                bn_layer.running_var.data = torch.from_numpy(self.bn_var)
            elif maintain_weights:
                raise ValueError("Cannot maintain weights of modified layer.")
            return PytorchSequential(self, bn_layer, nn.ReLU(), nn.AvgPool2d((h, w)))


class Reshape(Layer):
    PATTERNS = [["reshape"]]

    def __init__(self, node_list):
        super().__init__(node_list)
        shape = [1, -1]
        if node_list[0][0].op_type.lower() == "reshape":
            shape = tuple(as_numpy(node_list[0][2]))
        self.shape = shape
        self.flatten = False
        if len(shape) == 2 and (shape == (0, -1) or shape[0] == 1 or shape[0] == -1):
            self.flatten = True

    def __repr__(self):
        return f"Reshape({self.output_shape})"

    def scale(self, factor):
        raise ValueError("Cannot scale reshape layers!")

    def drop(self):
        raise ValueError("Cannot drop reshape layers!")

    def as_pytorch(self, maintain_weights=False):
        if not self.dropped:
            if self.flatten:
                return PytorchSequential(self, PytorchFlatten())
            else:
                return PytorchSequential(self, PytorchReshape(self.shape))


class Flatten(Reshape):
    PATTERNS = [["flatten"]]

    def __repr__(self):
        return f"Flatten({self.output_shape})"


class Flatten_(Flatten):
    PATTERNS = [["shape", "gather", "unsqueeze", "unsqueeze", "concat", "reshape"]]

    def __init__(self, node_list):
        super().__init__(node_list)
        assert node_list[0][0].op_type.lower() == "shape"
        assert isinstance(node_list[0][1], Layer)

        assert node_list[1][0].op_type.lower() == "gather"
        assert np.all(as_numpy(node_list[1][0].attribute[0]) == 0)
        assert node_list[1][0].input[0:1] == node_list[0][0].output

        assert node_list[2][0].op_type.lower() == "unsqueeze"
        assert np.all(as_numpy(node_list[2][0].attribute[0]) == 0)
        assert node_list[2][0].input == node_list[1][0].output

        assert node_list[3][0].op_type.lower() == "unsqueeze"
        assert np.all(as_numpy(node_list[3][0].attribute[0]) == 0)
        assert as_numpy(node_list[3][1]) == -1

        assert node_list[4][0].op_type.lower() == "concat"
        assert list(node_list[4][0].input) == list(node_list[2][0].output) + list(
            node_list[3][0].output
        )
        assert np.all(as_numpy(node_list[4][0].attribute[0]) == 0)

        assert node_list[5][0].op_type.lower() == "reshape"
        assert list(node_list[5][0].input) == list(node_list[0][0].input) + list(
            node_list[4][0].output
        )


class Transpose(Layer):
    PATTERNS = [["transpose"]]

    def __init__(self, node_list):
        super().__init__(node_list)
        self.dims = tuple(node_list[0][0].attribute[0].ints)

    def __repr__(self):
        return f"Transpose({self.dims})"

    def scale(self, factor):
        raise ValueError("Cannot scale transpose layers!")

    def drop(self):
        raise ValueError("Cannot drop transpose layers!")

    def as_pytorch(self, maintain_weights=False):
        if not self.dropped:
            return PytorchSequential(self, PytorchTranspose(*self.dims))


class Softmax(Layer):
    PATTERNS = [["softmax"]]

    def __init__(self, node_list):
        super().__init__(node_list)
        self.shape_preserving = True

    def __repr__(self):
        return f"Softmax()"

    def as_pytorch(self, maintain_weights=False):
        logger = logging.getLogger(__name__)
        if not self.dropped:
            logger.warning("Treating Softmax as identity.")
            return PytorchSequential(self)


class LogSoftmax(Layer):
    PATTERNS = [["logsoftmax"]]

    def __init__(self, node_list):
        super().__init__(node_list)
        self.shape_preserving = True

    def __repr__(self):
        return f"LogSoftmax()"

    def as_pytorch(self, maintain_weights=False):
        logger = logging.getLogger(__name__)
        logger.warning("Treating LogSoftmax as identity.")
        return PytorchSequential(self)


class Sigmoid(Layer):
    PATTERNS = [["sigmoid"]]

    def __init__(self, node_list):
        super().__init__(node_list)
        self.shape_preserving = True

    def __repr__(self):
        return f"Sigmoid()"

    def as_pytorch(self, maintain_weights=False):
        if not self.dropped:
            return PytorchSequential(self, torch.nn.Sigmoid())


class Atan(Layer):
    PATTERNS = [["atan"]]

    def __init__(self, node_list):
        super().__init__(node_list)
        self.shape_preserving = True

    def __repr__(self):
        return f"Atan()"

    def as_pytorch(self, maintain_weights=False):
        if not self.dropped:
            return PytorchSequential(self, PytorchAtan())


class Identity(Droppable):
    PATTERNS = [["identity"]]

    def __init__(self, node_list):
        super().__init__(node_list)

    def __repr__(self):
        return f"Identity()"

    def as_pytorch(self, maintain_weights=False):
        if not self.dropped:
            return PytorchSequential(self)


class Multiply(Layer):
    PATTERNS = [["mul"]]

    def __init__(self, node_list):
        super().__init__(node_list)
        self.shape_preserving = True
        self.value = as_numpy(node_list[0][2])
        assert (
            self.value.shape == ()
        ), f"Received tensor of shape {self.value.shape}. Expected ()"

    def __repr__(self):
        return f"Multiply({self.value})"

    def as_pytorch(self, maintain_weights=False):
        if not self.dropped:
            return PytorchSequential(self, PytorchMultiply(self.value))


def drop_layer(layer):
    logger = logging.getLogger(__name__)
    if layer.dropped:
        return
    logger.info("Dropping layer: %s", layer)
    return layer.drop()


def drop_operation(layer, op_type):
    logger = logging.getLogger(__name__)
    if layer.dropped:
        logger.warning(
            "Skipping dropping operation %s of dropped layer: %s", op_type, layer
        )
        return
    logger.info("Dropping operation %s from layer: %s", op_type, layer)
    return layer.drop_operation(op_type)


def linearize(layer):
    logger = logging.getLogger(__name__)
    if layer.dropped:
        logger.warning("Skipping linearization of dropped layer: %s", layer)
        return
    logger.info("Linearizing layer: %s", layer)
    return layer.linearize()


def scale_convolution_stride(layer, factor):
    logger = logging.getLogger(__name__)
    if layer.dropped:
        logger.warning(
            "Skipping scaling convolution stride of dropped layer: %s", layer
        )
        return
    logger.info("Scaling convolutional stride for layer: %s", layer)
    return layer.scale_stride(factor)


def replace_convolution_padding(layer, padding):
    logger = logging.getLogger(__name__)
    if layer.dropped:
        logger.warning(
            "Skipping replacing convolution padding of dropped layer: %s", layer
        )
        return
    logger.info("Replacing convolutional padding for layer: %s", layer)
    return layer.replace_padding(padding)


def scale_layer(layer, factor):
    logger = logging.getLogger(__name__)
    if layer.dropped:
        logger.warning("Skipping scaling of dropped layer: %s", layer)
        return
    logger.info("Scaling layer: %s", layer)
    return layer.scale(factor)


def get_bases(cls):
    c = list(cls.__bases__)
    for base in c:
        c.extend(get_bases(base))
    return set(c) - set([object])


def get_subclasses(cls):
    c = list(cls.__subclasses__())
    for sub in c:
        c.extend(get_subclasses(sub))
    return set(c)


class DNN:
    def __init__(
        self,
        layers,
        input_name="data",
        input_shape=(1, 3, 224, 224),
        input_format="NCHW",
    ):
        self.input_layer = layers[0]
        self.layers = layers[1:]
        if (
            input_format == "NHWC"
            and isinstance(self.layers[0], Transpose)
            and self.layers[0].dims == (0, 3, 1, 2)
        ):
            layers[0]._input_shape = tuple(np.asarray(input_shape)[[0, 3, 1, 2]])
            layers[0]._output_shape = layers[0]._input_shape
            layers[0].input_format = "NCHW"
            Droppable.drop(self.layers[0])
            self.layers = self.layers[1:]
        self.input_shape = input_shape
        if input_format == "NHWC":
            self.input_shape = tuple(np.asarray(input_shape)[[0, 3, 1, 2]])
        self.input_format = input_format
        self.input_name = input_name

        self.layers_by_type = defaultdict(list)
        self.final_layer_index = 0
        for i, layer in enumerate(self.layers):
            if not layer.shape_preserving:
                self.final_layer_index = i
            self.layers_by_type[layer.__class__].append(i)
            for cls in get_bases(layer.__class__):
                self.layers_by_type[cls].append(i)

        self.layer_order = [
            Convolutional,
            FullyConnected,
            MatmulFullyConnected,
            GemmFullyConnected,
        ] + list(self.layers_by_type.keys())

    def __repr__(self):
        return "DNN([\n    %s],\n  input_shape=%s)" % (
            "\n    ".join(repr(layer) for layer in self.layers if not layer.dropped),
            self.input_shape,
        )

    def __eq__(self, other):
        return repr(self) == repr(other)

    def __hash__(self):
        return hash(repr(self))

    @property
    def droppable_layers(self):
        return set(
            [
                layer_id
                for layer_id in self.layers_by_type[Droppable]
                if layer_id < self.final_layer_index
            ]
        )

    @property
    def rescalable_layers(self):
        return set(
            [
                layer_id
                for layer_id in self.layers_by_type[Rescalable]
                if layer_id < self.final_layer_index
            ]
        )

    def forall(
        self,
        layer_type=Layer,
        strategy=partial(scale_layer, factor=0.5),
        excluding=None,
    ):
        if isinstance(layer_type, str):
            if layer_type not in globals():
                raise ValueError("Unknown layer type: %s" % layer_type)
            layer_type = globals()[layer_type]
        excluding_layer_type = None.__class__
        if excluding is not None and "layer_type" in excluding:
            excluding_layer_type = globals()[excluding["layer_type"]]
        excluding_layer_ids = []
        if excluding is not None and "layer_id" in excluding:
            excluding_layer_ids = excluding["layer_id"]
            if not isinstance(excluding_layer_ids, list):
                excluding_layer_ids = [excluding_layer_ids]
        for layer_id in self.layers_by_type[layer_type]:
            layer = self.layers[layer_id]
            if isinstance(layer, excluding_layer_type):
                continue
            if layer_id in excluding_layer_ids:
                continue
            if layer_id >= self.final_layer_index:
                continue
            func = strategy
            if isinstance(func, partial):
                func = func.func
            if hasattr(func, "__self__") and isinstance(func.__self__, DNN):
                strategy(layer_id, layer_type=layer_type)
            else:
                strategy(layer)

    def drop_layer(self, layer_id, layer_type=Layer):
        if isinstance(layer_type, str):
            if layer_type not in globals():
                raise ValueError("Unknown layer type: %s" % layer_type)
            layer_type = globals()[layer_type]
        if not isinstance(layer_id, Iterable):
            layer_id = [layer_id]
        for idx in sorted(
            layer_id,
            key=lambda lid: (self.layer_order.index(type(self.layers[lid])), -lid),
        ):
            layer = self.layers[idx]
            if not isinstance(layer, layer_type):
                raise ValueError("Layer %s is not of type %s" % (layer, layer_type))
            if self.layers.index(layer) >= self.final_layer_index:
                raise ValueError("Cannot remove final layer.")
            drop_layer(layer)
        return self

    def drop_operation(self, layer_id, op_type, layer_type=DroppableOperations):
        if isinstance(layer_type, str):
            if layer_type not in globals():
                raise ValueError("Unknown layer type: %s" % layer_type)
            layer_type = globals()[layer_type]
        if not isinstance(layer_id, Iterable):
            layer_id = [layer_id]
        for idx in sorted(
            layer_id,
            key=lambda lid: (self.layer_order.index(type(self.layers[lid])), -lid),
        ):
            layer = self.layers[idx]
            if not isinstance(layer, DroppableOperations):
                raise ValueError("Layer %s does not have droppable operations" % layer)
            if self.layers.index(layer) >= self.final_layer_index:
                raise ValueError("Cannot remove final layer.")
            drop_operation(layer, op_type)
        return self

    def linearize(self, layer_id, layer_type=Linearizable):
        if isinstance(layer_type, str):
            if layer_type not in globals():
                raise ValueError("Unknown layer type: %s" % layer_type)
            layer_type = globals()[layer_type]
        if not isinstance(layer_id, Iterable):
            layer_id = [layer_id]
        for idx in layer_id:
            layer = self.layers[idx]
            if not isinstance(layer, Linearizable):
                raise ValueError("Layer %s is not linearizable" % layer)
            linearize(layer)
        return self

    def scale_input(self, factor):
        logger = logging.getLogger(__name__)
        self.input_shape = tuple(
            int(dim * factor) if i in (2, 3) else dim
            for i, dim in enumerate(self.input_shape)
        )
        logger.debug("Scaling input: %s %s", self.input_shape, self.input_layer)
        self.input_layer.input_shape = self.input_shape
        return self

    def scale_layer(self, layer_id, factor, layer_type=Layer):
        if isinstance(layer_type, str):
            if layer_type not in globals():
                raise ValueError("Unknown layer type: %s" % layer_type)
            layer_type = globals()[layer_type]
        if not isinstance(layer_id, Iterable):
            layer_id = [layer_id]
        if not isinstance(factor, Iterable):
            factor = [factor for _ in range(len(layer_id))]
        if len(layer_id) != len(factor):
            raise ValueError(
                "Number of scaling factors does not match number of layers."
            )
        for idx, f in zip(layer_id, factor):
            layer = self.layers[idx]
            '''
            if type(factor) not in [int,float]:
                assert len(factor) == len(layer_id)
                print(factor)
                for i in zip(layer_id, factor):
                    print(i)
                print(idx)
                factor = factor[idx]
            '''
            if not isinstance(layer, layer_type):
                raise ValueError("Layer %s is not of type %s" % (layer, layer_type))
            if self.layers.index(layer) >= self.final_layer_index:
                raise ValueError("Cannot scale final layer.")
            scale_layer(layer, f)
        return self

    def scale_convolution_stride(self, layer_id, factor, layer_type=Convolutional):
        if isinstance(layer_type, str):
            if layer_type not in globals():
                raise ValueError("Unknown layer type: %s" % layer_type)
            layer_type = globals()[layer_type]
        if not isinstance(layer_id, Iterable):
            layer_id = [layer_id]
        for idx in layer_id:
            layer = self.layers[idx]
            if not isinstance(layer, layer_type):
                raise ValueError("Layer %s is not of type %s" % (layer, layer_type))
            if self.layers.index(layer) >= self.final_layer_index:
                raise ValueError("Cannot scale final layer.")
            scale_convolution_stride(layer, factor)
        return self

    def replace_convolution_padding(self, layer_id, padding, layer_type=Convolutional):
        if isinstance(layer_type, str):
            if layer_type not in globals():
                raise ValueError("Unknown layer type: %s" % layer_type)
            layer_type = globals()[layer_type]
        if not isinstance(layer_id, Iterable):
            layer_id = [layer_id]
        for idx in layer_id:
            layer = self.layers[idx]
            if not isinstance(layer, layer_type):
                raise ValueError("Layer %s is not of type %s" % (layer, layer_type))
            if self.layers.index(layer) >= self.final_layer_index:
                raise ValueError("Cannot replace final layer padding.")
            replace_convolution_padding(layer, padding)
        return self

    def as_pytorch(self, maintain_weights=False):
        if any(layer.modified for layer in self.layers) and maintain_weights:
            raise ValueError("Cannot maintain weights. Network has been modified.")
        return Net(
            [
                layer.as_pytorch(maintain_weights=maintain_weights)
                for layer in self.layers
                if not layer.dropped
            ],
            self.input_name,
            input_shape=self.input_shape,
            input_format=self.input_format,
        )


class Net(nn.Module):
    def __init__(
        self, layers, input_name, input_shape=(1, 3, 224, 224), input_format="NCHW"
    ):
        super(Net, self).__init__()
        self._layers = layers
        self.layers = nn.Sequential(*layers)
        self.input_shape = input_shape
        self.input_format = input_format
        self.cache = {True: {}, False: {}}

    def forward(self, x, cache_ids=None, validation=False):
        if cache_ids is not None and all(
            [int(cache_id.item()) in self.cache[validation] for cache_id in cache_ids]
        ):
            return torch.stack(
                [
                    torch.from_numpy(self.cache[validation][int(cache_id.item())])
                    for cache_id in cache_ids
                ]
            )
        assert tuple(int(dim) for dim in x.size()[1:]) == tuple(self.input_shape[1:]), (
            "shape of input (%s) is different than expected (%s)"
            % (x.size(), self.input_shape)
        )
        y = self.layers(x)
        if cache_ids is not None:
            y_ = y.cpu().detach().numpy()
            for i, cache_id in enumerate(cache_ids):
                self.cache[validation][int(cache_id.item())] = y_[i]
        return y

    def num_neurons(self, device=torch.device("cpu")):
        neuron_count = 0
        x = torch.ones(self.input_shape).to(device)
        for layer in self.layers:
            if layer.__class__ in [
                PytorchFlatten,
                PytorchTranspose,
                nn.ReLU,
                nn.BatchNorm2d,
                PytorchBatchNorm,
            ]:
                continue
            elif layer.__class__ in [PytorchResidualBlock, PytorchSequential]:
                num_neurons = layer.num_neurons(device=device)
                neuron_count += num_neurons
                x = torch.ones(layer.output_shape).to(device)
            else:
                x = layer(x)
                num_neurons = np.product(x.size())
                neuron_count += num_neurons
        return neuron_count

    @property
    def num_parameters(self):
        parameter_count = 0
        for parameter in self.parameters():
            parameter_count += np.product(parameter.size())
        return parameter_count

    def export_onnx(self, path):
        dummy_input = torch.ones(self.input_shape).to(next(self.parameters()).device)
        torch.onnx.export(self, dummy_input, path)


class LayerConverter:
    def __init__(self):
        self._transition_set = {}
        self.current_transition_set = self._transition_set
        self.build()

    def build(self):
        for cls in get_subclasses(Layer):
            for pattern in cls.PATTERNS:
                last_transition_set = self._transition_set
                for node in pattern:
                    if node not in last_transition_set:
                        last_transition_set[node] = {}
                    last_transition_set = last_transition_set[node]
                    if "_" not in last_transition_set:
                        last_transition_set["_"] = set()
                    last_transition_set["_"].add(cls)
                if last_transition_set.get("*", False):
                    raise ValueError("Shared pattern!: %s" % pattern)
                last_transition_set["*"] = cls

    def reset(self):
        self.current_transition_set = self._transition_set

    def match(self, node):
        if node in self.current_transition_set:
            self.current_transition_set = self.current_transition_set[node]
            layer_type = self.current_transition_set.get("*", False)
            if layer_type:
                return layer_type
            return self.current_transition_set.get("_", None)
        return False

    def create_layer(self, layer_type, node_path, layer_map, node_map, var_map):
        nodes = [
            [node_map[n.output[0]]]
            + [
                layer_map[name]
                if name in layer_map
                else var_map[name]
                if name in var_map
                else node_map[name]
                for name in n.input
            ]
            for n in node_path
        ]
        layer = layer_type(nodes)
        input_layers = [
            layer_map[name] for name in node_path[0].input if name in layer_map
        ]
        assert len(input_layers) == 1
        nodes_in_path = [n.output for n in node_path]
        input_nodes = []
        for node in node_path:
            for name in node.input:
                if name in layer_map and name not in nodes_in_path:
                    input_nodes.append(layer_map[name])
        layer.inputs = layer_map[node_path[0].input[0]]
        return layer

    def convert(self, onnx_model, input_shape=(1, 3, 224, 224), input_format="NCHW"):
        logger = logging.getLogger(__name__)
        self.reset()

        idx = 0
        node_list = []
        last_layer_set = set()
        last_layer_type = None
        last_layer_idx = None
        layers = [Input(input_shape, input_format)]
        input_name = onnx_model.input_name
        layer_map = {input_name: layers[0]}
        while idx < len(onnx_model.nodes):
            node = onnx_model.nodes[idx]
            op_type = node.op_type.lower()
            layer_type = self.match(op_type)
            if layer_type:
                last_layer_set = layer_type
                node_list.append(node)
                if not isinstance(layer_type, set):
                    last_layer_type = layer_type
                    last_layer_idx = idx
                idx += 1
            elif layer_type is None:
                node_list.append(node)
                idx += 1
            elif last_layer_type is None:
                op_group = [n.op_type for n in node_list + [node]]
                raise ValueError("Unknown operation group: %s" % op_group)
            elif isinstance(last_layer_set, set):
                normed_last_idx = len(node_list) - (idx - last_layer_idx) + 1
                layer = self.create_layer(
                    last_layer_type,
                    node_list[:normed_last_idx],
                    layer_map,
                    onnx_model.node_map,
                    onnx_model.var_map,
                )
                layers.append(layer)
                layer_map[layer.output_names[0]] = layer
                idx = last_layer_idx + 1
                node_list = []
                last_layer_type = None
                last_layer_idx = None
                last_layer_set = set()
                self.reset()
            else:
                layer = self.create_layer(
                    last_layer_type,
                    node_list,
                    layer_map,
                    onnx_model.node_map,
                    onnx_model.var_map,
                )
                layers.append(layer)
                layer_map[layer.output_names[0]] = layer
                last_layer_type = None
                node_list = []
                self.reset()
        layer = self.create_layer(
            last_layer_type,
            node_list,
            layer_map,
            onnx_model.node_map,
            onnx_model.var_map,
        )
        layers.append(layer)
        layer_map[layer.output_names[0]] = layer
        return DNN(
            layers,
            input_name=input_name,
            input_shape=input_shape,
            input_format=input_format,
        )


class OnnxModel:
    def __init__(self, model_path, output_node_name=None):
        logger = logging.getLogger(__name__)
        self.model_path = model_path
        self.model = onnx.load(self.model_path)
        self.node_map = {}
        self.var_map = {}
        for initializer in self.model.graph.initializer:
            self.var_map[initializer.name] = initializer
        for node in self.model.graph.node:
            if len(node.output) > 1 and node.op_type not in ["Dropout"]:
                logger.warning("Node has multiple outputs:")
                logger.warning(node)
            if (
                node.op_type in ["Transpose"]
                and len(node.input) == 1
                and node.input[0] in self.var_map
            ):
                assert len(node.output) == 1
                attributes = {a.name: as_numpy(a) for a in node.attribute}
                self.var_map[node.output[0]] = numpy_helper.from_array(
                    as_numpy(self.var_map[node.input[0]]).transpose(attributes["perm"])
                )
                continue
            for output_name in node.output:
                if node.op_type in ["Constant"]:
                    self.var_map[output_name] = node
                else:
                    self.node_map[output_name] = node

        self.nodes = []
        visited = set()

        def topo_sort(node):
            if id(node) in visited:
                return
            visited.add(id(node))
            for name in node.input:
                if name in self.node_map:
                    topo_sort(self.node_map[name])
            self.nodes.append(node)

        self.output_node_name = output_node_name
        if self.output_node_name is None:
            for node in self.node_map.values():
                topo_sort(node)
        elif isinstance(self.output_node_name, str):
            topo_sort(self.node_map[self.output_node_name])
        elif isinstance(self.output_node_name, list):
            for output_name in self.output_node_name:
                topo_sort(output_name)
        else:
            raise ValueError(
                "Unkown output node name type: %s" % type(output_node_name)
            )

        possible_input_nodes = []
        for input_node in self.model.graph.input:
            if input_node.name in self.var_map:
                continue
            possible_input_nodes.append(input_node.name)
        assert len(possible_input_nodes) > 0, "No input nodes found."
        if len(possible_input_nodes) > 1:
            logger.warning("Choosing one input node from: %s", possible_input_nodes)
        self.input_name = possible_input_nodes[0]

        for i, node in enumerate(self.nodes):
            logger.debug("%3d: %s %s %s", i + 1, node.op_type, node.input, node.output)

    def __repr__(self):
        return f"OnnxModel({repr(self.model_path)})"
