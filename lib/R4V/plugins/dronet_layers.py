from r4v.nn import *


class DronetOutput(Rescalable):
    PATTERNS = [
        ["matmul", "add", "sigmoid", "matmul", "add", "identity", "identity", "concat"],
        ["gemm", "gemm", "sigmoid", "concat"],
    ]

    def __init__(self, node_list):
        super().__init__(node_list)

        concat_op = node_list[-1]
        if node_list[0][0].op_type.lower() == "matmul":
            self.fc1 = MatmulFullyConnected(node_list[3:5])
            self.fc2 = MatmulFullyConnected(node_list[0:2])
            assert node_list[5][0].input == [node_list[2][0].output[0]]
            assert node_list[6][0].input == [node_list[4][0].output[0]]
            assert concat_op[0].input == [
                node_list[6][0].output[0],
                node_list[5][0].output[0],
            ]
        else:
            self.fc1 = GemmFullyConnected(node_list[0:1])
            self.fc2 = GemmFullyConnected(node_list[1:2])
            assert concat_op[0].input == [
                node_list[0][0].output[0],
                node_list[2][0].output[0],
            ]
        self.layers = [self.fc1, self.fc2]

        concat_attributes = {a.name: as_numpy(a) for a in concat_op[0].attribute}
        self.concat_axis = concat_attributes["axis"]

    def __repr__(self):
        return "Concat(%s, Sigmoid(%s), axis=%d)" % (
            repr(self.fc1),
            repr(self.fc2),
            self.concat_axis,
        )

    @Layer.inputs.setter
    def inputs(self, value):
        self.fc1.inputs = value
        self.fc2.inputs = value
        Layer.inputs.fset(self, value)

    @Layer.input_shape.setter
    def input_shape(self, value):
        assert value[0] == 1
        self.fc1.input_shape = value
        self.fc2.input_shape = value
        self._output_shape = (1, self.fc1.output_shape[1] + self.fc2.output_shape[1])
        Layer.input_shape.fset(self, value)

    def scale(self, factor):
        self.fc1.scale(factor)
        self.fc2.scale(factor)

    def as_pytorch(self, maintain_weights=False):
        if not self.dropped:
            return PytorchConcat(
                self,
                self.layers[0].as_pytorch(maintain_weights=maintain_weights),
                torch.nn.Sequential(
                    self.layers[1].as_pytorch(maintain_weights=maintain_weights),
                    torch.nn.Sigmoid(),
                ),
            )


class DronetResidualBlock(ResidualConnection, Droppable, DroppableOperations):
    PATTERNS = [
        [
            "batchnormalization",
            "relu",
            "conv",
            "batchnormalization",
            "relu",
            "conv",
            "conv",
            "add",
        ]
    ]

    def __init__(self, node_list):
        super().__init__(node_list)

        self.use_residual = True

        conv_op = [node_list[2], node_list[5]]
        batchnorm_op = [node_list[0], node_list[3]]
        activation_op = [node_list[1], node_list[4]]
        downsample = [node_list[6]]
        add_op = node_list[7]

        assert batchnorm_op[0][0].output[0] == activation_op[0][0].input[0]
        assert activation_op[0][0].output[0] == conv_op[0][0].input[0]
        assert conv_op[0][0].output[0] == batchnorm_op[1][0].input[0]
        assert batchnorm_op[1][0].output[0] == activation_op[1][0].input[0]
        assert activation_op[1][0].output[0] == conv_op[1][0].input[0]
        assert conv_op[1][0].output[0] in add_op[0].input
        assert downsample[0][0].output[0] in add_op[0].input

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
            conv_params = get_conv_parameters(conv_op[i])
            self.conv_weight[i], self.conv_bias[i], self.conv_kernel_shape[
                i
            ], self.conv_strides[i], self.conv_pads[i] = conv_params
            self.conv_padding[i] = as_implicit_padding(self.conv_pads[i])
            self.in_features[i] = self.conv_weight[i].shape[1]
            self.out_features[i] = self.conv_bias[i].shape[0]
            self.bn_attributes[i], self.bn_weight[i], self.bn_bias[i], self.bn_mean[
                i
            ], self.bn_var[i] = get_batchnorm_parameters(batchnorm_op[i])
        if activation_op[0][0].op_type.lower() == "relu":
            self.activation_1 = "relu"
        if activation_op[1][0].op_type.lower() == "relu":
            self.activation_2 = "relu"
        assert self.activation_1 == "relu"
        assert self.activation_2 == "relu"

        self.downsample = False
        if downsample is not None:
            self.downsample = Convolutional(downsample)

        self.dropped_operations = set()

    def __repr__(self):
        residual = (
            "None"
            if not self.use_residual
            else "Identity"
            if not self.downsample
            else self.downsample
        )
        return (
            f"ResidualBlock(\n"
            f"  {self.in_features[0]},"
            f"{self.in_features[1]},"
            f"{self.out_features[1]},\n"
            f"  kernel_shape={self.conv_kernel_shape},\n"
            f"  strides={self.conv_strides},\n"
            f"  pads={self.conv_padding},\n"
            f"  residual={residual},\n"
            f"  dropped_ops={self.dropped_operations},\n"
            f")"
        )

    @Layer.inputs.setter
    def inputs(self, value):
        if not isinstance(value, list):
            value = [value]
        if self.downsample and self.use_residual:
            self.downsample._inputs = value
            assert len(self.downsample._inputs) <= 1
            if len(self.downsample._inputs) == 0:
                assert self.downsample.dropped
                return
            self.downsample.input_shape = self.downsample._inputs[0].output_shape
        Layer.inputs.fset(self, value)

    @Layer.input_shape.setter
    def input_shape(self, value):
        self.in_features[0] = value[1]
        Layer.input_shape.fset(self, value)
        if (
            self.use_residual
            and not self.downsample
            and self.out_features[-1] != self.in_features[0]
        ):
            raise ValueError("Residual downsample needed to resize output!")

    def drop(self):
        if self.downsample:
            self.downsample.dropped = True
            self.downsample.modified = True
        super().drop()

    def drop_operation(self, op_type):
        if op_type.lower() not in set(["batchnormalization"]):
            raise ValueError(
                "Cannot drop %s operations from layer type %s."
                % (op_type, self.__class__.__name__)
            )
        self.dropped_operations.add(op_type)

    def linearize(self):
        self.modified = True
        self.use_residual = False
        if self.downsample:
            self.downsample = False
        return self

    def as_pytorch(self, maintain_weights=False):
        if not self.dropped:
            return PytorchResidualBlockV2(self, maintain_weights=maintain_weights)

