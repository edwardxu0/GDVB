import os
import numpy as np
import onnx

from onnx import numpy_helper, shape_inference
from ..nn.layers import Input, Dense, Conv, ReLU, Flatten, Pool, Transpose


class ONNXU:
    def __init__(self, path):
        self.name = os.path.splitext(path.split("/")[-1])[0]
        self.path = path
        self.nb_ops = []
        self.nb_neurons = []
        self.arch = self.parse()

    def __str__(self):
        lines = [self.network_name]

        for l in self.onnx_model.arch:
            lines += ["\t{}\t{} --> {}".format(l.type, l.in_shape, l.out_shape)]
            if hasattr(l, "weights"):
                lines += ["\t\tWeights: {} bias: {}".format(l.weights, l.shape)]
        return "\n".join(lines)

    def parse(self):
        model = onnx.load(self.path)
        model = shape_inference.infer_shapes(model)

        var_dict = {}
        for initializer in model.graph.initializer:
            var_dict[initializer.name] = initializer

        self.input_shape = np.array(
            [int(x.dim_value) for x in model.graph.input[0].type.tensor_type.shape.dim]
        )[1:]

        if len(self.input_shape) == 3:
            if (
                self.input_shape[0] == self.input_shape[1]
                and self.input_shape[1] != self.input_shape[2]
            ):
                self.input_format = "NHWC"
            elif (
                self.input_shape[1] == self.input_shape[2]
                and self.input_shape[0] != self.input_shape[1]
            ):
                self.input_format = "NCHW"
            else:
                if self.input_shape[0] == 1:
                    self.input_format = "NCHW"
                else:
                    assert False
        elif len(self.input_shape) == 1:
            if self.input_shape == [5]:
                self.input_format = "ACAS"
            else:
                assert False
        else:
            print(self.input_shape)
            assert False

        nodes = iter(model.graph.node)
        nodes_list = model.graph.node
        arch = [Input(self.input_shape)]

        transpose_first_fc_layer = False
        for node_id, node in enumerate(nodes):
            if node.op_type == "Conv":
                W_name = node.input[1]
                W = self._as_numpy(var_dict[W_name])
                # print('weights', W.shape)

                if len(node.input) == 3:
                    B_name = node.input[2]
                    B = self._as_numpy(var_dict[B_name])
                else:
                    # bias are zero if not defined
                    B = np.zeros(W.shape[0])
                for a in node.attribute:
                    # print(a.name)
                    if a.name == "auto_pad":
                        if a.s == "VALID" or a.s == b"VALID":
                            padding = 0
                        elif a.s == "SAME":
                            padding = "SAME"
                        else:
                            assert False
                    elif a.name == "kernel_shape":
                        kernel_size = self._as_numpy(a)[0]
                    elif a.name == "strides":
                        # print('strides', a)
                        stride = self._as_numpy(a)[0]
                    elif a.name == "pads":
                        # print('pads',a)
                        padding = self._as_numpy(a)[0]
                    else:
                        pass

                # if previous layer is a Pad layer, adjust padding of this convolutional layer
                # e.g. cifar_convbig onnx from ERAN benchmark
                pre_node = None if node_id == 0 else nodes_list[node_id - 1]
                if pre_node and pre_node.op_type == "Pad":
                    for a in pre_node.attribute:
                        if a.name == "pads":
                            pads = self._as_numpy(a)
                            assert len(pads) == 8
                            assert set(pads[[0, 1, 4, 5]]) == set([0])
                            assert set(pads[[2, 3, 6, 7]]) == set([pads[2]])
                            padding = pads[2]

                layer = Conv(
                    B.shape[0], W, B, kernel_size, stride, padding, arch[-1].out_shape
                )

                nb_op_add = (kernel_size**3) * np.prod(layer.out_shape)
                nb_op_mul = ((kernel_size - 1) * (kernel_size**2) + 1) * np.prod(
                    layer.out_shape
                )
                self.nb_ops += [(nb_op_add, nb_op_mul)]
                self.nb_neurons += [np.prod(layer.out_shape)]

            elif node.op_type == "Gemm" or node.op_type == "MatMul":
                if node.op_type == "Gemm":
                    W_name = node.input[1]
                    B_name = node.input[2]
                    W = self._as_numpy(var_dict[W_name])
                    B = self._as_numpy(var_dict[B_name])
                elif node.op_type == "MatMul":
                    B_node = next(nodes)
                    assert B_node.op_type == "Add"
                    W_name = node.input[1]
                    B_name = B_node.input[1]
                    W = self._as_numpy(var_dict[W_name])
                    B = self._as_numpy(var_dict[B_name])
                else:
                    assert False

                if transpose_first_fc_layer:
                    for i in reversed(range(len(arch))):
                        if arch[i].type == "Conv":
                            break
                    assert arch[i].type in ["Conv", "Input"]

                    if self.input_format == "NHWC":
                        W = W.T
                    (
                        h,
                        w,
                        c,
                    ) = arch[i].out_shape
                    m = np.zeros((h * w * c, h * w * c))
                    column = 0
                    for i in range(h * w):
                        for j in range(c):
                            m[i + j * h * w, column] = 1
                            column += 1
                    W = np.matmul(W, m)
                transpose_first_fc_layer = False

                layer = Dense(B.shape[0], W, B, arch[-1].out_shape)

                nb_op_add = np.prod(layer.out_shape)
                nb_op_mul = np.prod(layer.out_shape)
                self.nb_ops += [(nb_op_add, nb_op_mul)]
                self.nb_neurons += [np.prod(layer.out_shape)]

            elif node.op_type == "Relu":
                layer = ReLU(arch[-1].out_shape)

            elif node.op_type == "Flatten":
                layer = Flatten(arch[-1].out_shape)

            elif node.op_type == "Constant":
                next_node = next(nodes)
                assert next_node.op_type == "Reshape"
                layer = Flatten(arch[-1].out_shape)

            elif node.op_type == "Reshape":
                shape = self._as_numpy(var_dict[node.input[1]])
                assert len(shape) == 2 and shape[0] == 1
                layer = Flatten(arch[-1].out_shape)

            # only transpose
            elif node.op_type == "Transpose":
                order = None
                for a in node.attribute:
                    if a.name == "perm":
                        order = a.ints
                assert order is not None and len(order) == 4
                order = order[1:]
                layer = Transpose(order, arch[-1].out_shape)

                # print("Transpose layer detected.")
                transpose_first_fc_layer = True

            elif node.op_type == "MaxPool":
                for a in node.attribute:
                    if a.name == "kernel_shape":
                        kernel_size = self._as_numpy(a)[0]
                    elif a.name == "strides":
                        stride = self._as_numpy(a)[0]
                    elif a.name == "pads":
                        padding = self._as_numpy(a)[0]
                    else:
                        assert False
                layer = Pool("max", kernel_size, stride, padding, arch[-1].out_shape)

            # elif node.op_type == 'Concat':
            #     nb_inputs = len(node.input)
            #     concat_layers = arch[-nb_inputs:]
            #     arch = arch[:-nb_inputs]
            #
            #     W = []
            #     B = []
            #     for cl in concat_layers:
            #         assert cl.type == 'FC'
            #         W += [x for x in cl.weights]
            #         B += [x for x in cl.bias]
            #
            #     W = np.array(W)
            #     B = np.array(B)
            #     #print(W.shape, B.shape, concat_layers[0].in_shape)
            #
            # # elif node.op_type == 'Pad':
            #     next_node = next(nodes)
            #     if next_node.op_type == 'AveragePool':
            #         print('Hacking the code to handle the global average pool for ResNet.')
            #         assert next_node.op_type == 'AveragePool'
            #         in_shape = arch[-1].out_shape
            #         kernel_size = in_shape[1]
            #         stride = kernel_size
            #         padding = 0
            #         layer = Pool("average", kernel_size, stride, padding, in_shape)
            #
            # # elif node.op_type == 'BatchNormalization':
            #     W = self._as_numpy(var_dict[node.input[1]])
            #     B = self._as_numpy(var_dict[node.input[2]])
            #     mean = self._as_numpy(var_dict[node.input[3]])
            #     var = self._as_numpy(var_dict[node.input[4]])
            #     # 'https://pytorch.org/docs/stable/nn.html' defines the normalization operation
            #     eps = 1e-5
            #     W_ = W/np.sqrt(var+eps)
            #     B_ = W*mean/np.sqrt(var+eps) + B
            #     #RESHAPE LAYER
            #     layer = FCLayer(B_.shape[0], W_, B_)
            #     #RESHAPE LAYER
            #     assert False, "not implemented; not supported."

            elif node.op_type in [
                "Identity",
                "Dropout",
                "Softmax",
                "Add",
                "GlobalAveragePool",
                "BatchNormalization",
                "Atan",
                "Mul",
                "Pad",
                "Sigmoid",
            ]:
                continue
            else:
                assert False, "Unsupported operator type: " + str(node.op_type)
            arch += [layer]
        return arch

    @staticmethod
    def _as_numpy(node):
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
            raise ValueError("Unknown attribute type: %s" % (node,))
        else:
            raise ValueError("Unknown node type: %s" % type(node))
