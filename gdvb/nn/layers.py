import numpy as np


# abstract layers
class Layer(object):
    def __init__(self, type, in_shape, out_shape):
        self.type = type
        self.in_shape = in_shape
        self.out_shape = out_shape

    def __str__(self):
        return "{}: {} -> {}".format(self.type, self.in_shape, self.out_shape)


class WeightedLayer(Layer):
    def __init__(self, type, size, weights, bias, in_shape, out_shape):
        super().__init__(type, in_shape, out_shape)
        self.size = size
        self.weights = weights
        self.bias = bias


# Concrete layers
class Input(Layer):
    def __init__(self, in_shape):
        super().__init__('Input', in_shape, in_shape)


class Dense(WeightedLayer):
    def __init__(self, size, weights, bias, in_shape):
        super().__init__('FC', size, weights, bias, in_shape, np.array(size))


class Conv(WeightedLayer):
    def __init__(self, size, weights, bias, kernel_size, stride, padding, in_shape):
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        shape = (((in_shape[1]-kernel_size+2*padding) // stride)+1)
        out_shape = np.array([size, shape, shape])
        super().__init__('Conv', size, weights, bias, in_shape, out_shape)


class ReLU(Layer):
    def __init__(self, in_shape):
        super().__init__('ReLU', in_shape, in_shape)


class Flatten(Layer):
    def __init__(self, in_shape):
        super().__init__('Flatten', in_shape, np.array(np.array(in_shape).prod()))


class Pool(Layer):
    def __init__(self, pool_type, kernel_size, stride, padding, in_shape):
        stride = kernel_size
        # assert kernel_size == stride
        assert in_shape.size == 3
        self.pool_type = pool_type
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        shape = (((in_shape[1]-kernel_size+2*padding) // stride)+1)
        out_shape = np.append(in_shape[0], [shape, shape])
        super().__init__('Pool', in_shape, out_shape)


class Transpose(Layer):
    def __init__(self, order, in_shape):
        self.order = order
        assert len(order) == len(in_shape)
        out_shape = np.array([in_shape[x-1] for x in order])
        super().__init__('Transpose', in_shape, out_shape)
