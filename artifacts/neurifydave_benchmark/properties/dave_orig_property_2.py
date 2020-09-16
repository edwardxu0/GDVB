from dnnv.properties import *
import numpy as np

N = Network("N")
means = np.array([103.939, 116.779, 123.68], dtype=np.float32).reshape((1, 3, 1, 1))
mins = np.zeros((1, 3, 100, 100)) - np.tile(means, (1, 1, 100, 100))
maxs = np.zeros((1, 3, 100, 100)) + 255 - np.tile(means, (1, 1, 100, 100))
x = Image("properties/dave_orig_image2.npy") - np.tile(means, (1, 1, 100, 100))
input_layer = 0
output_layer = -2

epsilon = Parameter("epsilon", type=float)
gamma = Parameter("gamma", type=float, default=15.0) * np.pi / 180
output = N[input_layer:](x)
gamma_lb = np.tan(max(-np.pi / 2, (output - gamma) / 2))
gamma_ub = np.tan(min(np.pi / 2, (output + gamma) / 2))
Forall(
    x_,
    Implies(
        (mins <= x_ <= maxs) & ((x - epsilon) < x_ < (x + epsilon)),
        (gamma_lb < N[input_layer:output_layer](x_) < gamma_ub),
    ),
)
