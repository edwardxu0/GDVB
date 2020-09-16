from dnnv.properties import *
import numpy as np

N = Network("N")
x = Image("properties/dave_small_image9.npy") / 255.0
input_layer = 0
output_layer = -2

epsilon = Parameter("epsilon", type=float) / 255.0
gamma = Parameter("gamma", type=float, default=15.0) * np.pi / 180
output = N[input_layer:](x)
gamma_lb = np.tan(max(-np.pi / 2, (output - gamma) / 2))
gamma_ub = np.tan(min(np.pi / 2, (output + gamma) / 2))
Forall(
    x_,
    Implies(
        (0 <= x_ <= 1) & ((x - epsilon) < x_ < (x + epsilon)),
        (gamma_lb < N[input_layer:output_layer](x_) < gamma_ub),
    ),
)
