from dnnv.properties import *
import numpy as np

N1 = Network("N1")
N2 = Network("N2")

x = Image("properties/input_4.npy")

epsilon = Parameter("epsilon", type=float)

Forall(
    x_,
    Implies(
        ((x - epsilon) < x_ < (x + epsilon)) & (0 <= x_ <= 1),
        argmax(N1(x_)) == argmax(N2(x_)),
    ),
)
