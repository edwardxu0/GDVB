from dnnv.properties import *
import numpy as np

N = Network("N")

a = Parameter("a", type=int)
b = Parameter("b", type=int)
c = Parameter("c", type=int)

Forall(
    x,
    Implies(
        (0 <= x <= 1),
        And(
            abs(N(x)[0, a] - N(x)[0, b]) < abs(N(x)[0, a] - N(x)[0, c]),
            abs(N(x)[0, a] - N(x)[0, b]) < abs(N(x)[0, b] - N(x)[0, c]),
        ),
    ),
)
