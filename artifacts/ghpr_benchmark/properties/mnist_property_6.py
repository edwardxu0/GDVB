from dnnv.properties import *
import numpy as np

N = Network("N")

a = 8
b = 2
c = 4

# Forall(
#     x,
#     Implies(
#         (0 <= x <= 1),
#         And(
#             abs(N(x)[0, a] - N(x)[0, b]) < abs(N(x)[0, a] - N(x)[0, c]),
#             abs(N(x)[0, a] - N(x)[0, b]) < abs(N(x)[0, b] - N(x)[0, c]),
#         ),
#     ),
# )

Forall(
    x,
    Implies(
        (0 <= x <= 1),
        And(
            abs(N(x)[0, 8] - N(x)[0, 2]) < abs(N(x)[0, 8] - N(x)[0, 4]),
            abs(N(x)[0, 8] - N(x)[0, 2]) < abs(N(x)[0, 2] - N(x)[0, 4]),
        ),
    ),
)
