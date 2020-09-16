from dnnv.properties import *
import numpy as np

N = Network("N")

a = 7
b = 1
c = 6

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
            abs(N(x)[0, 7] - N(x)[0, 1]) < abs(N(x)[0, 7] - N(x)[0, 6]),
            abs(N(x)[0, 7] - N(x)[0, 1]) < abs(N(x)[0, 1] - N(x)[0, 6]),
        ),
    ),
)
