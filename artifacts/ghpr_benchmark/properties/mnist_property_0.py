from dnnv.properties import *
import numpy as np

N = Network("N")

a = 4
b = 9
c = 8

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
            abs(N(x)[0, 4] - N(x)[0, 9]) < abs(N(x)[0, 4] - N(x)[0, 8]),
            abs(N(x)[0, 4] - N(x)[0, 9]) < abs(N(x)[0, 9] - N(x)[0, 8]),
        ),
    ),
)
