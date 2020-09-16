from dnnv.properties import *
import numpy as np

N = Network("N")

a = 6
b = 0
c = 7

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
            abs(N(x)[0, 6] - N(x)[0, 0]) < abs(N(x)[0, 6] - N(x)[0, 7]),
            abs(N(x)[0, 6] - N(x)[0, 0]) < abs(N(x)[0, 0] - N(x)[0, 7]),
        ),
    ),
)
