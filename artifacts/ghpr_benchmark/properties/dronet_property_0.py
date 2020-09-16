from dnnv.properties import *
import numpy as np

N = Network("N")
N_prob_coll = N[2:-2, 1]
N_steer_angle = N[2:-1, 0]

logit = lambda x: np.log(x / (1 - x))
P_coll_max = logit(0.1)

steer_max = 5 * np.pi / 180

Forall(
    x,
    Implies(
        And(0 <= x <= 1, N_prob_coll(x) <= P_coll_max),
        -steer_max <= N_steer_angle(x) <= steer_max,
    ),
)
