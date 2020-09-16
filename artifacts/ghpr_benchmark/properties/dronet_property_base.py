from dnnv.properties import *
import numpy as np

N = Network("N")
N_prob_coll = N[:-2, 1]
N_steer_angle = N[:-1, 0]

logit = lambda x: np.log(x / (1 - x))
P_coll_min = logit(Parameter("P_coll_min", type=float, default=1e-9))
P_coll_max = logit(Parameter("P_coll_max", type=float, default=1.0))

steer_max = Parameter("steer_max", type=float, default=90) * np.pi / 180

Forall(
    x,
    Implies(
        And(0 <= x <= 1, P_coll_min <= N_prob_coll(x) <= P_coll_max),
        -steer_max <= N_steer_angle(x) <= steer_max,
    ),
)
