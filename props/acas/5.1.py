from dnnv.properties import *
import numpy as np
N = Network('N')
x_min = np.array([[250, 0.2, -3.141592, 100, 0]])
x_max = np.array([[400, 0.4, -3.1365920000000003, 400, 400]])
x_mean = np.array([[19791.091, 0.0, 0.0, 650.0, 600.0]])
x_range = np.array([[60261.0, 6.28318530718, 6.28318530718, 1100.0, 1200.0]])
x_min = (x_min-x_mean)/x_range * 1
x_max = (x_max-x_mean)/x_range * 1
y_mean = 7.5188840201005975
y_range = 373.94992
Forall(
    x,
    Implies(
        (x_min < x < x_max),
        argmin(N(x)) == 4,
    ),
)
