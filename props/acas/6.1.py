from dnnv.properties import *
import numpy as np
N = Network('N')
x_min = np.array([[12000, 0.7, -3.141592, 100, 0]])
x_max = np.array([[62000, 3.141592, -3.1365920000000003, 1200, 1200]])
x_mean = np.array([[19791.091, 0.0, 0.0, 650.0, 600.0]])
x_range = np.array([[60261.0, 6.28318530718, 6.28318530718, 1100.0, 1200.0]])
x_min = (x_min-x_mean)/x_range * 1
x_max = (x_max-x_mean)/x_range * 1
y_mean = 7.5188840201005975
y_range = 373.94992
x_min2 = np.array([[12000, -3.141592, -3.141592, 100, 0]])
x_max2 = np.array([[62000, -0.7, -3.1365920000000003, 1200, 1200]])
x_min2 = (x_min2-x_mean)/x_range * 1
x_max2 = (x_max2-x_mean)/x_range * 1
Forall(
    x,
    Implies(
        Or(x_min < x < x_max, x_min2 < x < x_max2),
        argmin(N(x)) == 0
    )
)
