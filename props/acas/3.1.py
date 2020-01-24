from dnnv.properties import *
import numpy as np
N = Network('N')
x_min = np.array([[1500, -0.06, 3.1, 980, 960]])
x_max = np.array([[1500, 0.06, 3.141593, 1200.0, 1200.0]])
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
        Or(N(x)[0,0]>N(x)[0,1], N(x)[0,0]>N(x)[0,2], N(x)[0,0]>N(x)[0,3], N(x)[0,0]>N(x)[0,4])
    ),
)
