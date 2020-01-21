from dnnv.properties import *
import numpy as np
N = Network('N')
x_min = np.array([0.0,-3.141593,-3.141593,100.0,0.0,])
x_max = np.array([60760.0,3.141593,3.141593,1200.0,1200.0,])
x_mean = np.array([1.9791091e+04,0.0,0.0,650.0,600.0,])
x_range = np.array([60261.0,6.28318530718,6.28318530718,1100.0,1200.0,])
x_min[0] = 1500
x_max[0] = 1800
x_min[1] = -0.06
x_max[1] = 0.06
x_min[2] = 3.10
x_min[3] = 980
x_min[4] = 960
x_min = (x_min-x_mean)/x_range
x_max = (x_max-x_mean)/x_range
x_mean = np.mean(np.array([x_min, x_max]), axis=0)
x_min = x_mean-(x_mean - x_min) * 1
x_max = x_mean+(x_max - x_mean) * 1
y_mean = 7.5188840201005975
y_range = 373.94992
Forall(
    x,
    Implies(
        (x_min < x < x_max),
        Or(N(x)[0]>N(x)[1], N(x)[0]>N(x)[2], N(x)[0]>N(x)[3], N(x)[0]>N(x)[4])
    ),
)
