from dnnv.properties import *
import numpy as np
N = Network('N')
x_min = np.array([0.0,-3.141593,-3.141593,100.0,0.0,])
x_max = np.array([60760.0,3.141593,3.141593,1200.0,1200.0,])
x_mean = np.array([1.9791091e+04,0.0,0.0,650.0,600.0,])
x_range = np.array([60261.0,6.28318530718,6.28318530718,1100.0,1200.0,])
x_min[0] = 12000
x_max[0] = 62000
x_min[1] = np.array([0.7,-3.141592])
x_max[1] = np.array([3.141592,-0.7])
x_min[2] = -3.141592
x_max[2] = -3.141592+0.005
x_min[3] = 100
x_max[3] = 1200
x_min[4] = 0
x_max[4] = 1200
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
        And(
            x_min[0] <= x[0] <= x_max[0],
            Or(x_min[1][0] <= x[1] <= x_max[1][0], x_min[1][1] <= x[1] <= x_max[1][1]),
            x_min[2] <= x[2] <= x_max[2],
            x_max[3] <= x[3] <= x_max[3],
            x_max[4] <= x[4] <= x_max[4]
        ),
        argmin(N(x)) == 0
    )
)
