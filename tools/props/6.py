from dnnv.properties import *
import numpy as np

N = Network("N")

x_mean = np.array([1.9791091e+04,0.0,0.0,650.0,600.0])
x_range = np.array([60261.0,6.28318530718,6.28318530718,1100.0,1200.0])

y_mean = 7.5188840201005975
y_range = 373.94992

inv_norm = lambda a, i: a * x_range[i] + x_mean[i]

Forall(
    x,
    Implies(
        And(
            12000 <= inv_norm(x[0],0) <= 62000,
            Or(0.7 <= inv_norm(x[1],1) <= 3.141592, -3.141592 <= inv_norm(x[1],1) <= -0.7),
            -3.141592 <= inv_norm(x[2],2) <= -3.141592 + 0.005,
            100 <= inv_norm(x[3],3) <= 1200,
            0 <= inv_norm(x[4],4) <= 1200
        ),
        argmin(N(x)) == 0
    )
)
