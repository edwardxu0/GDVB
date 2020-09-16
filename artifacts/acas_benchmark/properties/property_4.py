"""
Property $\phi_4$.
  – Description: If the intruder is directly ahead and is moving away from the ownship but at a lower speed than that of the ownship, the score for COC will not be minimal.
  – Tested on: all networks except $N_{1,7}, $N_{1,8}$, and $N_{1,9}$.
  – Input constraints: $1500 \le \rho \le 1800$, $−0.06 \le θ \le 0.06$, $\psi = 0$, $v_{own} \ge 1000$, $700 \le v_{int} \le 800$.
  – Desired output property: the score for COC is not the minimal score.
"""
from dnnv.properties import *
import numpy as np

N = Network("N")
# x: $\rho$, $\theta$, $\psi$, $v_{own}$, $v_{int}$
# x_{min}: 0.0, -3.141593, -3.141593, 100.0, 0.0
# x_{max}: 60760.0, 3.141593, 3.141593, 1200.0, 1200.0
x_min = np.array([[1500.0, -0.06, 0.0, 1000.0, 700.0]])
x_max = np.array([[1800.0, 0.06, 0.0, 1200.0, 800.0]])

x_mean = np.array([[1.9791091e04, 0.0, 0.0, 650.0, 600.0]])
x_range = np.array([[60261.0, 6.28318530718, 6.28318530718, 1100.0, 1200.0]])

x_min_normalized = (x_min - x_mean) / x_range
x_max_normalized = (x_max - x_mean) / x_range

# y: Clear-of-Conflict, weak left, weak right, strong left, strong right
y_mean = 7.5188840201005975
y_range = 373.94992

Forall(
    x, Implies(x_min_normalized <= x <= x_max_normalized, argmin(N(x)) != 0),
)
