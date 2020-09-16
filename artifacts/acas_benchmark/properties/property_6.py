"""
Property $\phi_6$.
  – Description: If the intruder is sufficiently far away, the network advises COC.
  – Tested on: $N_{1,1}$.
  – Input constraints: $12000 \le \rho \le 62000$, $(0.7 \le \theta \le 3.141592) \lor (−3.141592 \le \theta \le −0.7)$, $−3.141592 \le \psi \le −3.141592 + 0.005$, $100 \le v_{own} \le 1200$, $0 \le v_{int} \le 1200$.
  – Desired output property: the score for COC is the minimal score.
"""
from dnnv.properties import *
import numpy as np

N = Network("N")
# x: $\rho$, $\theta$, $\psi$, $v_{own}$, $v_{int}$
# x_{min}: 0.0, -3.141593, -3.141593, 100.0, 0.0
# x_{max}: 60760.0, 3.141593, 3.141593, 1200.0, 1200.0
x_min_1 = np.array([[12000.0, 0.7, -3.141592, 100.0, 0.0]])
x_max_1 = np.array([[62000.0, 3.141592, -3.141592 + 0.005, 1200.0, 1200.0]])
x_min_2 = np.array([[12000.0, -3.141592, -3.141592, 100.0, 0.0]])
x_max_2 = np.array([[62000.0, -0.7, -3.141592 + 0.005, 1200.0, 1200.0]])

x_mean = np.array([[1.9791091e04, 0.0, 0.0, 650.0, 600.0]])
x_range = np.array([[60261.0, 6.28318530718, 6.28318530718, 1100.0, 1200.0]])

x_min_1_normalized = (x_min_1 - x_mean) / x_range
x_max_1_normalized = (x_max_1 - x_mean) / x_range
x_min_2_normalized = (x_min_2 - x_mean) / x_range
x_max_2_normalized = (x_max_2 - x_mean) / x_range

# y: Clear-of-Conflict, weak left, weak right, strong left, strong right
y_mean = 7.5188840201005975
y_range = 373.94992

Forall(
    x,
    Implies(
        Or(
            x_min_1_normalized <= x <= x_max_1_normalized,
            x_min_2_normalized <= x <= x_max_2_normalized,
        ),
        argmin(N(x)) == 0,
    ),
)
