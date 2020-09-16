"""
Property $\phi_8$.
  – Description: For a large vertical separation and a previous ``weak left'' advisory, the network will either output COC or continue advising ``weak left''.
  – Tested on: $N_{2,9}$.
  – Input constraints: $0 \le \rho \le 60760$, $−3.141592 \le \theta \le −0.75 * 3.141592$, $−0.1 \le \psi \le 0.1$, $600 \le v_{own} \le 1200$, $600 \le v_{int} \le 1200$.
  – Desired output property: the score for ``weak left'' is minimal or the score for COC is minimal.
"""
from dnnv.properties import *
import numpy as np

N = Network("N")
# x: $\rho$, $\theta$, $\psi$, $v_{own}$, $v_{int}$
# x_{min}: 0.0, -3.141593, -3.141593, 100.0, 0.0
# x_{max}: 60760.0, 3.141593, 3.141593, 1200.0, 1200.0
x_min = np.array([[0.0, -3.141592, -0.1, 600.0, 600.0]])
x_max = np.array([[60760.0, -0.75 * 3.141592, 0.1, 1200.0, 1200.0]])

x_mean = np.array([[1.9791091e04, 0.0, 0.0, 650.0, 600.0]])
x_range = np.array([[60261.0, 6.28318530718, 6.28318530718, 1100.0, 1200.0]])

x_min_normalized = (x_min - x_mean) / x_range
x_max_normalized = (x_max - x_mean) / x_range

# y: Clear-of-Conflict, weak left, weak right, strong left, strong right
y_mean = 7.5188840201005975
y_range = 373.94992

Forall(
    x,
    Implies(
        x_min_normalized <= x <= x_max_normalized,
        Or(argmin(N(x)) == 1, argmin(N(x)) == 0),
    ),
)
