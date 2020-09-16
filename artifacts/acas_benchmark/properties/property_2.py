"""
Property $\phi_2$.
  – Description: If the intruder is distant and is significantly slower than the ownship, the score of a COC advisory will never be maximal.
  – Tested on: $N_{x,y}$ for all $x \ge 2$ and for all $y$.
  – Input constraints: $\rho \ge 55947.691$, $v_{own} \ge 1145$, $v_{int} \le 60$.
  – Desired output property: the score for COC is not the maximal score.
"""
from dnnv.properties import *
import numpy as np

N = Network("N")
# x: $\rho$, $\theta$, $\psi$, $v_{own}$, $v_{int}$
# x_{min}: 0.0, -3.141593, -3.141593, 100.0, 0.0
# x_{max}: 60760.0, 3.141593, 3.141593, 1200.0, 1200.0
x_min = np.array([[55947.691, -3.141593, -3.141593, 1145.0, 0.0]])
x_max = np.array([[60760.0, 3.141593, 3.141593, 1200.0, 60.0]])

x_mean = np.array([[1.9791091e04, 0.0, 0.0, 650.0, 600.0]])
x_range = np.array([[60261.0, 6.28318530718, 6.28318530718, 1100.0, 1200.0]])

x_min_normalized = (x_min - x_mean) / x_range
x_max_normalized = (x_max - x_mean) / x_range

# y: Clear-of-Conflict, weak left, weak right, strong left, strong right
y_mean = 7.5188840201005975
y_range = 373.94992

Forall(
    x, Implies(x_min_normalized <= x <= x_max_normalized, argmax(N(x)) != 0),
)
