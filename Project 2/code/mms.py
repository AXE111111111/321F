"""Manufactured solution u* = exp(x+y) from the Problem Pack (Direction 2)."""
import numpy as np

def u_star(x, y):
    return np.exp(x + y)

def f_star(x, y):
    return -2.0 * np.exp(x + y)

def g_star(x, y):
    return np.exp(x + y)

def r_star(y, beta: float):
    return (1.0 + beta) * np.exp(1.0 + y)

# Predicted truncation leading coefficients for u* (report §3.3)
TAU_I_OVER_H2 = np.e**2 / 6.0          # → ≈ 1.2315
TAU_B_FIRST_OVER_H = 0.5 * np.e**2     # → ≈ 3.6945
TAU_B_GHOST_OVER_H = np.e**2 / 3.0     # → ≈ 2.4630
