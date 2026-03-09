# Libraries
import numpy as np

# Discrete-time plant dynamics
def dynamics(x, x_star):
    A = np.array([[0.8, -0.3],
                  [0.3,  0.8]])
    x = np.asarray(x, dtype=float)
    return x_star + A @ (x - x_star)

def affine_params(x_star):
    """Return (A, b) such that x_{t+1} = A x_t + b matches dynamics(x, x_star)."""
    x_star = np.asarray(x_star, dtype=float)
    b = (np.eye(2) - A) @ x_star
    return A.copy(), b

# Simple simulation
if __name__ == "__main__":
    x = [0.0, 0.0]
    x_star = [5.0, 5.0]
    for _ in range(50):
        x = dynamics(x, x_star)
        print(x)
