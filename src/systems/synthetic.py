import numpy as np

class SyntheticSystem:
    def __init__(self):
        self.A = np.array([[0.8, -0.3],
                           [0.3,  0.8]])
        self.x_star = np.array([5.0, 5.0])

    def step(self, state):
        state = np.asarray(state, dtype=float)
        return (state - self.x_star) @ self.A.T + self.x_star
