import numpy as np

from abstraction import Rect, RectPartition
from krish_abstraction import KrishAbstraction

class SyntheticSystem:
    def __init__(self):
        self.A = np.array([[0.8, -0.3],
                           [0.3,  0.8]])
        self.x_star = np.array([5.0, 5.0])

    def step(self, state):
        state = np.asarray(state, dtype=float)
        return (state - self.x_star) @ self.A.T + self.x_star


def build(nx: int = 40, ny: int = 40, method: str = "POLY"):
    """Factory matching the interface expected by unknown_worklist.py.

    Returns an object with:
      - absys: KrishAbstraction (Ethan CEGAR interface)
      - domain: Rect
      - phi: str (CTL objective string for logging)
      - goal_all_fn(points)->bool
    """
    domain = Rect(-10.0, 10.0, -10.0, 10.0)
    part = RectPartition.uniform_grid(domain, nx, ny)
    system = SyntheticSystem()
    absys = KrishAbstraction(part=part, system=system, method=method)

    center = np.array([5.0, 5.0], dtype=float)
    radius = 2.0
    r2 = radius * radius

    def goal_all_fn(points: np.ndarray) -> bool:
        pts = np.asarray(points, dtype=float)
        d = pts - center[None, :]
        return bool(np.all(np.sum(d * d, axis=1) <= r2))

    return {

        "absys": absys,
        "phi": "A (safe U goal)",
        "goal_all_fn": goal_all_fn,
        "domain": domain,
        "case_study": "synthetic",
    }
    # class Spec:  # simple namespace
    #     pass
    #
    # spec = Spec()
    # spec.absys = absys
    # spec.domain = domain
    # spec.phi = "A (safe U goal)"
    # spec.goal_all_fn = goal_all_fn
    # return spec
