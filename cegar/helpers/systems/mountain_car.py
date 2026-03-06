# helpers/systems/mountain_car.py
import numpy as np

from abstraction import Rect, RectPartition
from krish_abstraction import KrishAbstraction
from helpers.systems.mountain_car_impl import MountainCarSystem, P_MIN, P_MAX, V_MIN, V_MAX, GOAL_P


def build(nx: int = 20, ny: int = 20, *, method: str = "POLY"):
    method = method.upper()
    if method not in ("AABB", "POLY"):
        raise ValueError(f"method must be AABB or POLY, got {method}")

    domain = Rect(float(P_MIN), float(P_MAX), float(V_MIN), float(V_MAX))
    part = RectPartition.uniform_grid(domain, nx, ny)

    # Use Krish's closed-loop system implementation (no custom wrapper)
    system = MountainCarSystem()

    absys = KrishAbstraction(part=part, system=system, method=method)

    phi = "A (safe U goal)"

    def goal_all_fn(points: np.ndarray) -> bool:
        pts = np.asarray(points, dtype=float)
        return bool(np.all(pts[:, 0] >= float(GOAL_P)))

    return {
        "absys": absys,
        "domain": domain,
        "phi": phi,
        "goal_all_fn": goal_all_fn,
        "case_study": "mountain_car",
    }

