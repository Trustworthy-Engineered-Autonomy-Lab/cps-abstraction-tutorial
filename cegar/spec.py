from __future__ import annotations
from abstraction import Rect
import numpy as np

GOAL_CENTER = np.array([5.0, 5.0], dtype=float)
GOAL_RADIUS = 2.0

def cell_is_goal(rect: Rect) -> bool:
    corners = np.array([
        [rect.xmin, rect.ymin],
        [rect.xmin, rect.ymax],
        [rect.xmax, rect.ymin],
        [rect.xmax, rect.ymax],
    ], dtype=float)
    dists = np.linalg.norm(corners - GOAL_CENTER[None, :], axis=1)
    return bool(np.all(dists <= GOAL_RADIUS))
