
from __future__ import annotations
import numpy as np
from dataclasses import dataclass
from typing import List, Tuple, Optional

from unicycle_partition_3d import Box3D, wrap_to_pi, theta_min_arc_intervals

@dataclass
class UnicycleParams:
    dt: float = 0.5
    v: float = 1.0
    omega_max: float = np.pi/4
    p_bounds: Tuple[float,float] = (0.0, 50.0)
    q_bounds: Tuple[float,float] = (0.0, 40.0)
    goal_center: Tuple[float,float] = (40.0, 20.0)
    goal_radius: float = 8.0
    obs_center: Tuple[float,float] = (25.0, 25.0)
    obs_radius: float = 5.0
    k_goal: float = 0.3
    k_rep: float = 300.0
    alpha: float = 0.1
    k_theta: float = 2.0
    eps: float = 1e-6

class UnicycleClosedLoop:
    """
    Closed-loop unicycle dynamics with deterministic controller.
    Matches the provided collaborator implementation.
    """
    def __init__(self, params: UnicycleParams = UnicycleParams()):
        self.p = params
        self.goal_center = np.array(self.p.goal_center, dtype=float)
        self.obs_center = np.array(self.p.obs_center, dtype=float)

    def _state_controller(self, s: np.ndarray) -> float:
        px, py, theta = float(s[0]), float(s[1]), float(s[2])
        pos = np.array([px, py], dtype=float)

        v_att = self.p.k_goal * (self.goal_center - pos)

        diff = pos - self.obs_center
        dist = np.sqrt(diff[0]**2 + diff[1]**2 + self.p.eps)
        clearance = dist - self.p.obs_radius
        w = np.exp(-self.p.alpha * clearance)
        v_rep = self.p.k_rep * w * diff / (dist**3 + self.p.eps)

        v = v_att + v_rep
        if np.linalg.norm(v) < 1e-9:
            return 0.0

        theta_d = float(np.arctan2(v[1], v[0]))
        e_theta = wrap_to_pi(theta_d - theta)
        omega = self.p.omega_max * float(np.tanh(self.p.k_theta * e_theta))
        return float(np.clip(omega, -self.p.omega_max, self.p.omega_max))

    def step(self, s: np.ndarray) -> np.ndarray:
        px, py, theta = float(s[0]), float(s[1]), float(s[2])
        omega = self._state_controller(np.array([px, py, theta], dtype=float))
        nx = px + self.p.dt * self.p.v * float(np.cos(theta))
        ny = py + self.p.dt * self.p.v * float(np.sin(theta))
        nth = wrap_to_pi(theta + self.p.dt * omega)
        return np.array([nx, ny, nth], dtype=float)

    def box_corners(self, box: Box3D) -> np.ndarray:
        return box.corners()

    def any_corner_oob(self, corners: np.ndarray) -> bool:
        pmin, pmax = self.p.p_bounds
        qmin, qmax = self.p.q_bounds
        ps = corners[:, 0]
        qs = corners[:, 1]
        return bool(np.any(ps < pmin) or np.any(ps > pmax) or np.any(qs < qmin) or np.any(qs > qmax))

    def any_corner_in_obstacle(self, corners: np.ndarray) -> bool:
        xy = corners[:, :2]
        d = np.linalg.norm(xy - self.obs_center[None, :], axis=1)
        return bool(np.any(d <= self.p.obs_radius))

    def all_corners_in_goal(self, corners: np.ndarray) -> bool:
        xy = corners[:, :2]
        d = np.linalg.norm(xy - self.goal_center[None, :], axis=1)
        return bool(np.all(d <= self.p.goal_radius))

    def image_from_box(self, box: Box3D) -> tuple[np.ndarray, list[Box3D], bool, float]:
        verts = box.corners()
        next_verts = np.array([self.step(v) for v in verts], dtype=float)
        hits_oob = self.any_corner_oob(next_verts)

        p_lo = float(next_verts[:,0].min()); p_hi = float(next_verts[:,0].max())
        q_lo = float(next_verts[:,1].min()); q_hi = float(next_verts[:,1].max())

        # minimal arc(s) in [-pi,pi]
        intervals = theta_min_arc_intervals(next_verts[:,2])
        img_boxes = [Box3D(p_lo, p_hi, q_lo, q_hi, tlo, thi) for (tlo,thi) in intervals]

        first_lo = intervals[-1][0]  # if wrapping, second interval is (lo,pi)
        start_u = float((first_lo + np.pi) % (2*np.pi))
        return next_verts, img_boxes, hits_oob, start_u
