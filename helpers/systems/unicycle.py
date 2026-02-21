import numpy as np


_DEFAULT_OBS_CENTER = np.array([25.0, 25.0], dtype=np.float64)
_DEFAULT_OBS_RADIUS = 5.0
_DEFAULT_GOAL_CENTER = np.array([40.0, 20.0], dtype=np.float64)
_DEFAULT_OBS_CENTERS = np.array([_DEFAULT_OBS_CENTER], dtype=np.float64)
_DEFAULT_OBS_RADII = np.array([_DEFAULT_OBS_RADIUS], dtype=np.float64)


def wrap_to_pi(angle):
    angle = np.asarray(angle, dtype=np.float64)
    return (angle + np.pi) % (2.0 * np.pi) - np.pi


def unicycle_dynamics(state, control, control_bound=np.pi / 4):
    state = np.asarray(state, dtype=np.float64)
    control = np.asarray(control, dtype=np.float64)

    delta_t = 0.5
    velocity = 2.0

    if state.ndim == 1:
        pose_x, pose_y, theta = state
        bounded_control = np.clip(control, -control_bound, control_bound)
        next_pose_x = pose_x + (delta_t * velocity * np.cos(theta))
        next_pose_y = pose_y + (delta_t * velocity * np.sin(theta))
        next_theta = wrap_to_pi(theta + (delta_t * bounded_control))
        return np.array([next_pose_x, next_pose_y, next_theta], dtype=np.float64)

    pose_x = state[:, 0]
    pose_y = state[:, 1]
    theta = state[:, 2]
    bounded_control = np.clip(control, -control_bound, control_bound)

    next_pose_x = pose_x + (delta_t * velocity * np.cos(theta))
    next_pose_y = pose_y + (delta_t * velocity * np.sin(theta))
    next_theta = wrap_to_pi(theta + (delta_t * bounded_control))

    return np.column_stack([next_pose_x, next_pose_y, next_theta])


def state_controller(
    state,
    *,
    goal_center,
    obstacle_centers,
    obstacle_radii,
    k_goal=1.0,
    k_rep=8.0,
    alpha=0.6,
    k_theta=2.5,
    omega_max=np.pi / 4,
    eps=1e-6,
):
    state = np.asarray(state, dtype=np.float64)
    single_state = state.ndim == 1
    state_batch = state[None, :] if single_state else state

    p = state_batch[:, :2]
    theta = state_batch[:, 2]
    if not (isinstance(goal_center, np.ndarray) and goal_center.dtype == np.float64):
        goal_center = np.asarray(goal_center, dtype=np.float64)
    if not (isinstance(obstacle_centers, np.ndarray) and obstacle_centers.dtype == np.float64):
        obstacle_centers = np.asarray(obstacle_centers, dtype=np.float64)
    if not (isinstance(obstacle_radii, np.ndarray) and obstacle_radii.dtype == np.float64):
        obstacle_radii = np.asarray(obstacle_radii, dtype=np.float64)

    v_att = k_goal * (goal_center - p)
    v_rep = np.zeros_like(v_att)

    for center, radius in zip(obstacle_centers, obstacle_radii):
        diff = p - center
        dist = np.sqrt(np.sum(diff * diff, axis=1) + eps)
        clearance = dist - radius
        weight = np.exp(-alpha * clearance)
        v_rep += k_rep * weight[:, None] * diff / (dist[:, None] ** 3 + eps)

    v = v_att + v_rep
    v_norm = np.linalg.norm(v, axis=1)
    theta_d = np.arctan2(v[:, 1], v[:, 0])
    e_theta = wrap_to_pi(theta_d - theta)
    omega = omega_max * np.tanh(k_theta * e_theta)
    omega = np.where(v_norm < 1e-9, 0.0, omega)

    if single_state:
        return float(omega[0])
    return omega


def cl_unicycle_dynamics(state):
    control_input = state_controller(
        state,
        goal_center=_DEFAULT_GOAL_CENTER,
        obstacle_centers=_DEFAULT_OBS_CENTERS,
        obstacle_radii=_DEFAULT_OBS_RADII,
        k_goal=0.3,
        k_rep=300.0,
        alpha=0.1,
        k_theta=2.0,
        omega_max=np.pi / 4,
    )
    return unicycle_dynamics(state, control_input)


class UnicycleSystem:
    def step(self, state):
        return cl_unicycle_dynamics(state)
