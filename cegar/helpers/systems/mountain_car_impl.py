# helpers/systems/mountain_car_impl.py
import os
import numpy as np
import torch

# --- closed-loop dynamics copied/ported from Krish's mountain_car.py ---
# We keep it self-contained here so your repo doesn't depend on his script layout.

P_MIN, P_MAX = -1.2, 0.6
V_MIN, V_MAX = -0.07, 0.07
GOAL_P = 0.5

# The policy file must be present at repo root or adjust this path.
_POLICY_PATH_DEFAULT = os.path.join(os.path.dirname(__file__), "..", "..", "policy.pth")


class DQN(torch.nn.Module):
    # minimal architecture loader; must match policy.pth saved architecture
    # If your policy uses a different module, replace this with the exact class used.
    def __init__(self, obs_dim=2, act_dim=3, hidden=128):
        super().__init__()
        self.net = torch.nn.Sequential(
            torch.nn.Linear(obs_dim, hidden),
            torch.nn.ReLU(),
            torch.nn.Linear(hidden, hidden),
            torch.nn.ReLU(),
            torch.nn.Linear(hidden, act_dim),
        )

    def forward(self, x):
        return self.net(x)


def _load_policy(policy_path: str):
    model = DQN()
    sd = torch.load(policy_path, map_location="cpu")
    model.load_state_dict(sd)
    model.eval()
    return model


def _mc_step_single(state_xy: np.ndarray, action: int) -> np.ndarray:
    """
    Standard MountainCar dynamics (Gym-style).
    action: 0 push left, 1 no push, 2 push right
    """
    p, v = float(state_xy[0]), float(state_xy[1])

    force = (action - 1) * 0.001
    gravity = -0.0025 * np.cos(3.0 * p)
    v2 = v + force + gravity
    v2 = float(np.clip(v2, V_MIN, V_MAX))
    p2 = p + v2
    p2 = float(np.clip(p2, P_MIN, P_MAX))

    # in classic MountainCar, if at left bound, velocity resets to 0
    if p2 <= P_MIN and v2 < 0:
        v2 = 0.0

    return np.array([p2, v2], dtype=float)


class MountainCarSystem:
    """
    Closed-loop deterministic system: step(points) -> next_points.
    """
    def __init__(self, policy_path: str = _POLICY_PATH_DEFAULT):
        if not os.path.exists(policy_path):
            raise FileNotFoundError(
                f"policy file not found at {policy_path}. "
                f"Put policy.pth at repo root or update the path in mountain_car_impl.py."
            )
        self.policy = _load_policy(policy_path)

    def step(self, points: np.ndarray) -> np.ndarray:
        points = np.asarray(points, dtype=float)
        if points.ndim == 1:
            points = points[None, :]
        obs = torch.tensor(points, dtype=torch.float32)
        with torch.no_grad():
            q = self.policy(obs)
            act = torch.argmax(q, dim=1).cpu().numpy().astype(int)

        out = np.zeros_like(points, dtype=float)
        for i in range(points.shape[0]):
            out[i, :] = _mc_step_single(points[i, :], int(act[i]))
        return out

