from pathlib import Path

import numpy as np
import torch
import torch.nn as nn


class DQN(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, action_dim),
        )

    def forward(self, x):
        return self.net(x)


_DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
_POLICY_CACHE = None
_POLICY_PATH = Path(__file__).resolve().parent / 'policy.pth'


def mc_ol_dynamics(state, action):
    state = np.asarray(state, dtype=np.float64)
    action = np.asarray(action)

    if state.ndim == 1:
        p, v = state
    else:
        p = state[:, 0]
        v = state[:, 1]

    v_next = v + 0.005 * (action - 1) - 0.0025 * np.cos(3 * p)
    v_next = np.clip(v_next, -0.07, 0.07)
    p_next = np.clip(p + v_next, -1.2, 0.6)

    reset_mask = (p_next <= -1.2) & (v_next < 0)
    v_next = np.where(reset_mask, 0.0, v_next)

    if state.ndim == 1:
        return np.array([p_next, v_next], dtype=np.float64)
    return np.column_stack([p_next, v_next])


def get_policy():
    global _POLICY_CACHE
    if _POLICY_CACHE is None:
        net = DQN(2, 3, 128).to(_DEVICE)
        state_dict = torch.load(str(_POLICY_PATH), map_location=_DEVICE)
        net.load_state_dict(state_dict)
        net.eval()
        _POLICY_CACHE = net
    return _POLICY_CACHE


def policy_action(state):
    state = np.asarray(state, dtype=np.float32)
    net = get_policy()
    with torch.no_grad():
        if state.ndim == 1:
            q = net(torch.as_tensor(state, device=_DEVICE).unsqueeze(0))
            return int(q.argmax(dim=1).item())
        q = net(torch.as_tensor(state, device=_DEVICE))
        return q.argmax(dim=1).cpu().numpy()


def mc_cl_dynamics(state):
    state = np.asarray(state, dtype=np.float64)
    action = policy_action(state)
    return mc_ol_dynamics(state, action)


class MountainCarSystem:
    def step(self, state):
        state = np.asarray(state, dtype=np.float64)
        return mc_cl_dynamics(state)
