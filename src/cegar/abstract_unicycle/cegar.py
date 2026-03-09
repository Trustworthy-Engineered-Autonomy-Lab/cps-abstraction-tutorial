
from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, List, Set, Tuple, Optional
import numpy as np

from unicycle_partition_3d import Box3D
from unicycle_dyn import UnicycleClosedLoop
from abstraction import UnicycleAbstraction

@dataclass
class CegarResult:
    classification: Dict[int, str]  # uid -> verified/refuted/unknown
    sat_uids: Set[int]
    refined_uids: Set[int]
    iters: int
    last_cex: List[int]

def _pack_successors(succ_list: List[np.ndarray]) -> np.ndarray:
    """Pack ragged successor lists into a dense int64 matrix with padding."""
    n = len(succ_list)
    max_deg = 0
    for s in succ_list:
        if s.size > max_deg:
            max_deg = int(s.size)
    if max_deg == 0:
        return np.zeros((n, 0), dtype=np.int64)
    pad_idx = n
    mat = np.full((n, max_deg), pad_idx, dtype=np.int64)
    for i, s in enumerate(succ_list):
        if s.size:
            mat[i, : s.size] = s
    return mat


def bounded_A_safe_U_goal(
    uids: List[int],
    succ_list: List[np.ndarray],
    is_goal: np.ndarray,
    is_safe: np.ndarray,
    horizon: int,
    *,
    return_layers: bool = False,
) -> np.ndarray | Tuple[np.ndarray, List[np.ndarray]]:
    """\
    Compute sat set for A(safe U goal) within bounded horizon.

    Semantics (unchanged):
      sat_0 = goal
      sat_{t+1} = goal OR (safe AND AX(sat_t))
    where AX(phi) means: all successors satisfy phi.

    This implementation is vectorized over states using a packed successor matrix.
    """
    n = len(uids)
    sat = is_goal.copy()
    layers: List[np.ndarray] = [sat.copy()] if return_layers else []

    succ_mat = _pack_successors(succ_list)
    max_deg = succ_mat.shape[1]

    if max_deg == 0:
        for _ in range(horizon):
            sat = is_goal | is_safe
            if return_layers:
                layers.append(sat.copy())
        return (sat, layers) if return_layers else sat

    for _ in range(horizon):
        # Append sentinel True for padding index.
        sat_pad = np.concatenate([sat, np.array([True], dtype=bool)])
        ax = np.all(sat_pad[succ_mat], axis=1)
        sat = is_goal | (is_safe & ax)
        if return_layers:
            layers.append(sat.copy())
    return (sat, layers) if return_layers else sat

def extract_counterexample(
    uids: List[int],
    succ_list: List[np.ndarray],
    is_goal: np.ndarray,
    is_safe: np.ndarray,
    sat: np.ndarray,
    horizon: int,
    init_set: Set[int],
) -> List[int]:
    uid_to_i = {u:i for i,u in enumerate(uids)}
    # pick initial violating state
    start_uid = None
    for u in init_set:
        i = uid_to_i.get(u, None)
        if i is None:
            continue
        if not sat[i]:
            start_uid = u
            break
    if start_uid is None:
        return []

    path = [start_uid]
    cur = start_uid
    # Compute sat layers once (vectorized) for extraction.
    _, layers = bounded_A_safe_U_goal(
        uids, succ_list, is_goal, is_safe, horizon, return_layers=True
    )

    # layers[t] is sat after t iterations; final is layers[horizon]
    # We want to follow a witness for violation of layers[horizon]
    for t in range(horizon, 0, -1):
        i = uid_to_i[cur]
        succ = succ_list[i]
        if succ.size == 0:
            path.append(cur)
            continue
        # If not safe at cur, we can self-loop
        chosen = None
        # prefer a successor that violates layers[t-1] (keeping violation alive)
        for j in succ:
            if not layers[t-1][j]:
                chosen = uids[j]; break
        if chosen is None:
            chosen = uids[int(succ[0])]
        path.append(chosen)
        cur = chosen
    return path

def propagate_box_one_step(dyn: UnicycleClosedLoop, box: Box3D) -> Box3D:
    corners = box.corners()
    nxt = np.array([dyn.step(c) for c in corners], dtype=float)
    p_lo = float(nxt[:,0].min()); p_hi = float(nxt[:,0].max())
    q_lo = float(nxt[:,1].min()); q_hi = float(nxt[:,1].max())
    th_lo = float(nxt[:,2].min()); th_hi = float(nxt[:,2].max())
    return Box3D(p_lo, p_hi, q_lo, q_hi, th_lo, th_hi)

def validate_counterexample(
    absys: UnicycleAbstraction,
    dyn: UnicycleClosedLoop,
    path: List[int],
    labeler,
) -> Tuple[bool, Optional[int]]:
    """
    Concrete validation via set propagation:
    """
    if not path:
        return (False, None)

    cur_uid = path[0]
    cur_set = absys.part.get_box(cur_uid)

    for t in range(len(path)-1):
        # if current is goal, property satisfied along this prefix
        if "goal" in labeler(absys.part.get_box(cur_uid)):
            return (False, None)

        nxt_uid = path[t+1]
        nxt_box = absys.part.get_box(nxt_uid) if nxt_uid != absys.OUT_UID else None

        img = propagate_box_one_step(dyn, cur_set)
        if nxt_box is None:
            # OUT reached -> treat as unsafe reach => real cex
            return (True, None)

        inter = img.intersection(nxt_box)
        if inter is None:
            # spurious at transition from cur_uid -> nxt_uid
            return (False, cur_uid)

        # advance
        cur_uid = nxt_uid
        cur_set = inter

        # If unsafe at this step (any corner unsafe), it's a real counterexample
        if "unsafe" in labeler(absys.part.get_box(cur_uid)):
            return (True, None)

    return (False, None)

def run_cegar(
    absys: UnicycleAbstraction,
    init_uids: Set[int],
    labeler,
    *,
    horizon: int,
    split_budget: int,
    max_iters: int,
    verbose: bool = False,
) -> CegarResult:
    refined: Set[int] = set()
    last_cex: List[int] = []

    for it in range(max_iters):
        leaves = sorted(list(absys.part.leaves.keys()))
        uid_to_i = {u:i for i,u in enumerate(leaves)}
        # build successor lists in index-space
        succ_list = []
        for u in leaves:
            vs = [v for v in absys.tr.successors(u) if v in uid_to_i]  # ignore OUT in dp
            succ_list.append(np.array([uid_to_i[v] for v in vs], dtype=np.int64))

        # labels
        is_goal = np.array([("goal" in labeler(absys.part.get_box(u))) for u in leaves], dtype=bool)
        is_safe = np.array([("unsafe" not in labeler(absys.part.get_box(u))) for u in leaves], dtype=bool)

        sat = bounded_A_safe_U_goal(leaves, succ_list, is_goal, is_safe, horizon)
        sat_uids = {u for u in leaves if sat[uid_to_i[u]]}

        # pick counterexample if any init violates
        violating = [u for u in init_uids if u in uid_to_i and (not sat[uid_to_i[u]])]
        if not violating:
            last_cex = []
            break

        last_cex = extract_counterexample(leaves, succ_list, is_goal, is_safe, sat, horizon, init_uids)
        if not last_cex:
            break

        real, refine_uid = validate_counterexample(absys, absys.dyn, last_cex, labeler)
        if real:
            # mark start as refuted by making it unsafe? classification later uses sat; keep as unknown/refuted
            # To preserve Clarke-style, we record refuted start cells separately by labeling unsafe? We'll return classification mapping later.
            # Here we just stop (one counterexample found).
            break

        if refine_uid is None:
            # spurious but couldn't identify refinement location -> stop
            break

        if len(refined) >= split_budget:
            break

        if refine_uid in refined:
            # already refined; avoid spinning
            break

        # split
        absys.refine_split(refine_uid)
        refined.add(refine_uid)

        if verbose:
            print(f"[cegar] refined uid={refine_uid} ({len(refined)}/{split_budget})")

    # final classification based on sat and trivial labels
    leaves = sorted(list(absys.part.leaves.keys()))
    uid_to_i = {u:i for i,u in enumerate(leaves)}
    succ_list = []
    for u in leaves:
        vs = [v for v in absys.tr.successors(u) if v in uid_to_i]
        succ_list.append(np.array([uid_to_i[v] for v in vs], dtype=np.int64))
    is_goal = np.array([("goal" in labeler(absys.part.get_box(u))) for u in leaves], dtype=bool)
    is_safe = np.array([("unsafe" not in labeler(absys.part.get_box(u))) for u in leaves], dtype=bool)
    sat = bounded_A_safe_U_goal(leaves, succ_list, is_goal, is_safe, horizon)
    sat_uids = {u for u in leaves if sat[uid_to_i[u]]}

    classification: Dict[int,str] = {}
    for u in leaves:
        labs = labeler(absys.part.get_box(u))
        if "unsafe" in labs:
            classification[u] = "refuted"
        elif sat[uid_to_i[u]]:
            classification[u] = "verified"
        else:
            classification[u] = "unknown"

    return CegarResult(
        classification=classification,
        sat_uids=sat_uids,
        refined_uids=refined,
        iters=it+1,
        last_cex=last_cex
    )
