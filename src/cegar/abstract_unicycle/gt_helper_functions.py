import pickle
import numpy as np

def _box_bounds_tuple(box):
    # Your Box3D in unicycle_partition_3d.py
    if all(hasattr(box, a) for a in ("p_lo", "p_hi", "q_lo", "q_hi", "th_lo", "th_hi")):
        return (float(box.p_lo), float(box.p_hi),
                float(box.q_lo), float(box.q_hi),
                float(box.th_lo), float(box.th_hi))

    # Fallbacks if you ever reuse other box types
    if all(hasattr(box, a) for a in ("xmin", "xmax", "ymin", "ymax", "thmin", "thmax")):
        return (float(box.xmin), float(box.xmax),
                float(box.ymin), float(box.ymax),
                float(box.thmin), float(box.thmax))

    if hasattr(box, "bounds") and callable(box.bounds):
        b = box.bounds()
        if len(b) == 6:
            return tuple(map(float, b))

    raise RuntimeError("Could not extract bounds from Box3D. Update _box_bounds_tuple().")


def _build_goal_array_from_gt(gt_reach_regions, grid_resolution):
    """
    gt_reach_regions: dict[(i,j,k)] -> 'goal'/'fail'/'unk'
    grid_resolution is the SAME value you used to generate it (100 -> cells are 99^3)
    """
    n = grid_resolution - 1
    goal = np.zeros((n, n, n), dtype=bool)
    for (i, j, k), lab in gt_reach_regions.items():
        if lab == "goal":
            goal[int(i), int(j), int(k)] = True
    return goal


def _fixed_index_range(edges, lo, hi):
    """
    Return inclusive fixed-cell index range whose intervals intersect [lo, hi].
    edges has length res (grid_resolution), so there are res-1 cells.
    """
    n = len(edges) - 1
    if hi <= edges[0] or lo >= edges[-1]:
        return None
    i_lo = int(np.searchsorted(edges, lo, side="right") - 1)
    i_hi = int(np.searchsorted(edges, hi, side="left") - 1)
    i_lo = max(0, min(n - 1, i_lo))
    i_hi = max(0, min(n - 1, i_hi))
    if i_hi < i_lo:
        return None
    return i_lo, i_hi


def _box_true_goal_under_gt(box, *, domain, goal_arr, grid_resolution):
    """
    True-goal iff the entire abstraction cell is covered by fixed-grid 'goal' cells.
    This matches your earlier "all corners / containment in goal-cells union" spec.
    """
    p_min, p_max, q_min, q_max, th_min, th_max = map(float, domain)
    p_lo, p_hi, q_lo, q_hi, th_lo, th_hi = _box_bounds_tuple(box)

    p_edges  = np.linspace(p_min,  p_max,  grid_resolution)
    q_edges  = np.linspace(q_min,  q_max,  grid_resolution)
    th_edges = np.linspace(th_min, th_max, grid_resolution)

    r1 = _fixed_index_range(p_edges,  p_lo,  p_hi)
    r2 = _fixed_index_range(q_edges,  q_lo,  q_hi)
    r3 = _fixed_index_range(th_edges, th_lo, th_hi)
    if r1 is None or r2 is None or r3 is None:
        return False

    i_lo, i_hi = r1
    j_lo, j_hi = r2
    k_lo, k_hi = r3

    # True goal iff ALL overlapping fixed cells are goal
    return bool(goal_arr[i_lo:i_hi+1, j_lo:j_hi+1, k_lo:k_hi+1].all())


def compute_fnr_against_unicycle_gt(absys, label_map, *, gt_pkl_path, domain, grid_resolution=100):
    gt = pickle.load(open(gt_pkl_path, "rb"))
    goal_arr = _build_goal_array_from_gt(gt, grid_resolution)

    leaves = list(absys.part.leaves.keys())

    true_goal = set()
    for u in leaves:
        box = absys.part.get_box(u)
        if _box_true_goal_under_gt(box, domain=domain, goal_arr=goal_arr, grid_resolution=grid_resolution):
            true_goal.add(u)

    validated = {u for u, lab in label_map.items() if lab == "validated"}
    fn = true_goal - validated
    denom = len(true_goal)
    fnr = (len(fn) / denom) if denom > 0 else float("nan")

    # (Optional) also compute FPR
    true_fail = set(leaves) - true_goal
    fp = validated - true_goal
    fpr = (len(fp) / len(true_fail)) if len(true_fail) > 0 else float("nan")

    print("\n[GT COMPARISON]")
    print(f"true_goal_states     = {len(true_goal)}")
    print(f"validated_states     = {len(validated)}")
    print(f"FN states            = {len(fn)}  FNR = {fnr}")
    print(f"FP states            = {len(fp)}  FPR = {fpr}")

    return fnr, fn, fpr, fp

