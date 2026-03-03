import numpy as np

def corners_of_box(box):
    x1_min, x1_max, x2_min, x2_max = box
    return np.array([
        [x1_min, x2_min],
        [x1_min, x2_max],
        [x1_max, x2_min],
        [x1_max, x2_max],
    ], dtype=float)

def aabb_of_points(P):
    mins = P.min(axis=0)
    maxs = P.max(axis=0)
    return (mins[0], maxs[0], mins[1], maxs[1])

def box_intersection(a, b, eps=0.0):
    ax1, ax2, ay1, ay2 = a
    bx1, bx2, by1, by2 = b
    x1 = max(ax1, bx1)
    x2 = min(ax2, bx2)
    y1 = max(ay1, by1)
    y2 = min(ay2, by2)

    if (x2 - x1) <= eps or (y2 - y1) <= eps:
        return None
    return (x1, x2, y1, y2)

def box_area(box):
    x1, x2, y1, y2 = box
    return max(0.0, x2 - x1) * max(0.0, y2 - y1)

def step_points(system, P):
    P = np.asarray(P, dtype=np.float64)
    return np.vstack([system.step(p) for p in P])

def iterative_shrink_inside_cell(system, cell, max_steps=25, eps=1e-12, area_tol=0.0, verbose=False):
    C = cell
    S = cell
    last_area = None

    for k in range(max_steps):
        P = corners_of_box(S)
        Pn = step_points(system, P)
        B = aabb_of_points(Pn)
        S_next = box_intersection(C, B, eps=eps)

        if verbose:
            print(f"[self-loop shrink] k={k:02d} area(S)={box_area(S):.3e} -> "
                  f"{'EMPTY' if S_next is None else f'area={box_area(S_next):.3e}'}")

        if S_next is None:
            return {"status": "empty", "steps": k + 1, "final": None}

        a = box_area(S_next)
        if last_area is not None and abs(a - last_area) <= area_tol:
            return {"status": "stalled", "steps": k + 1, "final": S_next}

        last_area = a
        S = S_next

    return {"status": "nonempty", "steps": max_steps, "final": S}

def refine_aabb_self_loops_by_shrink(transition_map, cells, system, *,
                                     max_steps=25, eps=1e-12, area_tol=0.0, verbose=False):
    refined = [set(s) for s in transition_map]
    candidates = [i for i, succ in enumerate(refined) if i in succ]
    print("DEBUG candidates:", len(candidates), "first:", candidates[:10])

    removed = 0
    kept = 0
    stalled = 0

    for i in candidates:
        out = iterative_shrink_inside_cell(
            system,
            cell=cells[i],
            max_steps=max_steps,
            eps=eps,
            area_tol=area_tol,
            verbose=verbose,
        )

        if out["status"] == "empty":
            refined[i].discard(i)
            removed += 1
        elif out["status"] == "stalled":
            stalled += 1
            kept += 1
        else:
            kept += 1

    stats = {
        "candidates": len(candidates),
        "removed": removed,
        "kept": kept,
        "stalled": stalled,
    }
    return refined, stats

def make_transition_map_total(transition_map):
    total = [set(s) for s in transition_map]
    for i, succ in enumerate(total):
        if len(succ) == 0:
            succ.add(i)
    return total

def _sl_sample__point_in_cell_batch(P, cell):
    x1_min, x1_max, x2_min, x2_max = cell
    return (
        (P[:, 0] >= x1_min) & (P[:, 0] <= x1_max) &
        (P[:, 1] >= x2_min) & (P[:, 1] <= x2_max)
    )

def _sl_sample__sample_points_in_cell(cell, n_samples, rng):
    x1_min, x1_max, x2_min, x2_max = cell
    return rng.uniform(
        low=np.array([x1_min, x2_min], dtype=np.float64),
        high=np.array([x1_max, x2_max], dtype=np.float64),
        size=(n_samples, 2),
    )

def _sl_sample__step_points(system, P):
    P = np.asarray(P, dtype=np.float64)

    try:
        out = system.step(P)
        out = np.asarray(out, dtype=np.float64)
        if out.shape == P.shape:
            return out
    except Exception:
        pass

    return np.vstack([system.step(p) for p in P])

def _sl_sample__all_samples_exit_cell_within_horizon(
    system,
    cell,
    *,
    n_samples=256,
    max_steps=25,
    seed=0,
):
    rng = np.random.default_rng(seed)
    P = _sl_sample__sample_points_in_cell(cell, n_samples, rng)

    active = _sl_sample__point_in_cell_batch(P, cell) 

    for _ in range(max_steps):
        if not np.any(active):
            return True

        idx = np.where(active)[0]
        P_active = P[idx]

        P_next = _sl_sample__step_points(system, P_active)
        P[idx] = P_next

        still_inside = _sl_sample__point_in_cell_batch(P_next, cell)
        active[idx] = still_inside

    return not np.any(active)

def sl_refine_self_loops_by_sample_exit(
    transition_map,
    cells,
    system,
    *,
    n_samples=256,
    max_steps=25,
    seed=0,
    verbose=False,
):
    refined = [set(s) for s in transition_map]
    candidates = [i for i, succ in enumerate(refined) if i in succ]

    removed = 0
    kept = 0

    for i in candidates:
        cell = cells[i]

        state_seed = seed + 1000003 * i

        ok_remove = _sl_sample__all_samples_exit_cell_within_horizon(
            system,
            cell,
            n_samples=n_samples,
            max_steps=max_steps,
            seed=state_seed,
        )

        if ok_remove:
            refined[i].discard(i)
            removed += 1
            if verbose:
                print(f"[sample-exit] removed self-loop at state {i}")
        else:
            kept += 1

    stats = {
        "candidates": len(candidates),
        "removed": removed,
        "kept": kept,
        "n_samples": n_samples,
        "max_steps": max_steps,
    }
    return refined, stats