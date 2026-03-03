import numpy as np

def _box_dim(cell):
    return len(cell) // 2

def _box_lows(cell):
    d = _box_dim(cell)
    return np.array([cell[2*i] for i in range(d)], dtype=np.float64)

def _box_highs(cell):
    d = _box_dim(cell)
    return np.array([cell[2*i+1] for i in range(d)], dtype=np.float64)

def _box_clip(cell, x):
    lo = _box_lows(cell)
    hi = _box_highs(cell)
    return np.minimum(np.maximum(x, lo), hi)

def _box_intersection(a, b):
    lo = np.maximum(_box_lows(a), _box_lows(b))
    hi = np.minimum(_box_highs(a), _box_highs(b))
    if np.any(hi < lo):
        return None
    out = []
    for i in range(len(lo)):
        out.extend([float(lo[i]), float(hi[i])])
    return tuple(out)

def _box_volume(cell):
    lo = _box_lows(cell)
    hi = _box_highs(cell)
    side = np.maximum(0.0, hi - lo)
    return float(np.prod(side))

def _points_in_box(P, cell):
    lo = _box_lows(cell)
    hi = _box_highs(cell)
    return np.all((P >= lo) & (P <= hi), axis=1)

def _box_corners(cell):
    lo = _box_lows(cell)
    hi = _box_highs(cell)
    d = len(lo)
    corners = []
    for mask in range(1 << d):
        c = np.empty(d, dtype=np.float64)
        for i in range(d):
            c[i] = hi[i] if (mask & (1 << i)) else lo[i]
        corners.append(c)
    return np.vstack(corners)

def wrap_to_pi(angle):
    angle = np.asarray(angle, dtype=np.float64)
    return (angle + np.pi) % (2.0 * np.pi) - np.pi

def _unwrap_to_reference(theta, ref):
    theta = np.asarray(theta, dtype=np.float64)
    ref = np.asarray(ref, dtype=np.float64)
    return ref + wrap_to_pi(theta - ref)

def _box_center(cell):
    lo = _box_lows(cell)
    hi = _box_highs(cell)
    return 0.5 * (lo + hi)

def _sl3_step_points(system, P):
    P = np.asarray(P, dtype=np.float64)
    try:
        out = system.step(P)
        out = np.asarray(out, dtype=np.float64)
        if out.shape == P.shape:
            return out
    except Exception:
        pass
    return np.vstack([system.step(p) for p in P])



def _sl3_image_box_of_cell(system, cell, angle_dim=None):
    C = _box_corners(cell)      
    Y = _sl3_step_points(system, C)

    if angle_dim is not None:
        ref = _box_center(cell)[angle_dim]
        Y[:, angle_dim] = _unwrap_to_reference(Y[:, angle_dim], ref)

    lo = np.min(Y, axis=0)
    hi = np.max(Y, axis=0)

    out = []
    for i in range(len(lo)):
        out.extend([float(lo[i]), float(hi[i])])
    return tuple(out)


def sl3_refine_self_loops_by_shrink(
    transition_map,
    cells,
    system,
    *,
    max_steps=25,
    eps=1e-12,
    volume_tol=0.0,
    angle_dim=None, 
    verbose=False,
):
    refined = [set(s) for s in transition_map]
    candidates = [i for i, succ in enumerate(refined) if i in succ]

    removed = 0
    kept = 0
    stalled = 0

    for i in candidates:
        cell = cells[i]
        P = cell
        vol_prev = _box_volume(P)

        removed_here = False
        stalled_here = False

        for k in range(max_steps):
            img = _sl3_image_box_of_cell(system, P, angle_dim=angle_dim)
            Pn = _box_intersection(cell, img)
            if Pn is None or _box_volume(Pn) <= volume_tol:
                refined[i].discard(i)
                removed += 1
                removed_here = True
                break

            vol_now = _box_volume(Pn)
            if abs(vol_prev - vol_now) < eps:
                stalled += 1
                stalled_here = True
                break

            P = Pn
            vol_prev = vol_now

        if (not removed_here) and (not stalled_here):
            kept += 1

        if verbose:
            status = "REMOVED" if removed_here else ("STALLED" if stalled_here else "KEPT")
            print(f"[shrink] state {i}: {status}")

    stats = {
        "candidates": len(candidates),
        "removed": removed,
        "kept": kept,
        "stalled": stalled,
    }
    return refined, stats

def _sl3_sample_points_in_cell(cell, n_samples, rng, *, angle_dim=None):
    lo = _box_lows(cell)
    hi = _box_highs(cell)
    P = rng.uniform(lo, hi, size=(n_samples, len(lo)))

    if angle_dim is not None:
        P[:, angle_dim] = wrap_to_pi(P[:, angle_dim])
    return P

def _sl3_all_samples_exit_cell_within_horizon(
    system,
    cell,
    *,
    n_samples=512,
    max_steps=50,
    seed=0,
    angle_dim=None,
):
    rng = np.random.default_rng(seed)
    P = _sl3_sample_points_in_cell(cell, n_samples, rng, angle_dim=angle_dim)

    if angle_dim is not None:
        ref = _box_center(cell)[angle_dim]

    active = _points_in_box(P, cell) 

    for _ in range(max_steps):
        if not np.any(active):
            return True 

        idx = np.where(active)[0]
        Pn = _sl3_step_points(system, P[idx])

        if angle_dim is not None:
            Pn[:, angle_dim] = _unwrap_to_reference(Pn[:, angle_dim], ref)

        P[idx] = Pn
        active[idx] = _points_in_box(Pn, cell)

    return not np.any(active)

def sl3_refine_self_loops_by_sample_exit(
    transition_map,
    cells,
    system,
    *,
    n_samples=512,
    max_steps=50,
    seed=0,
    angle_dim=None,
    verbose=False,
):
    refined = [set(s) for s in transition_map]
    candidates = [i for i, succ in enumerate(refined) if i in succ]

    removed = 0
    kept = 0

    for i in candidates:
        cell = cells[i]
        state_seed = seed + 1000003 * i

        ok_remove = _sl3_all_samples_exit_cell_within_horizon(
            system,
            cell,
            n_samples=n_samples,
            max_steps=max_steps,
            seed=state_seed,
            angle_dim=angle_dim,
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