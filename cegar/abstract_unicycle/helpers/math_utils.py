import numpy as np
from scipy.optimize import linprog
from scipy.spatial import ConvexHull, QhullError


def build_hyperrectangle_vertices(mins, maxs):
    """
    Build vertices for axis-aligned hyperrectangles.

    Args:
        mins: (N, D) lower bounds
        maxs: (N, D) upper bounds

    Returns:
        verts: (N, 2^D, D) vertices
    """
    n_cells, dims = mins.shape
    n_corners = 1 << dims
    corner_mask = (
        (np.arange(n_corners, dtype=np.uint64)[:, None] >> np.arange(dims, dtype=np.uint64)) & 1
    ).astype(bool)
    return np.where(corner_mask[None, :, :], maxs[:, None, :], mins[:, None, :])


def any_vertex_in_box(vertices, box_min, box_max, tol=1e-9):
    """Return True if any vertex lies inside an axis-aligned box."""
    return bool(
        np.any(np.all((vertices >= box_min - tol) & (vertices <= box_max + tol), axis=1))
    )


def minimal_theta_arc_intervals(values, theta_lo, theta_hi, eps=1e-12):
    values = np.asarray(values, dtype=float).reshape(-1)
    period = float(theta_hi - theta_lo)

    if values.size == 0:
        return [(float(theta_lo), float(theta_hi))], 0.0

    u = np.mod(values - theta_lo, period)
    if u.size == 1:
        x = float(theta_lo + u[0])
        return [(x, x)], float(u[0])

    u = np.sort(u)
    gaps = np.diff(np.r_[u, u[0] + period])
    k = int(np.argmax(gaps))

    # Minimal covering arc is complement of largest gap.
    start_u = float(u[(k + 1) % u.size])
    end_u = float(u[k])
    arc_len = (end_u - start_u) % period

    if arc_len >= period - eps:
        return [(float(theta_lo), float(theta_hi))], 0.0

    end_u2 = start_u + arc_len
    if end_u2 <= period + eps:
        return [(float(theta_lo + start_u), float(theta_lo + min(end_u2, period)))], start_u

    # Wrapped arc split at the [-pi, pi]-style cut.
    low_interval = (float(theta_lo), float(theta_lo + (end_u2 - period)))
    high_interval = (float(theta_lo + start_u), float(theta_hi))
    return [low_interval, high_interval], start_u


def unwrap_theta_interval_options(
    interval_lo,
    interval_hi,
    theta_lo,
    theta_hi,
    arc_start_u,
    eps=1e-12,
):
    """
    Map wrapped theta interval [interval_lo, interval_hi] to one/two unwrapped intervals.
    """
    period = float(theta_hi - theta_lo)

    def to_u(x):
        x = float(x)
        if np.isclose(x, theta_hi, atol=eps, rtol=0.0):
            return period
        return float(np.mod(x - theta_lo, period))

    u0 = to_u(interval_lo)
    u1 = to_u(interval_hi)
    if u1 < u0 - eps:
        u0, u1 = u1, u0

    if u1 <= arc_start_u + eps or u0 >= arc_start_u - eps:
        if u0 < arc_start_u - eps:
            return [(float(theta_lo + (u0 + period)), float(theta_lo + (u1 + period)))]
        return [(float(theta_lo + u0), float(theta_lo + u1))]

    return [
        (float(theta_lo + arc_start_u), float(theta_lo + u1)),
        (float(theta_lo + (u0 + period)), float(theta_lo + (arc_start_u + period))),
    ]


def prepare_convex_hull_lp(vertices):
    """
    Prepare reusable data for conv(vertices) vs box intersection checks.

    Returns:
        dict containing:
            vertices: input vertices
            dims: ambient dimension
            corner_mask: (2^D, D) boolean mask for box corners
            H, h: half-space representation H x <= h when available
            barycentric LP data for robust degeneracy fallback
    """
    vertices = np.asarray(vertices, dtype=float)
    n_corners, dims = vertices.shape

    # Barycentric LP data (works even for degenerate hulls).
    A_ub_bary = np.vstack((vertices.T, -vertices.T))
    A_eq_bary = np.ones((1, n_corners), dtype=float)
    b_eq_bary = np.array([1.0], dtype=float)
    c_bary = np.zeros(n_corners, dtype=float)
    bounds_bary = [(0.0, None)] * n_corners

    H = None
    h = None
    if n_corners > dims:
        # Build half-space form once for fast candidate rejection/acceptance.
        # When Qhull fails (typically lower-dimensional hull), we fall back to
        # barycentric LP only to preserve correctness.
        try:
            hull = ConvexHull(vertices)
            H = np.asarray(hull.equations[:, :-1], dtype=float)
            h = np.asarray(-hull.equations[:, -1], dtype=float)
        except QhullError:
            H = None
            h = None

    n_box_corners = 1 << dims
    corner_mask = (
        (np.arange(n_box_corners, dtype=np.uint64)[:, None] >> np.arange(dims, dtype=np.uint64)) & 1
    ).astype(bool)

    return {
        'vertices': vertices,
        'dims': dims,
        'corner_mask': corner_mask,
        'H': H,
        'h': h,
        'c_bary': c_bary,
        'A_ub_bary': A_ub_bary,
        'A_eq_bary': A_eq_bary,
        'b_eq_bary': b_eq_bary,
        'bounds_bary': bounds_bary,
    }


def any_box_corner_in_hull(box_min, box_max, lp_prepared, tol=1e-9):
    """
    Return True iff any corner of the axis-aligned box is inside conv(vertices).

    Uses half-space representation when available; otherwise returns False.
    """
    H = lp_prepared.get('H')
    h = lp_prepared.get('h')
    if H is None or h is None:
        return False

    corner_mask = lp_prepared['corner_mask']
    corners = np.where(corner_mask, box_max[None, :], box_min[None, :])
    vals = corners @ H.T
    return bool(np.any(np.all(vals <= (h[None, :] + tol), axis=1)))


def boxes_disjoint_from_hull(box_mins, box_maxs, lp_prepared, tol=1e-9):
    box_mins = np.asarray(box_mins, dtype=float)
    box_maxs = np.asarray(box_maxs, dtype=float)

    H = lp_prepared.get('H')
    h = lp_prepared.get('h')
    if H is None or h is None:
        return np.zeros(box_mins.shape[0], dtype=bool)

    # For each facet normal n, compute min_{x in box} n^T x analytically.
    # If min > h for any facet, then the whole box lies outside that facet
    # and cannot intersect the polytope.
    choose = np.where(H[None, :, :] >= 0.0, box_mins[:, None, :], box_maxs[:, None, :])
    mins_over_boxes = np.sum(choose * H[None, :, :], axis=2)
    return np.any(mins_over_boxes > (h[None, :] + tol), axis=1)


def convex_hull_intersects_box(box_min, box_max, lp_prepared, tol=1e-9):
    """
    Check if conv(vertices) intersects an axis-aligned box using LP feasibility.

    Args:
        box_min: (D,) lower box bounds
        box_max: (D,) upper box bounds
        lp_prepared: output of prepare_convex_hull_lp(vertices)
        tol: numeric tolerance for relaxed box constraints
    """
    H = lp_prepared.get('H')
    h = lp_prepared.get('h')

    if H is not None and h is not None:
        dims = int(lp_prepared['dims'])
        c = np.zeros(dims, dtype=float)
        A_ub = np.vstack((H, np.eye(dims), -np.eye(dims)))
        b_ub = np.concatenate((h + tol, box_max + tol, -box_min + tol))
        bounds = [(None, None)] * dims

        res = linprog(
            c=c,
            A_ub=A_ub,
            b_ub=b_ub,
            bounds=bounds,
            method='highs',
        )
        return bool(res.success)

    # Degenerate-hull fallback: barycentric feasibility LP.
    c = lp_prepared['c_bary']
    A_ub = lp_prepared['A_ub_bary']
    A_eq = lp_prepared['A_eq_bary']
    b_eq = lp_prepared['b_eq_bary']
    bounds = lp_prepared['bounds_bary']
    b_ub = np.concatenate((box_max + tol, -box_min + tol))

    res = linprog(
        c=c,
        A_ub=A_ub,
        b_ub=b_ub,
        A_eq=A_eq,
        b_eq=b_eq,
        bounds=bounds,
        method='highs',
    )
    return bool(res.success)
