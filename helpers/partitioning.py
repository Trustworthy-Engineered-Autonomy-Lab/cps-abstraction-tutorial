from collections import defaultdict
from math import sqrt, log

import numpy as np
from helpers.math_utils import (
    any_box_corner_in_hull,
    any_vertex_in_box,
    boxes_disjoint_from_hull,
    build_hyperrectangle_vertices,
    convex_hull_intersects_box,
    minimal_theta_arc_intervals,
    prepare_convex_hull_lp,
    unwrap_theta_interval_options,
)


def _compute_grid_strides(grid_dims):
    """Return row-major strides mapping multi-index -> flat index."""
    dims = len(grid_dims)
    strides = np.ones(dims, dtype=np.int64)
    for d in range(dims - 2, -1, -1):
        strides[d] = strides[d + 1] * int(grid_dims[d + 1])
    return strides


def generate_grid(domain_ranges, cells_per_dim):
    """
    Creates a uniform rectilinear grid in N-dimensions.

    Args:
        domain_ranges: List of (min, max) tuples for each dimension. Length D.
        cells_per_dim: List of integers specifying number of cells for each dimension. Length D.

    Returns:
        grid context dict containing `cells`, `mins`, `maxs`, `widths`, and `grid_meta`
        (`grid_lines`, `grid_dims`, `strides`) for downstream
        transition pipelines.
    """
    domain_ranges = np.asarray(domain_ranges)
    cells_per_dim = np.asarray(cells_per_dim)
    
    dims = len(domain_ranges)

    # 1. Generate linspaces for each dimension (boundaries)
    # lines_per_dim is a list of arrays, each of length (n_i + 1)
    lines_per_dim = [
        np.linspace(r[0], r[1], n + 1) 
        for r, n in zip(domain_ranges, cells_per_dim)
    ]

    # 2. Create intervals (starts and ends) for each dimension
    starts_per_dim = [lines[:-1] for lines in lines_per_dim]
    ends_per_dim = [lines[1:] for lines in lines_per_dim]

    # 3. Create meshgrid for broadcasting
    # We need to construct grids for every dimension's starts and ends.
    # meshgrid returns a list of D arrays, each of shape (n1, n2, ..., nD)
    
    # Generate meshgrids for lower bounds
    starts_mesh = np.meshgrid(*starts_per_dim, indexing='ij')
    
    # Generate meshgrids for upper bounds
    ends_mesh = np.meshgrid(*ends_per_dim, indexing='ij')

    # 4. Stack and reshape
    # We want rows like: [start_1, end_1, start_2, end_2, ..., start_D, end_D]
    # starts_mesh[d] is the grid of starts for dimension d
    
    components = []
    for d in range(dims):
        components.append(starts_mesh[d])
        components.append(ends_mesh[d])
        
    # Stack along the last axis to get (n1, n2, ..., nD, 2*D)
    grid_ND = np.stack(components, axis=-1)
    
    # Flatten to list of cells (N, 2*D)
    cells = grid_ND.reshape(-1, 2 * dims)

    grid_dims = tuple(int(n) for n in cells_per_dim.tolist())
    grid_meta = {
        'grid_lines': [np.asarray(lines, dtype=float) for lines in lines_per_dim],
        'grid_dims': grid_dims,
        'strides': _compute_grid_strides(grid_dims),
    }
    mins = cells[:, 0::2]
    maxs = cells[:, 1::2]
    return {
        'cells': cells,
        'n_cells': cells.shape[0],
        'dims': dims,
        'mins': mins,
        'maxs': maxs,
        'widths': maxs - mins,
        'grid_meta': grid_meta,
    }

def _prepare_transition_geometry(grid_ctx, system):
    """
    Shared precompute for one-step transition methods.

    Returns:
        dict containing stepped vertices and image AABBs.
    """
    n_cells = grid_ctx['n_cells']
    dims = grid_ctx['dims']
    if n_cells == 0:
        return grid_ctx

    mins = grid_ctx['mins']
    maxs = grid_ctx['maxs']

    all_verts = build_hyperrectangle_vertices(mins, maxs)
    n_corners = all_verts.shape[1]
    flat_verts = all_verts.reshape(-1, dims)

    flat_next_verts = np.asarray(system.step(flat_verts), dtype=float)
    next_verts = flat_next_verts.reshape(n_cells, n_corners, dims)
    img_mins = next_verts.min(axis=1)
    img_maxs = next_verts.max(axis=1)

    return {
        **grid_ctx,
        'next_verts': next_verts,
        'img_mins': img_mins,
        'img_maxs': img_maxs,
    }


def _build_candidate_context(step_ctx, periodic_theta):
    """
    Build candidate index ranges per source.

    Stores one base index range [start,end] per dim and, when `periodic_theta=True`,
    an optional second theta range for wrapped arcs.
    """
    next_verts = step_ctx['next_verts']
    img_mins = step_ctx['img_mins']
    img_maxs = step_ctx['img_maxs']
    grid_meta = step_ctx['grid_meta']
    n_cells, _, dims = next_verts.shape

    valid = np.ones(n_cells, dtype=bool)
    starts = np.empty((n_cells, dims), dtype=np.int64)
    ends = np.empty((n_cells, dims), dtype=np.int64)

    theta_arc_starts = np.zeros(n_cells, dtype=float) if periodic_theta else None
    theta_starts_2 = np.full(n_cells, -1, dtype=np.int64) if periodic_theta else None
    theta_ends_2 = np.full(n_cells, -1, dtype=np.int64) if periodic_theta else None

    for d, lines in enumerate(grid_meta['grid_lines']):
        n_dim = grid_meta['grid_dims'][d]
        s = np.searchsorted(lines, img_mins[:, d], side='right') - 1
        e = np.searchsorted(lines, img_maxs[:, d], side='right') - 1
        s = np.clip(s, 0, n_dim - 1)
        e = np.clip(e, 0, n_dim - 1)
        in_domain = (img_maxs[:, d] >= lines[0]) & (img_mins[:, d] <= lines[-1])
        valid &= in_domain & (s <= e)
        starts[:, d] = s
        ends[:, d] = e

    if periodic_theta:
        d = 2
        lines = grid_meta['grid_lines'][d]
        n_dim = grid_meta['grid_dims'][d]
        theta_lo, theta_hi = float(lines[0]), float(lines[-1])
        for i in range(n_cells):
            intervals, arc_start = minimal_theta_arc_intervals(next_verts[i, :, d], theta_lo, theta_hi)
            theta_arc_starts[i] = arc_start

            if len(intervals) == 1:
                int_lo, int_hi = intervals[0]
                s = int(np.searchsorted(lines, int_lo, side='right') - 1)
                e = int(np.searchsorted(lines, int_hi, side='right') - 1)
                starts[i, d] = int(np.clip(s, 0, n_dim - 1))
                ends[i, d] = int(np.clip(e, 0, n_dim - 1))
                if starts[i, d] > ends[i, d]:
                    valid[i] = False
                continue

            if len(intervals) == 2:
                (int_lo_1, int_hi_1), (int_lo_2, int_hi_2) = intervals
                s1 = int(np.searchsorted(lines, int_lo_1, side='right') - 1)
                e1 = int(np.searchsorted(lines, int_hi_1, side='right') - 1)
                s2 = int(np.searchsorted(lines, int_lo_2, side='right') - 1)
                e2 = int(np.searchsorted(lines, int_hi_2, side='right') - 1)
                starts[i, d] = int(np.clip(s1, 0, n_dim - 1))
                ends[i, d] = int(np.clip(e1, 0, n_dim - 1))
                theta_starts_2[i] = int(np.clip(s2, 0, n_dim - 1))
                theta_ends_2[i] = int(np.clip(e2, 0, n_dim - 1))
                if (starts[i, d] > ends[i, d]) and (theta_starts_2[i] > theta_ends_2[i]):
                    valid[i] = False
                continue

            valid[i] = False

    return {
        'valid': valid,
        'theta_arc_starts': theta_arc_starts,
        'starts': starts,
        'ends': ends,
        'theta_starts_2': theta_starts_2,
        'theta_ends_2': theta_ends_2,
        'strides': grid_meta['strides'],
        'candidate_cache': {},
    }


def _expand_candidate_range(start_idx, end_idx, strides, cache):
    key = tuple(np.column_stack((start_idx, end_idx)).reshape(-1).tolist())
    candidates = cache.get(key)
    if candidates is not None:
        return candidates

    flat_candidates = np.array([0], dtype=np.int64)
    for d in range(start_idx.shape[0]):
        offsets = np.arange(start_idx[d], end_idx[d] + 1, dtype=np.int64) * strides[d]
        flat_candidates = (flat_candidates[:, None] + offsets[None, :]).reshape(-1)

    cache[key] = flat_candidates
    return flat_candidates


def _candidate_cells_for_source(src_idx, candidate_ctx):
    """
    Enumerate candidate destination cells for one source.
    """
    start_idx = candidate_ctx['starts'][src_idx].copy()
    end_idx = candidate_ctx['ends'][src_idx].copy()
    strides = candidate_ctx['strides']
    cache = candidate_ctx['candidate_cache']

    base = _expand_candidate_range(start_idx, end_idx, strides, cache)
    theta_starts_2 = candidate_ctx['theta_starts_2']
    if theta_starts_2 is None:
        return base

    s2 = int(theta_starts_2[src_idx])
    if s2 < 0:
        return base

    e2 = int(candidate_ctx['theta_ends_2'][src_idx])
    start_idx[2] = s2
    end_idx[2] = e2
    second = _expand_candidate_range(start_idx, end_idx, strides, cache)
    return np.unique(np.concatenate((base, second)))


def compute_transitions_AABB(grid_ctx, system, periodic_theta=False):
    """
    Computes transitions for N-d uniform grid cells using AABB overlaps.
    
    Args:
        grid_ctx: grid context dict returned by generate_grid(...)
        system: System object with a step() method
        periodic_theta: wrap theta dimension (index 2) for candidate generation
        
    Returns:
        transition_map: list of sets, where transition_map[i] contains indices of cells that cell i transitions to.
    """
    ctx = _prepare_transition_geometry(grid_ctx, system)
    n_cells = ctx['n_cells']
    if n_cells == 0:
        return []

    candidate_ctx = _build_candidate_context(ctx, periodic_theta)

    transition_map = [set() for _ in range(n_cells)]

    for i in np.flatnonzero(candidate_ctx['valid']):
        candidates = _candidate_cells_for_source(i, candidate_ctx)
        if candidates.size:
            transition_map[i].update(candidates.tolist())

    return transition_map


def compute_transitions_poly(grid_ctx, system, tol=1e-9, periodic_theta=False):
    """
    Computes transitions using exact convex-polytope vs hyperrectangle intersection.

    This routine first over-approximates candidate targets using AABB overlap, then
    performs an exact feasibility check for:
        conv(image_vertices_i) intersect target_cell_j != empty.

    Args:
        grid_ctx: grid context dict returned by generate_grid(...)
        system: System object with a vectorized step() method
        tol: numerical tolerance for intersection checks
        periodic_theta: wrap theta dimension (index 2) during candidate and box tests

    Returns:
        transition_map: list of sets, where transition_map[i] contains indices of
                        cells that cell i transitions to.
    """
    ctx = _prepare_transition_geometry(grid_ctx, system)
    n_cells = ctx['n_cells']
    if n_cells == 0:
        return []

    grid_meta = ctx['grid_meta']
    mins = ctx['mins']
    maxs = ctx['maxs']
    next_verts = ctx['next_verts']
    candidate_ctx = _build_candidate_context(ctx, periodic_theta)
    theta_arc_starts = candidate_ctx['theta_arc_starts']

    transition_map = [set() for _ in range(n_cells)]

    for i in np.flatnonzero(candidate_ctx['valid']):
        candidates = _candidate_cells_for_source(i, candidate_ctx)
        if candidates.size == 0:
            continue

        poly_verts = next_verts[i]
        if periodic_theta:
            # Unwrap theta coordinates into the source arc frame so box tests are contiguous.
            poly_verts = poly_verts.copy()
            d = 2
            lines = grid_meta['grid_lines'][d]
            theta_lo, theta_hi = float(lines[0]), float(lines[-1])
            period = theta_hi - theta_lo
            start_u = float(theta_arc_starts[i])
            u = np.mod(poly_verts[:, d] - theta_lo, period)
            u = np.where(u < start_u, u + period, u)
            poly_verts[:, d] = theta_lo + u

        lp_prepared = prepare_convex_hull_lp(poly_verts)

        if periodic_theta:
            # Wrapped theta is handled per-candidate using unwrapped options below.
            survivors = candidates
        else:
            box_mins = mins[candidates]
            box_maxs = maxs[candidates]
            disjoint_mask = boxes_disjoint_from_hull(box_mins, box_maxs, lp_prepared, tol=tol)
            if np.all(disjoint_mask):
                continue
            survivors = candidates[~disjoint_mask]

        for j in survivors:
            # Start from wrapped target cell bounds; split theta into unwrapped
            # options only when periodic_theta is enabled.
            box_options = [(mins[j], maxs[j])]
            if periodic_theta:
                d = 2
                lines = grid_meta['grid_lines'][d]
                theta_lo, theta_hi = float(lines[0]), float(lines[-1])
                start_u = float(theta_arc_starts[i])
                new_options = []
                for bmin, bmax in box_options:
                    int_options = unwrap_theta_interval_options(
                        bmin[d],
                        bmax[d],
                        theta_lo,
                        theta_hi,
                        start_u,
                    )
                    for int_lo, int_hi in int_options:
                        bmin_new = bmin.copy()
                        bmax_new = bmax.copy()
                        bmin_new[d] = int_lo
                        bmax_new[d] = int_hi
                        new_options.append((bmin_new, bmax_new))
                box_options = new_options

            hit = False
            if periodic_theta:
                option_mins = np.asarray([bmin for bmin, _ in box_options], dtype=float)
                option_maxs = np.asarray([bmax for _, bmax in box_options], dtype=float)
                option_disjoint = boxes_disjoint_from_hull(option_mins, option_maxs, lp_prepared, tol=tol)
                if np.all(option_disjoint):
                    continue
                option_indices = np.flatnonzero(~option_disjoint)
            else:
                option_indices = (0,)

            for k in option_indices:
                box_min, box_max = box_options[int(k)]
                if any_vertex_in_box(poly_verts, box_min, box_max, tol=tol):
                    hit = True
                    break
                if any_box_corner_in_hull(box_min, box_max, lp_prepared, tol=tol):
                    hit = True
                    break
                if convex_hull_intersects_box(box_min, box_max, lp_prepared, tol=tol):
                    hit = True
                    break
            if hit:
                transition_map[i].add(int(j))

    return transition_map


def compute_transitions_sample(
    grid_ctx,
    system,
    rng_seed=None,
    batch_n_samples=1_000,
    periodic_theta=False,
    delta=0.01,
    beta=0.01,
):
    """
    Computes transitions by global Monte Carlo sampling in vectorized batches.
    
    Args:
        grid_ctx: grid context dict returned by generate_grid(...)
        system: System object with a vectorized step() method
        rng_seed: optional random seed for reproducibility
        batch_n_samples: number of global samples per batch
        periodic_theta: wrap theta dimension (index 2) into its grid domain

    Returns:
        transition_map: list of sets, where transition_map[i] contains indices of
                        cells that sampled trajectories from cell i reach in one step.
    """
    MN_CONST = 2 * sqrt(2) + sqrt(3)

    n_cells = grid_ctx['n_cells']
    dims = grid_ctx['dims']
    if n_cells == 0:
        return []

    grid_meta = grid_ctx['grid_meta']
    rng = np.random.default_rng(rng_seed)
    transition_map = [set() for _ in range(n_cells)]
    transition_counts = defaultdict(int)
    singleton_tc = 0

    strides = grid_meta['strides']
    grid_dims = grid_meta['grid_dims']
    grid_lines = grid_meta['grid_lines']
    domain_mins = np.asarray([lines[0] for lines in grid_lines], dtype=float)
    domain_widths = np.asarray([lines[-1] - lines[0] for lines in grid_lines], dtype=float)
    theta_lines = grid_lines[2] if periodic_theta else None
    theta_lo = float(theta_lines[0]) if periodic_theta else None
    theta_hi = float(theta_lines[-1]) if periodic_theta else None
    theta_period = (theta_hi - theta_lo) if periodic_theta else None

    batch_n_samples = max(1, int(batch_n_samples))
    processed = 0
    while True:
        b = batch_n_samples
        sampled_states = domain_mins + rng.random((b, dims)) * domain_widths

        src_coords = np.empty((b, dims), dtype=np.int64)
        for d, lines in enumerate(grid_lines):
            if periodic_theta and d == 2:
                wrapped = theta_lo + np.mod(sampled_states[:, d] - theta_lo, theta_period)
                idx = np.searchsorted(lines, wrapped, side='right') - 1
                src_coords[:, d] = np.clip(idx, 0, grid_dims[d] - 1)
                continue
            idx = np.searchsorted(lines, sampled_states[:, d], side='right') - 1
            src_coords[:, d] = np.clip(idx, 0, grid_dims[d] - 1)
        src_ids = src_coords @ strides

        flat_next = np.asarray(system.step(sampled_states), dtype=float)
        if flat_next.ndim == 1:
            flat_next = flat_next[None, :]

        valid = np.ones(flat_next.shape[0], dtype=bool)
        dst_coords = np.empty((flat_next.shape[0], dims), dtype=np.int64)

        for d, lines in enumerate(grid_lines):
            if periodic_theta and d == 2:
                wrapped = theta_lo + np.mod(flat_next[:, d] - theta_lo, theta_period)
                idx = np.searchsorted(lines, wrapped, side='right') - 1
                dst_coords[:, d] = np.clip(idx, 0, grid_dims[d] - 1)
                continue

            idx = np.searchsorted(lines, flat_next[:, d], side='right') - 1
            # Treat boundary points (including exactly lines[-1]) as in-domain.
            # Using idx-range checks alone would drop values at the upper bound
            # because searchsorted(..., side='right') maps them to grid_dims[d].
            in_domain = (flat_next[:, d] >= lines[0]) & (flat_next[:, d] <= lines[-1])
            valid &= in_domain
            dst_coords[:, d] = np.clip(idx, 0, grid_dims[d] - 1)

        if not np.any(valid):
            processed += b

            missing_mass_ub = (singleton_tc / processed) + (MN_CONST * sqrt(log(3 / delta) / processed))
            # print(f'Samples: {processed} | f1: {singleton_tc} | Missing Mass UB: {missing_mass_ub}')
            if missing_mass_ub < beta:
                break

            continue

        dst_flat = dst_coords[valid] @ strides
        src_valid = src_ids[valid]

        packed = src_valid * n_cells + dst_flat
        uniq, counts = np.unique(packed, return_counts=True)
        src_uniq = uniq // n_cells
        dst_uniq = uniq % n_cells

        for s, d, c in zip(src_uniq, dst_uniq, counts):
            src_i = int(s)
            dst_i = int(d)
            increment = int(c)
            transition_map[src_i].add(dst_i)

            key = (src_i, dst_i)
            prev = transition_counts.get(key, 0)
            new = prev + increment
            transition_counts[key] = new
            if prev == 0 and new == 1:
                singleton_tc += 1
            elif prev == 1 and new > 1:
                singleton_tc -= 1

        processed += b

        # Compute missing mass upper bound
        missing_mass_ub = (singleton_tc / processed) + (MN_CONST * sqrt(log(3 / delta) / processed))
        # print(f'Samples: {processed} | f1: {singleton_tc} | Missing Mass UB: {missing_mass_ub}')
        if missing_mass_ub < beta:
            break

    return transition_map
