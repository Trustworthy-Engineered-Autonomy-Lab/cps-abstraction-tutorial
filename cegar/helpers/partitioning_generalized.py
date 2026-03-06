from __future__ import annotations

from typing import Dict, List, Optional, Sequence, Set, Tuple
import numpy as np

from helpers.math_utils import (
    any_box_corner_in_hull,
    any_vertex_in_box,
    boxes_disjoint_from_hull,
    build_hyperrectangle_vertices,
    convex_hull_intersects_box,
    prepare_convex_hull_lp,
)

# These functions generalize Krish's transition builders to Ethan's locally refined
# RectPartition (hierarchical adaptive rectilinear cells).
#
# Partition requirements:
#   - partition.leaves: Dict[int, CellNode] with .rect fields (xmin,xmax,ymin,ymax)
#   - partition.query_intersecting_leaves(Rect) -> List[int]
#   - partition.leaf_uid_for_point(x,y) -> Optional[int]
#   - partition.domain.contains_point / contains_rect (from Rect)
#
# System requirement:
#   - system.step(points: np.ndarray) -> np.ndarray, vectorized (N,2)->(N,2)

def _leaf_ids(partition) -> List[int]:
    return sorted(partition.leaves.keys())

def _rect_to_minmax_2d(rect) -> Tuple[np.ndarray, np.ndarray]:
    bmin = np.array([rect.xmin, rect.ymin], dtype=float)
    bmax = np.array([rect.xmax, rect.ymax], dtype=float)
    return bmin, bmax

def _rect_from_minmax_2d(bmin: np.ndarray, bmax: np.ndarray):
    from abstraction import Rect
    return Rect(float(bmin[0]), float(bmax[0]), float(bmin[1]), float(bmax[1]))

def compute_transitions_AABB_partition(partition, system) -> Dict[int, Set[int]]:
    """Krish AABB candidate builder generalized to arbitrary leaf rectangles.

    For each source leaf cell X:
      - compute next-vertex images via stepping all corners
      - form image AABB
      - candidates are all leaves intersecting that AABB (via partition query)
      - add OUT if AABB exits domain (the caller handles OUT sink)
    """
    leaf_ids = _leaf_ids(partition)
    if not leaf_ids:
        return {}

    # Build mins/maxs arrays for vertices stepping (vectorized)
    mins = np.zeros((len(leaf_ids), 2), dtype=float)
    maxs = np.zeros((len(leaf_ids), 2), dtype=float)
    for i, uid in enumerate(leaf_ids):
        r = partition.leaves[uid].rect
        mins[i, :] = (r.xmin, r.ymin)
        maxs[i, :] = (r.xmax, r.ymax)

    verts = build_hyperrectangle_vertices(mins, maxs)  # (N,4,2)
    flat_verts = verts.reshape(-1, 2)
    flat_next = np.asarray(system.step(flat_verts), dtype=float)
    next_verts = flat_next.reshape(len(leaf_ids), verts.shape[1], 2)
    img_mins = next_verts.min(axis=1)
    img_maxs = next_verts.max(axis=1)

    tr: Dict[int, Set[int]] = {uid: set() for uid in leaf_ids}
    for i, uid in enumerate(leaf_ids):
        box = _rect_from_minmax_2d(img_mins[i], img_maxs[i])
        cands = partition.query_intersecting_leaves(box)
        tr[uid].update(cands)
    return tr

def compute_transitions_poly_partition(partition, system, tol: float = 1e-9) -> Dict[int, Set[int]]:
    """Krish poly builder generalized to arbitrary leaf rectangles.

    Steps:
      1) compute next-verts of each source cell's corners
      2) candidate targets = leaves intersecting image AABB
      3) filter candidates by exact convex-hull vs box intersection checks (Krish helpers)
    """
    leaf_ids = _leaf_ids(partition)
    if not leaf_ids:
        return {}

    mins = np.zeros((len(leaf_ids), 2), dtype=float)
    maxs = np.zeros((len(leaf_ids), 2), dtype=float)
    for i, uid in enumerate(leaf_ids):
        r = partition.leaves[uid].rect
        mins[i, :] = (r.xmin, r.ymin)
        maxs[i, :] = (r.xmax, r.ymax)

    verts = build_hyperrectangle_vertices(mins, maxs)  # (N,4,2)
    flat_verts = verts.reshape(-1, 2)
    flat_next = np.asarray(system.step(flat_verts), dtype=float)
    next_verts = flat_next.reshape(len(leaf_ids), verts.shape[1], 2)
    img_mins = next_verts.min(axis=1)
    img_maxs = next_verts.max(axis=1)

    tr: Dict[int, Set[int]] = {uid: set() for uid in leaf_ids}

    for i, uid in enumerate(leaf_ids):
        # candidates from AABB overlap
        aabb_box = _rect_from_minmax_2d(img_mins[i], img_maxs[i])
        candidates = partition.query_intersecting_leaves(aabb_box)
        if not candidates:
            continue

        poly_verts = next_verts[i]
        lp_prepared = prepare_convex_hull_lp(poly_verts)

        # batch disjoint filtering (Krish helper) using candidate boxes
        cand_mins = np.zeros((len(candidates), 2), dtype=float)
        cand_maxs = np.zeros((len(candidates), 2), dtype=float)
        for k, cu in enumerate(candidates):
            r = partition.leaves[cu].rect
            cand_mins[k, :] = (r.xmin, r.ymin)
            cand_maxs[k, :] = (r.xmax, r.ymax)

        disjoint = boxes_disjoint_from_hull(cand_mins, cand_maxs, lp_prepared, tol=tol)
        survivors = [candidates[k] for k in range(len(candidates)) if not disjoint[k]]
        if not survivors:
            continue

        for cu in survivors:
            r = partition.leaves[cu].rect
            bmin, bmax = _rect_to_minmax_2d(r)

            hit = False
            if any_vertex_in_box(poly_verts, bmin, bmax, tol=tol):
                hit = True
            elif any_box_corner_in_hull(bmin, bmax, lp_prepared, tol=tol):
                hit = True
            elif convex_hull_intersects_box(bmin, bmax, lp_prepared, tol=tol):
                hit = True

            if hit:
                tr[uid].add(cu)

    return tr

def compute_transitions_sample_partition(
    partition,
    system,
    n_samples: int = 256,
    rng_seed: Optional[int] = None,
    batch_cells: Optional[int] = None,
) -> Dict[int, Set[int]]:
    """Krish sampling builder generalized to arbitrary leaf rectangles.

    For each source leaf, sample uniformly within the rectangle, step, then locate
    destination leaf using partition.leaf_uid_for_point.
    """
    leaf_ids = _leaf_ids(partition)
    if not leaf_ids:
        return {}

    rng = np.random.default_rng(rng_seed)
    tr: Dict[int, Set[int]] = {uid: set() for uid in leaf_ids}

    if batch_cells is None:
        batch_cells = min(len(leaf_ids), 256)

    for start in range(0, len(leaf_ids), batch_cells):
        batch = leaf_ids[start:start+batch_cells]
        # build samples per cell
        mins = np.zeros((len(batch), 2), dtype=float)
        widths = np.zeros((len(batch), 2), dtype=float)
        for i, uid in enumerate(batch):
            r = partition.leaves[uid].rect
            mins[i, :] = (r.xmin, r.ymin)
            widths[i, :] = (r.xmax - r.xmin, r.ymax - r.ymin)

        u = rng.random((len(batch), n_samples, 2))
        sampled = mins[:, None, :] + u * widths[:, None, :]
        flat = sampled.reshape(-1, 2)

        flat_next = np.asarray(system.step(flat), dtype=float)

        src_ids = np.repeat(np.array(batch, dtype=int), n_samples)
        for s_uid, (x, y) in zip(src_ids, flat_next):
            dst = partition.leaf_uid_for_point(float(x), float(y))
            if dst is not None:
                tr[int(s_uid)].add(int(dst))

    return tr
