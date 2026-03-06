from __future__ import annotations

import re
from dataclasses import dataclass
from typing import List, Optional, Set, Tuple

import numpy as np

from abstraction import Abstraction, Rect


# Data structures

@dataclass
class ValidationResult:
    """Clarke-style validation result.

    feasible=True  -> REAL concrete counterexample (terminate: NOT VERIFIED)
    feasible=False -> Spurious counterexample (apply refinement)
    """

    feasible: bool
    fracture_k: Optional[int]          
    uid_path: List[int]               
    reach_sets: List[Rect]           
    x_rects: List[Optional[Rect]]      


@dataclass
class CEGARResult:
    verified: bool
    iterations: int
    ignored_counterexamples: int
    last_cex: Optional[Tuple[List[int], List[int]]]  # (prefix, cycle)


def ctl_get_counterexample_lasso(
    absys: Abstraction,
    init_uids: Set[int],
    merge_actions: bool = True,
    **_kwargs,
) -> Optional[Tuple[List[int], List[int]]]:
    """Return a violating (prefix, cycle) lasso if NOT satisfied, else None.
        A (safe U goal)
    """
    from helpers.witness_ctl import find_witness_A_safe_U_goal

    checker, kripke, _stats, uid_to_idx, idx_to_uid, _cells, _tm = absys.build_kripke()

    sat = {int(s) for s in checker.model_check_kripke(kripke)}
    init_idxs = [uid_to_idx[u] for u in init_uids if u in uid_to_idx]
    if not init_idxs:
        init_idxs = list(kripke.S0)

    if set(init_idxs).issubset(sat):
        return None

    wit = find_witness_A_safe_U_goal(kripke, init_idxs)
    if wit is None:
        return None

    pref_i, cyc_i = wit
    pref_u = [idx_to_uid[i] for i in pref_i if i in idx_to_uid]
    cyc_u = [idx_to_uid[i] for i in cyc_i if i in idx_to_uid]
    return pref_u, cyc_u


def _extract_prefix_cycle_from_run(run) -> Tuple[Optional[List[int]], Optional[List[int]]]:
    """Parse Spot run in *line-oriented* format.

    Example:
      Prefix:
        53
        | goal & !unsafe
        89
        | !unsafe
      Cycle:
        -1
        | unsafe
    """
    s = str(run)
    prefix: List[int] = []
    cycle: List[int] = []

    mode: Optional[str] = None  # "prefix" or "cycle"

    for raw in s.splitlines():
        line = raw.strip()
        if not line:
            continue

        low = line.lower()
        if low.startswith("prefix"):
            mode = "prefix"
            continue
        if low.startswith("cycle") or low.startswith("loop"):
            mode = "cycle"
            continue

        # Skip label lines like "| goal & !unsafe"
        if line.startswith("|"):
            continue

        # Spot prints the state name on its own line (your names are integers)
        m = re.match(r"^-?\d+\s*$", line)
        if m and mode is not None:
            if mode == "prefix":
                prefix.append(int(line))
            else:
                cycle.append(int(line))

    if not prefix and not cycle:
        return None, None
    return prefix, cycle


def _expand_lasso(prefix: List[int], cycle: List[int], target_len: int) -> List[int]:
    """Expand (prefix, cycle) into a finite uid sequence of length target_len."""
    if not prefix and not cycle:
        return []

    seq = list(prefix)
    if not cycle:
        return seq[:target_len]

    while len(seq) < target_len:
        for u in cycle:
            seq.append(u)
            if len(seq) >= target_len:
                break
    return seq[:target_len]


# Clarke-style validation: reachable-set propagation

def validate_lasso_by_set_propagation(
    absys: Abstraction,
    prefix: List[int],
    cycle: List[int],
    *,
    max_steps: int = 100,
    goal_all_fn=None,
    ) -> ValidationResult:
    """Validate the model-checker lasso using Clarke-style reachable-set propagation.

    Returns:
      feasible=True  -> real CE
      feasible=False -> spurious CE (apply refinement)
    """
    uid_path = _expand_lasso(prefix, cycle, max_steps + 1)
    if not uid_path:
        return ValidationResult(True, None, [], [], [])

    domain = absys.part.domain
    dyn = absys.dyn_by_action["step"]

    # Build X-rect list as we go (None for OUT/non-leaf states)
    x_rects: List[Optional[Rect]] = []
    for u in uid_path:
        if u == absys.OUT_UID:
            x_rects.append(None)
        elif u in absys.part.leaves:
            x_rects.append(absys.part.leaves[u].rect)
        else:
            x_rects.append(None)

    # Find starting leaf uid (some runs may include non-leaf helper states; we skip them)
    start_idx = 0
    while start_idx < len(uid_path) and uid_path[start_idx] not in absys.part.leaves:
        if uid_path[start_idx] == absys.OUT_UID:
            # starts in unsafe
            return ValidationResult(True, None, uid_path, [], x_rects)
        start_idx += 1

    if start_idx >= len(uid_path):
        return ValidationResult(True, None, uid_path, [], x_rects)

    u0 = uid_path[start_idx]
    R = absys.part.leaves[u0].rect  # R_0 := X_0
    reach_sets: List[Rect] = [R]

    # Propagate along the remaining uid_path
    for k in range(start_idx + 1, len(uid_path)):
        postR = dyn.image_bbox(R)

        # Unsafe reachable => REAL CE (terminate immediately)
        if not domain.contains_rect(postR):
            return ValidationResult(True, None, uid_path, reach_sets, x_rects)

        uk = uid_path[k]

        # If model-checker expects OUT but Post(R) stays within domain => spurious fracture at k
        if uk == absys.OUT_UID:
            return ValidationResult(False, k, uid_path, reach_sets, x_rects)

        # If uk is not a leaf uid, we cannot continue set propagation safely
        if uk not in absys.part.leaves:
            return ValidationResult(True, None, uid_path, reach_sets, x_rects)

        Xk = absys.part.leaves[uk].rect
        R_next = postR.intersection(Xk)
        if R_next is None:
            # First empty intersection => fracture
            return ValidationResult(False, k, uid_path, reach_sets, x_rects)

        R = R_next
        reach_sets.append(R)

    # No fracture up to horizon; apply bounded-time proxy on corners from the initial cell
    spurious = _bounded_time_goal_proxy(absys, u0, max_steps=max_steps, goal_all_fn=goal_all_fn)
    if spurious:
        return ValidationResult(False, 0, uid_path, reach_sets, x_rects)
    return ValidationResult(True, None, uid_path, reach_sets, x_rects)


def _bounded_time_goal_proxy(
    absys: Abstraction,
    init_uid: int,
    *,
    max_steps: int = 100,
    goal_all_fn=None,
) -> bool:
    dyn = absys.dyn_by_action["step"]
    domain = absys.part.domain

    rect0 = absys.part.leaves[init_uid].rect
    pts = np.array(
        [
            [rect0.xmin, rect0.ymin],
            [rect0.xmin, rect0.ymax],
            [rect0.xmax, rect0.ymin],
            [rect0.xmax, rect0.ymax],
        ],
        dtype=float,
    )

    if goal_all_fn is None:
        center, r2 = _infer_goal_ball(absys)

        def goal_all_fn(P: np.ndarray) -> bool:
            d = P - center[None, :]
            return bool(np.all(np.sum(d * d, axis=1) <= r2))

    if goal_all_fn(pts):
        return True

    for _ in range(max_steps):
        pts = np.array([dyn.dynamics(p) for p in pts], dtype=float)

        # OOB => REAL
        for p in pts:
            if not domain.contains_point(float(p[0]), float(p[1])):
                return False

        # All corners in goal => spurious
        if goal_all_fn(pts):
            return True

    return False


def _infer_goal_ball(absys: Abstraction) -> Tuple[np.ndarray, float]:
    """Infer (center, radius^2) for the bounded proxy."""
    centers: List[np.ndarray] = []
    for uid, node in absys.part.leaves.items():
        aps = absys.ap_labeler(node.rect)
        if "goal" in aps:
            cx, cy = node.rect.center()
            centers.append(np.array([cx, cy], dtype=float))
    if centers:
        c = np.mean(np.stack(centers, axis=0), axis=0)
        return c.astype(float), 1.0
    return np.array([0.0, 0.0], dtype=float), 1.0


# Clarke-style refinement operators: ρ_split + ρ_purge

def refine_clarke(
    absys: Abstraction,
    val: ValidationResult,
    *,
    min_cell_width: float = 0.0,
    min_cell_height: float = 0.0,
    max_refine_depth: Optional[int] = None,
    verbose: bool = False,
) -> int:
    """Apply Clarke refinement for a spurious counterexample."""
    if val.fracture_k is None:
        return 0

    k = val.fracture_k
    path = val.uid_path
    reach = val.reach_sets
    x_rects = val.x_rects

    if k <= 0 or k >= len(path):
        return 0

    failures = 0

    # --- ρ_split for l=1..k-1 ---
    # We use the center of R_l to locate the current leaf containing it (robust to prior splits).
    for l in range(1, min(k, len(reach))):
        Rl = reach[l]
        cx, cy = Rl.center()
        uid = absys.part.leaf_uid_for_point(cx, cy)
        if uid is None or uid not in absys.part.leaves:
            continue

        Xl = absys.part.leaves[uid].rect
        if _rect_equal(Rl, Xl):
            continue

        # Ensure target is contained in the leaf we are splitting
        target = Rl if Xl.contains_rect(Rl) else (Xl.intersection(Rl) or None)
        if target is None:
            continue

        out_uid = absys.isolate_subrect(
            uid,
            target,
            min_w=min_cell_width,
            min_h=min_cell_height,
            max_depth=max_refine_depth,
        )
        if out_uid is None:
            failures += 1
            if verbose:
                print(f"[REFINE] isolate_subrect failed at l={l}, uid={uid}")
        else:
            if verbose:
                print(f"[REFINE] isolated R_{l} inside uid={uid} -> new_uid={out_uid}")

    # --- ρ_purge on the fractured transition (s_{k-1} -> s_k) ---
    # Determine a robust "from" state: the leaf that contains the center of R_{k-1}.
    Rkm1 = reach[min(k - 1, len(reach) - 1)]
    u_from = absys.part.leaf_uid_for_point(*Rkm1.center())
    if u_from is None:
        u_from = path[k - 1]

    # Determine X_k rectangle from validation-time snapshot if available.
    Xk_rect = x_rects[k]
    if Xk_rect is None:
        # If Spot's k-th state isn't a leaf/has no rect, purge only the direct edge (best-effort)
        absys.purge_transition(u_from, path[k])
    else:
        # Purge edges into all leaves that intersect X_k (robust to X_k having been split elsewhere)
        targets = set(absys.part.query_intersecting_leaves(Xk_rect))
        if not targets:
            # If nothing intersects, also purge direct edge (best-effort)
            absys.purge_transition(u_from, path[k])
        else:
            for v in targets:
                absys.purge_transition(u_from, v)

    # Rebuild transitions after splits + purges
    absys.rebuild_all_transitions()

    return failures


def _rect_equal(a: Rect, b: Rect, eps: float = 1e-12) -> bool:
    return (
        abs(a.xmin - b.xmin) <= eps
        and abs(a.xmax - b.xmax) <= eps
        and abs(a.ymin - b.ymin) <= eps
        and abs(a.ymax - b.ymax) <= eps
    )


# Main CEGAR loop (GLOBAL, step-only)

def run_cegar(
    absys: Abstraction,
    init_uids: Set[int],
    phi: str,
    *,
    max_iters: int = 200,
    merge_actions: bool = True,         # accepted for compatibility; step-only anyway
    max_steps_proxy: int = 100,
    goal_all_fn=None,
    min_cell_width: float = 0.0,
    min_cell_height: float = 0.0,
    max_refine_depth: Optional[int] = None,
    verbose: bool = True,
) -> CEGARResult:
    """Global Clarke-style CEGAR (step-only) with bounded-time proxy.

    Loop:
      1) Build abstraction transitions (conservative, includes OUT if unsafe reachable).
      2) Model check phi; if satisfied -> VERIFIED.
      3) Otherwise extract lasso (prefix, cycle) for !phi.
      4) Validate via reachable-set propagation:
            - unsafe reachable -> REAL CE -> terminate NOT VERIFIED
            - first fracture k -> spurious
      5) If spurious -> apply ρ_split for l=1..k-1 and then ρ_purge for (k-1 -> k).
      6) Repeat.
    """
    absys.rebuild_all_transitions()

    ignored = 0
    last_cex: Optional[Tuple[List[int], List[int]]] = None

    for it in range(max_iters):
        ce = ctl_get_counterexample_lasso(absys, init_uids, merge_actions=merge_actions)
        if ce is None:
            if verbose:
                print(f"[CEGAR] VERIFIED (iter {it}).")
            return CEGARResult(True, it, ignored, last_cex)

        prefix, cycle = ce
        last_cex = (prefix, cycle)

        if verbose:
            print(f"[CEGAR] Abstract counterexample found (iter {it}).")
            print(f"  prefix: {prefix}")
            print(f"  cycle:  {cycle}")

        val = validate_lasso_by_set_propagation(
            absys,
            prefix,
            cycle,
            max_steps=max_steps_proxy,
            goal_all_fn=goal_all_fn,
        )

        if val.feasible:
            if verbose:
                print("[CEGAR] REAL counterexample (unsafe reachable / bounded proxy violated).")
            return CEGARResult(False, it + 1, ignored, last_cex)

        # Spurious: refine
        if verbose:
            print(f"[CEGAR] Spurious counterexample. fracture_k={val.fracture_k}. Applying Clarke refinement.")

        failures = refine_clarke(
            absys,
            val,
            min_cell_width=min_cell_width,
            min_cell_height=min_cell_height,
            max_refine_depth=max_refine_depth,
            verbose=verbose,
        )
        ignored += failures

    if verbose:
        print("[CEGAR] Iteration budget reached; returning NOT VERIFIED at this precision.")
    return CEGARResult(False, max_iters, ignored, last_cex)

