from __future__ import annotations

import importlib
import sys
from collections import deque
from dataclasses import dataclass
from typing import Callable, Optional, Set, Tuple, List

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches

from abstraction import Rect, Abstraction
from cegar_loop import (
    ctl_get_counterexample_lasso,
    validate_lasso_by_set_propagation,
    refine_clarke,
)

GoalAllFn = Callable[[np.ndarray], bool]

@dataclass
class Classification:
    verified: Set[int]
    refuted: Set[int]
    unknown: Set[int]

def _corners(absys: Abstraction, uid: int) -> np.ndarray:
    r = absys.part.leaves[uid].rect
    return np.array(
        [[r.xmin, r.ymin],
         [r.xmin, r.ymax],
         [r.xmax, r.ymin],
         [r.xmax, r.ymax]],
        dtype=float,
    )


def infer_goal_all_fn(absys: Abstraction) -> GoalAllFn:
    centers = []
    goal_corners = []
    for uid, node in absys.part.leaves.items():
        if uid == absys.OUT_UID:
            continue
        aps = absys.ap_labeler(node.rect)
        if "goal" in aps:
            cx, cy = node.rect.center()
            centers.append([cx, cy])
            goal_corners.append(_corners(absys, uid))

    if centers:
        c = np.mean(np.array(centers, dtype=float), axis=0)
    else:
        c = np.array([0.0, 0.0], dtype=float)

    if goal_corners:
        pts = np.vstack(goal_corners)
        d = pts - c[None, :]
        r2 = float(np.max(np.sum(d * d, axis=1))) + 1e-9
    else:
        r2 = 1.0

    def goal_all(points: np.ndarray) -> bool:
        d = points - c[None, :]
        return bool(np.all(np.sum(d * d, axis=1) <= r2))

    return goal_all


def prove_cell_by_corners(
    absys: Abstraction,
    uid: int,
    goal_all_fn: GoalAllFn,
    *,
    max_steps: int = 100,
) -> bool:
    dyn = absys.dyn_by_action["step"]
    domain = absys.part.domain
    pts = _corners(absys, uid)

    if goal_all_fn(pts):
        return True

    for _ in range(max_steps):
        pts = np.array([dyn.dynamics(p) for p in pts], dtype=float)

        for p in pts:
            if not domain.contains_point(float(p[0]), float(p[1])):
                return False

        if goal_all_fn(pts):
            return True

    return False


def refute_cell_by_corners(
    absys: Abstraction,
    uid: int,
    *,
    max_steps: int = 100,
) -> bool:
    dyn = absys.dyn_by_action["step"]
    domain = absys.part.domain
    pts = _corners(absys, uid)

    # Already out of bounds
    for p in pts:
        if not domain.contains_point(float(p[0]), float(p[1])):
            return True

    for _ in range(max_steps):
        pts = np.array([dyn.dynamics(p) for p in pts], dtype=float)

        for p in pts:
            if not domain.contains_point(float(p[0]), float(p[1])):
                return True

    return False


def effective_verified_for_plot(absys: Abstraction, verified: Set[int]) -> Set[int]:
    """Plotting-only: a current leaf is green if it OR any ancestor was verified."""
    eff: Set[int] = set()
    for uid, node in absys.part.leaves.items():
        if uid == absys.OUT_UID:
            continue
        n = node
        while n is not None:
            if n.uid in verified:
                eff.add(uid)
                break
            n = n.parent
    return eff


def plot_classification(
    absys: Abstraction,
    domain: Rect,
    cls: Classification,
    *,
    title: str,
    save_path: str,
) -> None:
    fig, ax = plt.subplots()
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlim(domain.xmin, domain.xmax)
    ax.set_ylim(domain.ymin, domain.ymax)
    ax.set_title(title)

    for uid in sorted(absys.part.leaves.keys()):
        if uid == absys.OUT_UID:
            continue
        r = absys.part.leaves[uid].rect

        if uid in cls.verified:
            face = (0.1, 0.8, 0.1, 0.35)   # green
        elif uid in cls.refuted:
            face = (0.9, 0.1, 0.1, 0.35)   # red
        else:
            face = (0.9, 0.9, 0.1, 0.35)   # yellow (unknown)

        ax.add_patch(
            patches.Rectangle(
                (r.xmin, r.ymin),
                r.xmax - r.xmin,
                r.ymax - r.ymin,
                linewidth=0.2,
                edgecolor=(0, 0, 0, 0.15),
                facecolor=face,
            )
        )

    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    print(f"[PLOT] saved: {save_path}")
    plt.close(fig)


def classify_state_space_worklist(
    absys: Abstraction,
    phi: str,
    *,
    goal_all_fn: GoalAllFn,
    budget_steps: int = 10000,
    max_steps_validator: int = 100,
    min_cell_width: float = 0.0005,
    min_cell_height: float = 0.0005,
    max_refine_depth: int = 12,
    verbose_every: int = 200,
) -> Tuple[Classification, dict]:
    verified: Set[int] = set()
    refuted: Set[int] = set()

    all_leaves = [u for u in absys.part.leaves.keys() if u != absys.OUT_UID]

    # Trivial goal: immediately verified
    triv_goal = set()
    for u in all_leaves:
        if goal_all_fn(_corners(absys, u)):
            verified.add(u)
            triv_goal.add(u)

    work = deque([u for u in sorted(all_leaves) if u not in triv_goal])

    used_steps = 0
    refine_ops = 0
    ignored = 0
    real_cex = 0

    while work and used_steps < budget_steps:
        u = work.popleft()

        # Leaf might have been split away
        if u not in absys.part.leaves:
            continue

        if u in verified or u in refuted:
            continue

        used_steps += 1

        # (A) try to PROVE (sufficient proof rule)
        proven = prove_cell_by_corners(absys, u, goal_all_fn, max_steps=max_steps_validator)

        # (A0) trivial REFUTE check (fast concrete falsification)
        if refute_cell_by_corners(absys, u, max_steps=max_steps_validator):
            refuted.add(u)
            real_cex += 1
            continue

        if proven:
            verified.add(u)

        # (B) ask abstraction for a counterexample from this single cell
        ce = ctl_get_counterexample_lasso(absys, {u}, merge_actions=True)
        if ce is None:
            # No abstract cex for this init cell.
            # If not proven, we still don't know: re-check later because refinement elsewhere may help.
            if not proven:
                work.append(u)
            continue

        prefix, cycle = ce
        vr = validate_lasso_by_set_propagation(
            absys,
            prefix,
            cycle,
            max_steps=max_steps_validator,
            goal_all_fn=goal_all_fn,
        )

        if vr.feasible:
            # REAL counterexample: mark refuted.
            refuted.add(u)
            real_cex += 1
            # If we had marked it verified because proven, undo that.
            verified.discard(u)
            continue

        # Spurious: refine Clarke-style
        old_leaves = set(absys.part.leaves.keys())
        failures = refine_clarke(
            absys,
            vr,
            min_cell_width=min_cell_width,
            min_cell_height=min_cell_height,
            max_refine_depth=max_refine_depth,
            verbose=False,
        )
        ignored += failures
        refine_ops += 1

        # Enqueue new leaves created by splitting
        new_leaves = set(absys.part.leaves.keys())
        added = new_leaves - old_leaves
        for a in added:
            if a != absys.OUT_UID:
                work.append(a)

        # Re-check this cell later; it may now have no abstract cex, or become refutable.
        work.append(u)


        if verbose_every and (used_steps % verbose_every == 0):
            unknown_now = len([x for x in absys.part.leaves.keys()
                               if x != absys.OUT_UID and x not in verified and x not in refuted])
            print(
                f"[PROGRESS] steps={used_steps}/{budget_steps} "
                f"verified={len(verified)} refuted={len(refuted)} unknown={unknown_now} "
                f"leaves={len(absys.part.leaves)} refine_ops={refine_ops} "
                f"real_cex={real_cex} ignored={ignored}"
            )

    unknown_final = {x for x in absys.part.leaves.keys()
                     if x != absys.OUT_UID and x not in verified and x not in refuted}

    cls = Classification(verified=verified, refuted=refuted, unknown=unknown_final)

    stats = {
        "used_steps": used_steps,
        "budget_steps": budget_steps,
        "refine_ops": refine_ops,
        "ignored": ignored,
        "real_cex": real_cex,
        "total_leaves": len(absys.part.leaves),
    }
    return cls, stats


def main() -> None:
    # Usage:
    #   python unknown_worklist.py systems.synthetic 40 40 10000
    sysmod = sys.argv[1] if len(sys.argv) > 1 else "helpers.systems.synthetic"
    nx = int(sys.argv[2]) if len(sys.argv) > 2 else 40
    ny = int(sys.argv[3]) if len(sys.argv) > 3 else 40
    budget = int(sys.argv[4]) if len(sys.argv) > 4 else 10000

    mod = importlib.import_module(sysmod)
    spec = mod.build(nx=nx, ny=ny)

    absys: Abstraction = spec.absys
    domain: Rect = spec.domain
    phi: str = spec.phi

    goal_all_fn: Optional[GoalAllFn] = getattr(spec, "goal_all_fn", None)
    if goal_all_fn is None:
        goal_all_fn = infer_goal_all_fn(absys)

    print("[WORKLIST]")
    print(f"system={sysmod} budget_steps={budget} base_leaves={len(absys.part.leaves)}")

    cls, stats = classify_state_space_worklist(
        absys,
        phi,
        goal_all_fn=goal_all_fn,
        budget_steps=budget,
    )

    print("\n[FINAL]")
    print(f"Verified: {len(cls.verified)}")
    print(f"Refuted : {len(cls.refuted)}")
    print(f"Unknown : {len(cls.unknown)}")
    print(f"Total leaves: {stats['total_leaves']}")
    print(f"Used steps: {stats['used_steps']}/{stats['budget_steps']} refine_ops={stats['refine_ops']} "
          f"real_cex={stats['real_cex']} ignored={stats['ignored']}")

    verified_for_plot = effective_verified_for_plot(absys, cls.verified)

    plot_classification(
        absys,
        domain,
        Classification(verified=verified_for_plot, refuted=cls.refuted, unknown=cls.unknown),
        title=f"{sysmod} (worklist classification)",
        save_path="classification_worklist.png",
    )


if __name__ == "__main__":
    main()
