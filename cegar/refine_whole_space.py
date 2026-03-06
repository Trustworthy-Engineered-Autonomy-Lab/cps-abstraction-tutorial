from __future__ import annotations

import importlib
import sys
import time
import random
from dataclasses import dataclass
from typing import Optional, Set, Callable

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches

from abstraction import Abstraction, Rect
from cegar_loop import ctl_get_counterexample_lasso, validate_lasso_by_set_propagation

GoalAllFn = Callable[[np.ndarray], bool]

@dataclass
class RegionClassification:
    verified: Set[int]
    refuted: Set[int]
    unknown: Set[int]

def _cell_area(absys: Abstraction, uid: int) -> float:
    r = absys.part.leaves[uid].rect
    return (r.xmax - r.xmin) * (r.ymax - r.ymin)

def sample_leaf_uids_by_area(absys: Abstraction, k: int, seed: int) -> list[int]:
    rng = random.Random(seed)
    uids = [u for u in absys.part.leaves.keys() if u != absys.OUT_UID]
    if not uids:
        return []
    weights = [_cell_area(absys, u) for u in uids]
    total = sum(weights)
    if total <= 0:
        rng.shuffle(uids)
        return uids[: min(k, len(uids))]
    chosen: list[int] = []
    alive = list(zip(uids, weights))
    for _ in range(min(k, len(alive))):
        s = sum(w for _, w in alive)
        x = rng.random() * s
        acc = 0.0
        for i, (u, w) in enumerate(alive):
            acc += w
            if acc >= x:
                chosen.append(u)
                alive.pop(i)
                break
    return chosen

def classify_all_leaves_once(
    absys: Abstraction,
    phi: str,
    *,
    goal_all_fn: Optional[GoalAllFn] = None,
    sample_k: Optional[int] = None,
    seed: int = 0,
    verbose_every: int = 200,
) -> RegionClassification:
    verified: Set[int] = set()
    refuted: Set[int] = set()
    unknown: Set[int] = set()

    leaves = [u for u in absys.part.leaves.keys() if u != absys.OUT_UID]
    if sample_k is not None:
        leaves = sample_leaf_uids_by_area(absys, sample_k, seed)

    def corners(uid: int) -> np.ndarray:
        r = absys.part.leaves[uid].rect
        return np.array(
            [[r.xmin, r.ymin], [r.xmin, r.ymax], [r.xmax, r.ymin], [r.xmax, r.ymax]],
            dtype=float,
        )

    t0 = time.time()
    for i, init in enumerate(leaves, 1):
        if goal_all_fn is not None and goal_all_fn(corners(init)):
            verified.add(init)
            continue

        cex = ctl_get_counterexample_lasso(absys, {init}, merge_actions=True)
        if cex is None:
            verified.add(init)
        else:
            prefix, cycle = cex
            vr = validate_lasso_by_set_propagation(
                absys, prefix, cycle, max_steps=100, goal_all_fn=goal_all_fn
            )
            if vr.feasible:
                refuted.add(init)
            else:
                unknown.add(init)

        if verbose_every and (i % verbose_every == 0):
            dt = time.time() - t0
            print(
                f"[CLASSIFY] {i}/{len(leaves)} "
                f"ver={len(verified)} ref={len(refuted)} unk={len(unknown)} "
                f"elapsed={dt:.1f}s"
            )

    return RegionClassification(verified, refuted, unknown)

def plot_classification(
    absys: Abstraction,
    domain: Rect,
    cls: RegionClassification,
    *,
    title: str = "",
    save_path: Optional[str] = None,
    show: bool = True,
):
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
        elif uid in cls.unknown:
            face = (0.9, 0.9, 0.1, 0.35)   # yellow
        else:
            face = (0.6, 0.6, 0.6, 0.15)

        rect = patches.Rectangle(
            (r.xmin, r.ymin),
            r.xmax - r.xmin,
            r.ymax - r.ymin,
            linewidth=0.2,
            edgecolor=(0, 0, 0, 0.15),
            facecolor=face,
        )
        ax.add_patch(rect)

    if save_path is not None:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        print(f"[PLOT] saved: {save_path}")

    if show:
        plt.show()
    else:
        plt.close(fig)

def main() -> None:
    sysmod = sys.argv[1] if len(sys.argv) > 1 else "helpers.systems.synthetic"
    nx = int(sys.argv[2]) if len(sys.argv) > 2 else 60
    ny = int(sys.argv[3]) if len(sys.argv) > 3 else 60
    sample_k = int(sys.argv[4]) if len(sys.argv) > 4 else 2000

    mod = importlib.import_module(sysmod)
    spec = mod.build(nx=nx, ny=ny)

    cls = classify_all_leaves_once(
        spec.absys,
        spec.phi,
        goal_all_fn=getattr(spec, "goal_all_fn", None),
        sample_k=sample_k,
        seed=0,
        verbose_every=250,
    )

    plot_classification(
        spec.absys,
        spec.domain,
        cls,
        title=f"Classification (sampled) — {sysmod}",
        save_path="classification_sampled.png",
        show=True,
    )

if __name__ == "__main__":
    main()
