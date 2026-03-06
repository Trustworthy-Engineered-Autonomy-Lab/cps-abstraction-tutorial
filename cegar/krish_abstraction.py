from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Set, Tuple

import numpy as np

from abstraction import Rect, RectPartition, TransitionRelation, Abstraction


class StepDynamics:
    """Provides the interface Ethan's validator expects."""
    def __init__(self, system):
        self.system = system

    def dynamics(self, p: np.ndarray) -> np.ndarray:
        p2 = np.asarray(p, dtype=float).reshape(1, 2)
        nxt = np.asarray(self.system.step(p2), dtype=float)
        return nxt.reshape(2,)

    def image_bbox(self, r: Rect) -> Rect:
        corners = np.array(
            [
                [r.xmin, r.ymin],
                [r.xmin, r.ymax],
                [r.xmax, r.ymin],
                [r.xmax, r.ymax],
            ],
            dtype=float,
        )
        nxt = np.asarray(self.system.step(corners), dtype=float)
        xmin, ymin = np.min(nxt, axis=0)
        xmax, ymax = np.max(nxt, axis=0)
        return Rect(float(xmin), float(xmax), float(ymin), float(ymax))


@dataclass
class KrishAbstraction:

    part: RectPartition
    system: object
    method: str = "AABB"   # "AABB" | "POLY" | "SAMPLE"

    OUT_UID: int = -1

    def __post_init__(self):
        self.tr = TransitionRelation()
        self._purged: Dict[Tuple[int, str], Set[int]] = {}
        self.dyn_by_action = {"step": StepDynamics(self.system)}
        self.ap_labeler = self._ap_labeler

    def _ap_labeler(self, rect: Optional[Rect]) -> Set[str]:
        if rect is None:
            return {"unsafe"}

        # Default to Krish parameters (5,5), r=2
        center = np.array([5.0, 5.0], dtype=float)
        radius = 2.0
        r2 = radius * radius

        corners = np.array(
            [
                [rect.xmin, rect.ymin],
                [rect.xmin, rect.ymax],
                [rect.xmax, rect.ymin],
                [rect.xmax, rect.ymax],
            ],
            dtype=float,
        )
        d = corners - center[None, :]
        ok = np.all(np.sum(d * d, axis=1) <= r2)
        return {"goal"} if bool(ok) else set()

    # -------------------------
    # Purge + isolate hooks
    # -------------------------
    def purge_transition(self, u: int, v: int, action: str = "step") -> None:
        self._purged.setdefault((u, action), set()).add(v)

    def _apply_purge(self, u: int, action: str, succ: Set[int]) -> Set[int]:
        banned = self._purged.get((u, action))
        if not banned:
            return succ
        return {s for s in succ if s not in banned}

    def isolate_subrect(
        self,
        uid: int,
        target: Rect,
        *,
        min_w: float = 0.0,
        min_h: float = 0.0,
        max_depth: Optional[int] = None,
    ) -> Optional[int]:
        shim = Abstraction.__new__(Abstraction)
        shim.part = self.part
        shim.OUT_UID = self.OUT_UID
        shim.tr = self.tr
        shim._purged = self._purged
        return Abstraction.isolate_subrect(shim, uid, target, min_w=min_w, min_h=min_h, max_depth=max_depth)

    def rebuild_all_transitions(self) -> None:
        from helpers.partitioning_generalized import (
            compute_transitions_AABB_partition,
            compute_transitions_poly_partition,
            compute_transitions_sample_partition,
        )

        if self.method.upper() == "POLY":
            tr_uid = compute_transitions_poly_partition(self.part, self.system)
        elif self.method.upper() == "SAMPLE":
            tr_uid = compute_transitions_sample_partition(self.part, self.system, n_samples=256, rng_seed=0)
        else:
            tr_uid = compute_transitions_AABB_partition(self.part, self.system)

        self.tr = TransitionRelation()
        self.tr.set_succ(self.OUT_UID, "step", {self.OUT_UID})

        domain = self.part.domain

        for u in sorted(self.part.leaves.keys()):
            if u == self.OUT_UID:
                continue
            succ = set(tr_uid.get(u, set()))

            # Add OUT conservatively if any stepped corner exits domain
            X = self.part.leaves[u].rect
            corners = np.array(
                [[X.xmin, X.ymin], [X.xmin, X.ymax], [X.xmax, X.ymin], [X.xmax, X.ymax]],
                dtype=float,
            )
            nxt = np.asarray(self.system.step(corners), dtype=float)
            oob = np.any(
                (nxt[:, 0] < domain.xmin)
                | (nxt[:, 0] > domain.xmax)
                | (nxt[:, 1] < domain.ymin)
                | (nxt[:, 1] > domain.ymax)
            )
            if oob:
                succ.add(self.OUT_UID)

            if not succ:
                succ = {self.OUT_UID}

            succ = self._apply_purge(u, "step", succ)
            if not succ:
                succ = {self.OUT_UID}

            self.tr.set_succ(u, "step", succ)

    def build_kripke(self, Checker=None):
        from helpers.model_checking_tools import SyntheticModelChecker

        # Ensure transitions exist
        if not getattr(self.tr, "succ", None) or len(self.tr.succ) == 0:
            self.rebuild_all_transitions()

        # Include current leaves plus OUT sink
        leaf_ids = sorted(self.part.leaves.keys())
        if self.OUT_UID in leaf_ids:
            leaf_ids.remove(self.OUT_UID)
        leaf_ids.append(self.OUT_UID)

        uid_to_idx = {uid: i for i, uid in enumerate(leaf_ids)}
        idx_to_uid = {i: uid for uid, i in uid_to_idx.items()}

        cells = np.zeros((len(leaf_ids), 4), dtype=float)
        d = self.part.domain
        for i, uid in enumerate(leaf_ids):
            if uid == self.OUT_UID:
                cells[i, :] = (d.xmax + 1.0, d.xmax + 2.0, d.ymax + 1.0, d.ymax + 2.0)
            else:
                r = self.part.leaves[uid].rect
                cells[i, :] = (r.xmin, r.xmax, r.ymin, r.ymax)

        transition_map: List[Set[int]] = [set() for _ in range(len(leaf_ids))]
        out_i = uid_to_idx[self.OUT_UID]

        for u in leaf_ids:
            ui = uid_to_idx[u]
            succ_u = self.tr.succ.get(u, {}).get("step", set())

            for v in succ_u:
                if v in uid_to_idx:
                    transition_map[ui].add(uid_to_idx[v])

            # Totality
            if not transition_map[ui]:
                transition_map[ui].add(out_i if u != self.OUT_UID else ui)

        transition_map[out_i].add(out_i)

        # checker = SyntheticModelChecker(self.system)
        if Checker is None:
            Checker = SyntheticModelChecker
        checker = Checker(self.system)
        kripke, stats = checker.create_kripke(cells, transition_map)
        return checker, kripke, stats, uid_to_idx, idx_to_uid, cells, transition_map
