from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, List, Optional, Set, Tuple, Literal
import numpy as np

from unicycle_partition_3d import Box3D, Partition3D
from unicycle_dyn import UnicycleClosedLoop
from helpers.math_utils import (
    unwrap_theta_interval_options,
    prepare_convex_hull_lp,
    convex_hull_intersects_box,
)

from spatial_index import SpatialHash3D

Method = Literal["aabb", "poly"]


@dataclass
class TransitionGraph:
    # leaf uid -> set of leaf uids (successors)
    succ: Dict[int, Set[int]]
    pred: Dict[int, Set[int]]

    def __init__(self) -> None:
        self.succ = {}
        self.pred = {}

    def set_succ(self, u: int, vs: Set[int]) -> None:
        # remove old edges
        old = self.succ.get(u, set())
        for v in old:
            self.pred.get(v, set()).discard(u)
        self.succ[u] = set(vs)
        for v in vs:
            self.pred.setdefault(v, set()).add(u)

    def successors(self, u: int) -> Set[int]:
        return set(self.succ.get(u, set()))

    def predecessors(self, u: int) -> Set[int]:
        return set(self.pred.get(u, set()))


class UnicycleAbstraction:
    OUT_UID = -1

    def __init__(
        self,
        part: Partition3D,
        dyn: UnicycleClosedLoop,
        *,
        init_method: Method = "aabb",
        refine_method: Method = "aabb",
        allow_self_loops: bool = True,
        tol: float = 1e-9,
        bins: Tuple[int, int, int] = (40, 32, 40),
        # Backward-compat: allow old code to pass method=...
        method: Optional[Method] = None,
    ):
        self.part = part
        self.dyn = dyn

        if method is not None:
            init_method = method
            refine_method = method

        self.init_method: Method = init_method
        self.refine_method: Method = refine_method

        # Per-leaf transition method tag:
        # - all leaves start with init_method
        # - all newly created refined children get refine_method
        self.leaf_method: Dict[int, Method] = {}

        self.allow_self_loops = bool(allow_self_loops)
        self.tol = float(tol)
        self.tr = TransitionGraph()

        # Candidate filtering over current leaf AABBs.
        root = self.part.get_box(self.part.root_uid)
        self._index = SpatialHash3D(
            root.p_lo, root.p_hi,
            root.q_lo, root.q_hi,
            root.th_lo, root.th_hi,
            nb_p=int(bins[0]),
            nb_q=int(bins[1]),
            nb_th=int(bins[2]),
        )

        # Cache per-leaf one-step image info so that after a split we can update
        # predecessor transitions without recomputing predecessor images.
        self._img_cache: Dict[int, Dict[str, object]] = {}

    def rebuild_all(self, ap_labeler, verbose: bool = False) -> None:
        """
        Build the transition graph for all current leaves using init_method.
        """
        # rebuild spatial index from scratch
        self._index.clear()
        for u in self.part.leaves.keys():
            self._index.insert(u, self.part.get_box(u))

        # tag all leaves with init_method
        self.leaf_method = {u: self.init_method for u in self.part.leaves.keys()}

        # reset caches + transitions
        self._img_cache = {}
        self.tr = TransitionGraph()

        # OUT state self-loop
        self.tr.set_succ(self.OUT_UID, {self.OUT_UID})

        leaves = list(self.part.leaves.keys())
        if verbose:
            print(f"[abs] rebuild_all: {len(leaves)} leaves (init={self.init_method}, refine={self.refine_method})")

        for u in leaves:
            self._rebuild_outgoing(u)

    def _rebuild_outgoing(self, u: int) -> None:
        """
        Rebuild outgoing transitions for leaf u using the per-leaf method tag.
        """
        if u == self.OUT_UID:
            self.tr.set_succ(u, {self.OUT_UID})
            return
        if u not in self.part.leaves:
            # internal node; no outgoing
            self.tr.set_succ(u, set())
            return

        # Determine method for this leaf
        m: Method = self.leaf_method.get(u, self.init_method)

        box_u = self.part.get_box(u)
        next_verts, img_boxes, hits_oob, theta_arc_start_u = self.dyn.image_from_box(box_u)

        # Cache image info for future incremental updates.
        info: Dict[str, object] = {
            "img_boxes": img_boxes,
            "hits_oob": bool(hits_oob),
            "theta_arc_start": float(theta_arc_start_u),
            "method": m,
        }

        # Candidate successors via spatial hash (superset), then exact AABB check.
        cand: Set[int] = set()
        for ib in img_boxes:
            for v in self._index.query_candidates(ib):
                if v in self.part.leaves and self.part.get_box(v).intersects(ib):
                    cand.add(v)

        succs: Set[int] = set()
        if m == "aabb":
            succs = cand
        else:
            verts = np.asarray(next_verts, dtype=float).copy()
            theta_lo = -np.pi
            theta_hi = np.pi
            period = 2.0 * np.pi
            uvals = np.mod(verts[:, 2] - theta_lo, period)
            uvals = np.where(uvals < theta_arc_start_u, uvals + period, uvals)
            verts[:, 2] = theta_lo + uvals
            lp = prepare_convex_hull_lp(verts)
            info["verts"] = verts
            info["lp"] = lp

            for v in cand:
                b = self.part.get_box(v)
                # Unwrap target theta interval into the same frame as verts.
                opts = unwrap_theta_interval_options(
                    b.th_lo, b.th_hi,
                    theta_lo, theta_hi,
                    arc_start_u=float(theta_arc_start_u),
                )
                ok = False
                for (tlo_u, thi_u) in opts:
                    bmin = np.array([b.p_lo, b.q_lo, tlo_u], dtype=float)
                    bmax = np.array([b.p_hi, b.q_hi, thi_u], dtype=float)
                    if convex_hull_intersects_box(bmin, bmax, lp, tol=self.tol):
                        ok = True
                        break
                if ok:
                    succs.add(v)

        if not self.allow_self_loops and u in succs:
            succs.discard(u)

        if hits_oob:
            succs.add(self.OUT_UID)

        # ensure total (no dead-ends): add self-loop if empty
        if len(succs) == 0:
            succs.add(u)

        self._img_cache[u] = info
        self.tr.set_succ(u, succs)

    def refine_split(self, uid: int) -> List[int]:
        # Split leaf into children.
        kids = self.part.refine_oct(uid)
        if not kids:
            return []

        # update spatial index: remove parent leaf, add children
        self._index.remove(uid)
        for k in kids:
            self._index.insert(k, self.part.get_box(k))

        # Remove uid from cache and outgoing map (uid becomes internal).
        self._img_cache.pop(uid, None)
        self.tr.set_succ(uid, set())

        # Parent is no longer a leaf; remove its method tag
        self.leaf_method.pop(uid, None)

        # Tag children with refine_method
        for k in kids:
            self.leaf_method[k] = self.refine_method

        # Build outgoing transitions for each new child (only place we recompute images after a split).
        for k in kids:
            self._rebuild_outgoing(k)

        # Incrementally update predecessors that previously pointed to uid.
        # against the 8 children using cached image information.
        preds = self.tr.predecessors(uid)
        kid_boxes = {k: self.part.get_box(k) for k in kids}

        for p in preds:
            succs = self.tr.successors(p)
            if uid not in succs:
                continue
            succs.discard(uid)

            info = self._img_cache.get(p)
            if info is None:
                # Fallback if cache missing.
                self._rebuild_outgoing(p)
                continue

            img_boxes = info["img_boxes"]

            # Use predecessor's own method tag (hybrid semantics)
            pm: Method = self.leaf_method.get(p, self.init_method)

            if pm == "aabb":
                for k, kb in kid_boxes.items():
                    for ib in img_boxes:
                        if kb.intersects(ib):
                            succs.add(k)
                            break
            else:
                lp = info.get("lp")
                if lp is None:
                    self._rebuild_outgoing(p)
                    continue
                theta_lo = -np.pi
                theta_hi = np.pi
                arc_start_u = float(info.get("theta_arc_start", 0.0))

                for k, kb in kid_boxes.items():
                    # AABB quick reject
                    ok_aabb = False
                    for ib in img_boxes:
                        if kb.intersects(ib):
                            ok_aabb = True
                            break
                    if not ok_aabb:
                        continue

                    opts = unwrap_theta_interval_options(
                        kb.th_lo, kb.th_hi,
                        theta_lo, theta_hi,
                        arc_start_u=arc_start_u,
                    )
                    ok = False
                    for (tlo_u, thi_u) in opts:
                        bmin = np.array([kb.p_lo, kb.q_lo, tlo_u], dtype=float)
                        bmax = np.array([kb.p_hi, kb.q_hi, thi_u], dtype=float)
                        if convex_hull_intersects_box(bmin, bmax, lp, tol=self.tol):
                            ok = True
                            break
                    if ok:
                        succs.add(k)

            if bool(info.get("hits_oob", False)):
                succs.add(self.OUT_UID)

            if (not self.allow_self_loops) and (p in succs):
                succs.discard(p)
            if len(succs) == 0:
                succs.add(p)
            self.tr.set_succ(p, succs)

        return kids
