# unicycle_partition_3d.py
from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, List, Optional, Set, Tuple

import numpy as np


def wrap_to_pi(angle: float) -> float:
    return float((angle + np.pi) % (2 * np.pi) - np.pi)


def theta_min_arc_intervals(thetas: np.ndarray, *, eps: float = 1e-12) -> List[Tuple[float, float]]:
    th = np.asarray(thetas, dtype=float)
    if th.size == 0:
        return [(-np.pi, np.pi)]
    th = np.array([wrap_to_pi(t) for t in th], dtype=float)
    if th.size == 1:
        v = float(th[0])
        return [(v, v)]

    # Work on [0, 2pi)
    u = np.sort(th + np.pi)
    two_pi = 2.0 * np.pi

    gaps = np.diff(np.r_[u, u[0] + two_pi])
    k = int(np.argmax(gaps))

    start_u = float(u[(k + 1) % u.size])
    end_u = float(u[k])
    arc_len = (end_u - start_u) % two_pi

    if arc_len >= two_pi - eps:
        return [(-np.pi, np.pi)]

    end_u2 = start_u + arc_len
    if end_u2 <= two_pi + eps:
        lo = start_u - np.pi
        hi = min(end_u2, two_pi) - np.pi
        return [(float(lo), float(hi))]

    # Wraps across cut -> split
    lo1, hi1 = start_u - np.pi, np.pi
    lo2, hi2 = -np.pi, (end_u2 - two_pi) - np.pi
    return [(float(lo2), float(hi2)), (float(lo1), float(hi1))]


@dataclass
class Box3D:
    p_lo: float
    p_hi: float
    q_lo: float
    q_hi: float
    th_lo: float
    th_hi: float

    def corners(self) -> np.ndarray:
        ps = [self.p_lo, self.p_hi]
        qs = [self.q_lo, self.q_hi]
        ts = [self.th_lo, self.th_hi]
        pts = []
        for p in ps:
            for q in qs:
                for t in ts:
                    pts.append([p, q, t])
        return np.asarray(pts, dtype=float)  # (8,3)

    def intersects(self, other: "Box3D") -> bool:
        return not (
            self.p_hi <= other.p_lo or self.p_lo >= other.p_hi or
            self.q_hi <= other.q_lo or self.q_lo >= other.q_hi or
            self.th_hi <= other.th_lo or self.th_lo >= other.th_hi
        )

    def intersection(self, other: "Box3D") -> Optional["Box3D"]:
        p_lo = max(self.p_lo, other.p_lo)
        p_hi = min(self.p_hi, other.p_hi)
        q_lo = max(self.q_lo, other.q_lo)
        q_hi = min(self.q_hi, other.q_hi)
        t_lo = max(self.th_lo, other.th_lo)
        t_hi = min(self.th_hi, other.th_hi)
        if p_hi <= p_lo or q_hi <= q_lo or t_hi <= t_lo:
            return None
        return Box3D(p_lo, p_hi, q_lo, q_hi, t_lo, t_hi)

    def widths(self) -> Tuple[float, float, float]:
        return (self.p_hi - self.p_lo, self.q_hi - self.q_lo, self.th_hi - self.th_lo)


@dataclass
class Node3D:
    uid: int
    box: Box3D
    parent: Optional[int]
    children: Optional[List[int]]  # uids of children if internal, else None


class Partition3D:
    def __init__(self, root: Box3D):
        self.nodes: Dict[int, Node3D] = {}
        self.leaves: Dict[int, Node3D] = {}
        self._next_uid = 0

        root_uid = self._alloc_uid()
        self.root_uid = root_uid
        self.nodes[root_uid] = Node3D(uid=root_uid, box=root, parent=None, children=None)
        self.leaves[root_uid] = self.nodes[root_uid]

    def _alloc_uid(self) -> int:
        u = self._next_uid
        self._next_uid += 1
        return u

    def all_leaves(self) -> Set[int]:
        return set(self.leaves.keys())

    def get_box(self, uid: int) -> Box3D:
        return self.nodes[uid].box

    def is_leaf(self, uid: int) -> bool:
        return self.nodes[uid].children is None

    def _split_leaf_axis(self, uid: int, axis: str, cut: float) -> Tuple[int, int]:
        """
        Split a LEAF node into two children along axis at 'cut'.
        axis in {"p","q","th"}.
        Returns (left_uid, right_uid) where left is lower side.
        """
        node = self.nodes[uid]
        if node.children is not None:
            raise ValueError(f"uid {uid} is not a leaf")

        b = node.box
        if axis == "p":
            if not (b.p_lo < cut < b.p_hi):
                raise ValueError("cut not strictly inside leaf on p")
            bL = Box3D(b.p_lo, cut, b.q_lo, b.q_hi, b.th_lo, b.th_hi)
            bR = Box3D(cut, b.p_hi, b.q_lo, b.q_hi, b.th_lo, b.th_hi)
        elif axis == "q":
            if not (b.q_lo < cut < b.q_hi):
                raise ValueError("cut not strictly inside leaf on q")
            bL = Box3D(b.p_lo, b.p_hi, b.q_lo, cut, b.th_lo, b.th_hi)
            bR = Box3D(b.p_lo, b.p_hi, cut, b.q_hi, b.th_lo, b.th_hi)
        elif axis == "th":
            if not (b.th_lo < cut < b.th_hi):
                raise ValueError("cut not strictly inside leaf on th")
            bL = Box3D(b.p_lo, b.p_hi, b.q_lo, b.q_hi, b.th_lo, cut)
            bR = Box3D(b.p_lo, b.p_hi, b.q_lo, b.q_hi, cut, b.th_hi)
        else:
            raise ValueError("axis must be one of {'p','q','th'}")

        uL = self._alloc_uid()
        uR = self._alloc_uid()
        self.nodes[uL] = Node3D(uid=uL, box=bL, parent=uid, children=None)
        self.nodes[uR] = Node3D(uid=uR, box=bR, parent=uid, children=None)

        # make uid internal
        node.children = [uL, uR]
        self.nodes[uid] = node

        # update leaves
        self.leaves.pop(uid, None)
        self.leaves[uL] = self.nodes[uL]
        self.leaves[uR] = self.nodes[uR]
        return uL, uR

    def make_uniform_grid(self, p_bins: int, q_bins: int, th_bins: int) -> None:
        """
        Build a uniform p×q×theta grid by applying axis-cuts to all spanning leaves.
        """
        root = self.nodes[self.root_uid].box
        p_edges = np.linspace(root.p_lo, root.p_hi, p_bins + 1)
        q_edges = np.linspace(root.q_lo, root.q_hi, q_bins + 1)
        t_edges = np.linspace(root.th_lo, root.th_hi, th_bins + 1)

        # cut helper: split all leaves that strictly contain cut on that axis
        def apply_cuts(axis: str, cuts: np.ndarray) -> None:
            for cut in cuts[1:-1]:
                # snapshot leaves at this moment (we'll modify during loop)
                to_split = []
                for uid, nd in list(self.leaves.items()):
                    b = nd.box
                    if axis == "p" and (b.p_lo < cut < b.p_hi):
                        to_split.append(uid)
                    elif axis == "q" and (b.q_lo < cut < b.q_hi):
                        to_split.append(uid)
                    elif axis == "th" and (b.th_lo < cut < b.th_hi):
                        to_split.append(uid)
                for uid in to_split:
                    # some leaves may have been split already by earlier operations; guard
                    if uid in self.leaves:
                        b = self.leaves[uid].box
                        if axis == "p" and (b.p_lo < cut < b.p_hi):
                            self._split_leaf_axis(uid, "p", float(cut))
                        elif axis == "q" and (b.q_lo < cut < b.q_hi):
                            self._split_leaf_axis(uid, "q", float(cut))
                        elif axis == "th" and (b.th_lo < cut < b.th_hi):
                            self._split_leaf_axis(uid, "th", float(cut))

        apply_cuts("p", p_edges)
        apply_cuts("q", q_edges)
        apply_cuts("th", t_edges)

    def refine_oct(self, uid: int) -> List[int]:
        """
        Refine a leaf into 8 children by splitting at midpoints in p,q,th.
        Returns list of new leaf uids (8).
        """
        if uid not in self.leaves:
            return []

        b = self.leaves[uid].box
        pm = 0.5 * (b.p_lo + b.p_hi)
        qm = 0.5 * (b.q_lo + b.q_hi)
        tm = 0.5 * (b.th_lo + b.th_hi)

        # split p -> 2
        uL, uR = self._split_leaf_axis(uid, "p", pm)
        # split q on both -> 4
        uLL, uLR = self._split_leaf_axis(uL, "q", qm)
        uRL, uRR = self._split_leaf_axis(uR, "q", qm)
        # split th on all 4 -> 8
        kids = []
        for u in [uLL, uLR, uRL, uRR]:
            a, b_ = self._split_leaf_axis(u, "th", tm)
            kids.extend([a, b_])
        return kids

    def query_intersecting_leaves(self, query: Box3D) -> Set[int]:
        """
        Return all leaf uids whose boxes intersect query.
        """
        out: Set[int] = set()

        def rec(u: int) -> None:
            nd = self.nodes[u]
            if not nd.box.intersects(query):
                return
            if nd.children is None:
                out.add(u)
                return
            for c in nd.children:
                rec(c)

        rec(self.root_uid)
        return out

