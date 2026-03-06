from __future__ import annotations

import numpy as np

from dataclasses import dataclass
from typing import Callable, Dict, Iterable, List, Optional, Set, Tuple


# Geometry

@dataclass(frozen=True)
class Rect:
    """Axis-aligned rectangle."""

    xmin: float
    xmax: float
    ymin: float
    ymax: float

    def contains_point(self, x: float, y: float) -> bool:
        return (self.xmin <= x <= self.xmax) and (self.ymin <= y <= self.ymax)

    def contains_rect(self, other: "Rect") -> bool:
        return (
            self.xmin <= other.xmin
            and other.xmax <= self.xmax
            and self.ymin <= other.ymin
            and other.ymax <= self.ymax
        )

    def intersects(self, other: "Rect") -> bool:
        return not (
            self.xmax < other.xmin
            or other.xmax < self.xmin
            or self.ymax < other.ymin
            or other.ymax < self.ymin
        )

    def intersection(self, other: "Rect") -> Optional["Rect"]:
        xmin = max(self.xmin, other.xmin)
        xmax = min(self.xmax, other.xmax)
        ymin = max(self.ymin, other.ymin)
        ymax = min(self.ymax, other.ymax)
        if xmin > xmax or ymin > ymax:
            return None
        return Rect(xmin, xmax, ymin, ymax)

    def width(self) -> float:
        return self.xmax - self.xmin

    def height(self) -> float:
        return self.ymax - self.ymin

    def center(self) -> Tuple[float, float]:
        return ((self.xmin + self.xmax) / 2.0, (self.ymin + self.ymax) / 2.0)

    def split4(self, xm: float, ym: float) -> Tuple["Rect", "Rect", "Rect", "Rect"]:
        """Split into 4 rectangles at (xm, ym)."""
        return (
            Rect(self.xmin, xm, self.ymin, ym),
            Rect(xm, self.xmax, self.ymin, ym),
            Rect(self.xmin, xm, ym, self.ymax),
            Rect(xm, self.xmax, ym, self.ymax),
        )


@dataclass
class CellNode:
    uid: int
    rect: Rect
    parent: Optional["CellNode"] = None
    children: Optional[List["CellNode"]] = None
    depth: int = 0

    def is_leaf(self) -> bool:
        return self.children is None


# Partition

class RectPartition:
    """Rectilinear hierarchical partition.

    Roots are created by a uniform grid. Refinement may introduce additional
    axis-aligned split hyperplanes (binary splits along x or y).
    """

    def __init__(self, roots: List[CellNode], domain: Rect, nx: int, ny: int, next_uid: int):
        self.roots = roots
        self.domain = domain
        self.nx = nx
        self.ny = ny
        self.next_uid = next_uid

        self.leaves: Dict[int, CellNode] = {}
        self._rebuild_leaves()

    def _rebuild_leaves(self) -> None:
        self.leaves.clear()

        def _walk(n: CellNode) -> None:
            if n.children is None:
                self.leaves[n.uid] = n
                return
            for c in n.children:
                _walk(c)

        for r in self.roots:
            _walk(r)

    @staticmethod
    def uniform_grid(domain: Rect, nx: int, ny: int) -> "RectPartition":
        roots: List[CellNode] = []
        uid = 0
        xs = np.linspace(domain.xmin, domain.xmax, nx + 1)
        ys = np.linspace(domain.ymin, domain.ymax, ny + 1)
        for i in range(nx):
            for j in range(ny):
                r = Rect(float(xs[i]), float(xs[i + 1]), float(ys[j]), float(ys[j + 1]))
                roots.append(CellNode(uid=uid, rect=r, parent=None, children=None, depth=0))
                uid += 1
        return RectPartition(roots=roots, domain=domain, nx=nx, ny=ny, next_uid=uid)

    def leaf_uid_for_point(self, x: float, y: float) -> Optional[int]:
        if not self.domain.contains_point(x, y):
            return None

        gx = (x - self.domain.xmin) / (self.domain.xmax - self.domain.xmin)
        gy = (y - self.domain.ymin) / (self.domain.ymax - self.domain.ymin)
        i = int(min(max(gx * self.nx, 0), self.nx - 1))
        j = int(min(max(gy * self.ny, 0), self.ny - 1))
        root = self.roots[i * self.ny + j]

        n = root
        while n.children is not None:
            found = None
            for c in n.children:
                if c.rect.contains_point(x, y):
                    found = c
                    break
            if found is None:
                return None
            n = found
        return n.uid

    def query_intersecting_leaves(self, box: Rect) -> List[int]:
        out: List[int] = []

        def _q(n: CellNode) -> None:
            if not n.rect.intersects(box):
                return
            if n.children is None:
                out.append(n.uid)
                return
            for c in n.children:
                _q(c)

        for r in self.roots:
            _q(r)
        return out

    def _new_uid(self) -> int:
        u = self.next_uid
        self.next_uid += 1
        return u

    def split_leaf_x(self, uid: int, x_cut: float) -> Tuple[int, int]:
        """Binary split of a leaf along the vertical line x = x_cut."""
        if uid not in self.leaves:
            raise KeyError(f"uid {uid} is not a leaf")
        n = self.leaves[uid]
        r = n.rect
        if not (r.xmin < x_cut < r.xmax):
            raise ValueError("x_cut must lie strictly inside the leaf rectangle")

        uL = self._new_uid()
        uR = self._new_uid()
        left = CellNode(uid=uL, rect=Rect(r.xmin, x_cut, r.ymin, r.ymax), parent=n, children=None, depth=n.depth + 1)
        right = CellNode(uid=uR, rect=Rect(x_cut, r.xmax, r.ymin, r.ymax), parent=n, children=None, depth=n.depth + 1)
        n.children = [left, right]

        del self.leaves[uid]
        self.leaves[uL] = left
        self.leaves[uR] = right
        return uL, uR

    def split_leaf_y(self, uid: int, y_cut: float) -> Tuple[int, int]:
        """Binary split of a leaf along the horizontal line y = y_cut."""
        if uid not in self.leaves:
            raise KeyError(f"uid {uid} is not a leaf")
        n = self.leaves[uid]
        r = n.rect
        if not (r.ymin < y_cut < r.ymax):
            raise ValueError("y_cut must lie strictly inside the leaf rectangle")

        uB = self._new_uid()
        uT = self._new_uid()
        bot = CellNode(uid=uB, rect=Rect(r.xmin, r.xmax, r.ymin, y_cut), parent=n, children=None, depth=n.depth + 1)
        top = CellNode(uid=uT, rect=Rect(r.xmin, r.xmax, y_cut, r.ymax), parent=n, children=None, depth=n.depth + 1)
        n.children = [bot, top]

        del self.leaves[uid]
        self.leaves[uB] = bot
        self.leaves[uT] = top
        return uB, uT

    def split_leaf4(self, uid: int, xm: float, ym: float) -> Tuple[int, int, int, int]:
        """(Legacy) 4-way split at (xm, ym)."""
        if uid not in self.leaves:
            raise KeyError(f"uid {uid} is not a leaf")
        n = self.leaves[uid]
        r = n.rect
        if not (r.xmin < xm < r.xmax and r.ymin < ym < r.ymax):
            raise ValueError("(xm,ym) must lie strictly inside the leaf rectangle")

        rects = r.split4(xm, ym)
        u0, u1, u2, u3 = (self._new_uid(), self._new_uid(), self._new_uid(), self._new_uid())
        c0 = CellNode(uid=u0, rect=rects[0], parent=n, children=None, depth=n.depth + 1)
        c1 = CellNode(uid=u1, rect=rects[1], parent=n, children=None, depth=n.depth + 1)
        c2 = CellNode(uid=u2, rect=rects[2], parent=n, children=None, depth=n.depth + 1)
        c3 = CellNode(uid=u3, rect=rects[3], parent=n, children=None, depth=n.depth + 1)
        n.children = [c0, c1, c2, c3]

        del self.leaves[uid]
        self.leaves[u0] = c0
        self.leaves[u1] = c1
        self.leaves[u2] = c2
        self.leaves[u3] = c3
        return u0, u1, u2, u3


# Abstraction

class TransitionRelation:
    def __init__(self) -> None:
        self.succ: Dict[int, Dict[str, Set[int]]] = {}
        self.pred: Dict[int, Dict[str, Set[int]]] = {}

    def set_succ(self, u: int, a: str, vs: Set[int]) -> None:
        self.succ.setdefault(u, {})
        self.pred.setdefault(u, {})
        if a in self.succ[u]:
            for v_old in self.succ[u][a]:
                if v_old in self.pred and a in self.pred[v_old]:
                    self.pred[v_old][a].discard(u)
        self.succ[u][a] = set(vs)
        for v in vs:
            self.pred.setdefault(v, {})
            self.pred[v].setdefault(a, set())
            self.pred[v][a].add(u)


@dataclass
class AffineDynamics:
    """Discrete-time affine map: x_{t+1} = A x_t + b."""

    A: np.ndarray
    b: np.ndarray

    def dynamics(self, x: np.ndarray) -> np.ndarray:
        return self.A @ x + self.b

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
        img = (self.A @ corners.T).T + self.b[None, :]
        xmin, ymin = np.min(img, axis=0)
        xmax, ymax = np.max(img, axis=0)
        return Rect(float(xmin), float(xmax), float(ymin), float(ymax))


class Abstraction:
    OUT_UID = -1

    def __init__(
        self,
        part: RectPartition,
        dyn_by_action: Dict[str, AffineDynamics],
        ap_labeler: Callable[[Optional[Rect]], Set[str]],
    ) -> None:
        self.part = part
        self.dyn_by_action = dyn_by_action
        self.ap_labeler = ap_labeler
        self.tr = TransitionRelation()

        # Purged transitions are removed from the abstract model.
        self._purged: Dict[str, Dict[int, Set[int]]] = {"step": {}}

    def aps_and_labels(self) -> Tuple[Set[str], Dict[int, Set[str]]]:
        labels_by_uid: Dict[int, Set[str]] = {}
        all_aps: Set[str] = set()

        out_labels = set(self.ap_labeler(None))
        labels_by_uid[self.OUT_UID] = out_labels
        all_aps |= out_labels

        for u, node in self.part.leaves.items():
            labs = set(self.ap_labeler(node.rect))
            labels_by_uid[u] = labs
            all_aps |= labs

        return all_aps, labels_by_uid

    def to_spot_kripke(self, init_uids: Set[int], merge_actions: bool = True):
        import spot  # type: ignore
        from buddy import bdd_ithvar, bddtrue  # type: ignore

        all_aps, labels_by_uid = self.aps_and_labels()

        d = spot.make_bdd_dict()
        k = spot.make_kripke_graph(d)

        ap_to_bdd = {ap: bdd_ithvar(k.register_ap(ap)) for ap in sorted(all_aps)}

        uids = [self.OUT_UID] + sorted(self.part.leaves.keys())
        uid_to_sid: Dict[int, int] = {}
        state_names: List[str] = []

        for u in uids:
            b = bddtrue
            for ap in labels_by_uid.get(u, set()):
                b &= ap_to_bdd[ap]
            sid = k.new_state(b)
            uid_to_sid[u] = sid
            state_names.append(str(u))

        k.set_state_names(state_names)

        if len(init_uids) == 1:
            k.set_init_state(uid_to_sid[next(iter(init_uids))])
        else:
            init_sid = k.new_state(bddtrue)
            k.set_init_state(init_sid)
            for u in init_uids:
                k.new_edge(init_sid, uid_to_sid[u])

        for u, by_a in self.tr.succ.items():
            su = uid_to_sid[u]
            if merge_actions:
                vs: Set[int] = set()
                for _a, vs_a in by_a.items():
                    vs |= vs_a
                for v in vs:
                    k.new_edge(su, uid_to_sid[v])
            else:
                for v in by_a.get("step", set()):
                    k.new_edge(su, uid_to_sid[v])

        return k, d

    def purge_transition(self, u: int, v: int, action: str = "step") -> None:
        self._purged.setdefault(action, {})
        self._purged[action].setdefault(u, set()).add(v)

        if u in self.tr.succ and action in self.tr.succ[u]:
            if v in self.tr.succ[u][action]:
                self.tr.succ[u][action].discard(v)
                if v in self.tr.pred and action in self.tr.pred[v]:
                    self.tr.pred[v][action].discard(u)

    def _apply_purge(self, u: int, action: str, vs: Set[int]) -> Set[int]:
        blocked = self._purged.get(action, {}).get(u, set())
        if not blocked:
            return vs
        return set(vs) - set(blocked)

    def rebuild_all_transitions(self) -> None:
        self.tr = TransitionRelation()
        self.tr.set_succ(self.OUT_UID, "step", {self.OUT_UID})

        for u, node in self.part.leaves.items():
            rect = node.rect
            dyn = self.dyn_by_action["step"]
            box = dyn.image_bbox(rect)

            vs = set(self.part.query_intersecting_leaves(box))

            if not self.part.domain.contains_rect(box):
                vs.add(self.OUT_UID)

            if not vs:
                vs = {self.OUT_UID}

            vs = self._apply_purge(u, "step", vs)
            if not vs:
                vs = {self.OUT_UID}

            self.tr.set_succ(u, "step", vs)

    def isolate_subrect(
        self,
        uid: int,
        target: Rect,
        *,
        min_w: float = 0.0,
        min_h: float = 0.0,
        max_depth: Optional[int] = None,
    ) -> Optional[int]:
        if uid not in self.part.leaves:
            return None
        node = self.part.leaves[uid]
        X = node.rect
        if not X.contains_rect(target):
            raise ValueError("target must be contained in the leaf rect")

        def _depth_ok(u: int) -> bool:
            if max_depth is None:
                return True
            return self.part.leaves[u].depth < max_depth

        def _split_x(u: int, x_cut: float) -> Optional[int]:
            r = self.part.leaves[u].rect
            if not (r.xmin < x_cut < r.xmax):
                return u
            if (x_cut - r.xmin) < min_w or (r.xmax - x_cut) < min_w:
                return None
            if not _depth_ok(u):
                return None
            uL, uR = self.part.split_leaf_x(u, x_cut)
            if self.part.leaves[uL].rect.contains_rect(target):
                return uL
            return uR

        def _split_y(u: int, y_cut: float) -> Optional[int]:
            r = self.part.leaves[u].rect
            if not (r.ymin < y_cut < r.ymax):
                return u
            if (y_cut - r.ymin) < min_h or (r.ymax - y_cut) < min_h:
                return None
            if not _depth_ok(u):
                return None
            uB, uT = self.part.split_leaf_y(u, y_cut)
            if self.part.leaves[uB].rect.contains_rect(target):
                return uB
            return uT

        cur = uid
        if target.xmin > X.xmin:
            cur = _split_x(cur, target.xmin)
            if cur is None:
                return None
        if target.xmax < X.xmax:
            cur = _split_x(cur, target.xmax)
            if cur is None:
                return None

        if target.ymin > X.ymin:
            cur = _split_y(cur, target.ymin)
            if cur is None:
                return None
        if target.ymax < X.ymax:
            cur = _split_y(cur, target.ymax)
            if cur is None:
                return None

        if cur is None:
            return None
        return cur

