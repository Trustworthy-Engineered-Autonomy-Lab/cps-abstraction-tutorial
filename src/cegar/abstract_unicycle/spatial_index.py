from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, List, Set, Tuple

import numpy as np

from unicycle_partition_3d import Box3D


@dataclass
class SpatialHash3D:

    p_lo: float
    p_hi: float
    q_lo: float
    q_hi: float
    th_lo: float
    th_hi: float
    nb_p: int
    nb_q: int
    nb_th: int

    def __post_init__(self) -> None:
        self.nb_p = int(self.nb_p)
        self.nb_q = int(self.nb_q)
        self.nb_th = int(self.nb_th)
        assert self.nb_p > 0 and self.nb_q > 0 and self.nb_th > 0

        self._bins: Dict[Tuple[int, int, int], Set[int]] = {}
        self._uid_bins: Dict[int, List[Tuple[int, int, int]]] = {}

        self._wp = (self.p_hi - self.p_lo) / self.nb_p
        self._wq = (self.q_hi - self.q_lo) / self.nb_q
        self._wt = (self.th_hi - self.th_lo) / self.nb_th

    def clear(self) -> None:
        self._bins.clear()
        self._uid_bins.clear()

    @staticmethod
    def _clamp(i: int, n: int) -> int:
        if i < 0:
            return 0
        if i >= n:
            return n - 1
        return i

    def _idx_range(self, lo: float, hi: float, base: float, w: float, n: int) -> Tuple[int, int]:
        # Inclusive bin range covering [lo, hi).
        eps = 1e-12
        i0 = int(np.floor((lo - base) / w))
        i1 = int(np.floor(((hi - eps) - base) / w))
        return self._clamp(i0, n), self._clamp(i1, n)

    def _keys_for_box(self, b: Box3D) -> List[Tuple[int, int, int]]:
        ip0, ip1 = self._idx_range(b.p_lo, b.p_hi, self.p_lo, self._wp, self.nb_p)
        iq0, iq1 = self._idx_range(b.q_lo, b.q_hi, self.q_lo, self._wq, self.nb_q)
        it0, it1 = self._idx_range(b.th_lo, b.th_hi, self.th_lo, self._wt, self.nb_th)

        keys: List[Tuple[int, int, int]] = []
        for i in range(ip0, ip1 + 1):
            for j in range(iq0, iq1 + 1):
                for k in range(it0, it1 + 1):
                    keys.append((i, j, k))
        return keys

    def insert(self, uid: int, b: Box3D) -> None:
        keys = self._keys_for_box(b)
        self._uid_bins[uid] = keys
        for key in keys:
            self._bins.setdefault(key, set()).add(uid)

    def remove(self, uid: int) -> None:
        keys = self._uid_bins.pop(uid, None)
        if not keys:
            return
        for key in keys:
            s = self._bins.get(key)
            if not s:
                continue
            s.discard(uid)
            if not s:
                self._bins.pop(key, None)

    def bulk_build(self, items: Iterable[Tuple[int, Box3D]]) -> None:
        self.clear()
        for uid, b in items:
            self.insert(uid, b)

    def query_candidates(self, q: Box3D) -> Set[int]:
        keys = self._keys_for_box(q)
        out: Set[int] = set()
        for key in keys:
            s = self._bins.get(key)
            if s:
                out |= s
        return out
