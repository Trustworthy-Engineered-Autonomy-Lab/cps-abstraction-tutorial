from __future__ import annotations
from collections import deque
from typing import Dict, Iterable, List, Optional, Set, Tuple, Any

# Robust accessors for pyModelChecking Kripke objects across versions.

def _maybe_call(x):
    return x() if callable(x) else x

def kripke_states(K) -> List[int]:
    for attr in ("S", "states", "_Kripke__S", "_Kripke__states"):
        if hasattr(K, attr):
            v = _maybe_call(getattr(K, attr))
            return [int(s) for s in list(v)]
    for meth in ("getStates", "get_states", "States"):
        if hasattr(K, meth):
            v = getattr(K, meth)()
            return [int(s) for s in list(v)]
    # Fallback: derive from transitions + initials + labeling keys
    st: Set[int] = set()
    for (a,b) in kripke_edges(K):
        st.add(int(a)); st.add(int(b))
    for s in kripke_initials(K):
        st.add(int(s))
    L = kripke_labeling_dict(K)
    if L:
        st |= {int(k) for k in L.keys()}
    return sorted(st)

def kripke_initials(K) -> List[int]:
    for attr in ("S0", "initial_states", "_Kripke__S0", "_Kripke__initial"):
        if hasattr(K, attr):
            v = _maybe_call(getattr(K, attr))
            return [int(s) for s in list(v)]
    for meth in ("getInitialStates", "get_initial_states", "InitialStates"):
        if hasattr(K, meth):
            v = getattr(K, meth)()
            return [int(s) for s in list(v)]
    return []

def kripke_edges(K) -> List[Tuple[int,int]]:
    for attr in ("R", "edges", "transitions", "_Kripke__R", "_Kripke__edges"):
        if hasattr(K, attr):
            v = _maybe_call(getattr(K, attr))
            return [(int(a), int(b)) for (a,b) in list(v)]
    for meth in ("getTransitions", "get_transitions", "Transitions"):
        if hasattr(K, meth):
            v = getattr(K, meth)()
            return [(int(a), int(b)) for (a,b) in list(v)]
    return []

def kripke_labeling_dict(K) -> Optional[Dict[int, List[str]]]:
    for attr in ("L", "labeling", "labels", "_Kripke__L"):
        if hasattr(K, attr):
            v = getattr(K, attr)
            if isinstance(v, dict):
                return {int(k): [str(x) for x in v[k]] for k in v}
            # If it's callable, we'll handle per-state
    return None

def kripke_labels(K, s: int) -> List[str]:
    # Method-like
    for meth in ("L", "label", "labels", "getLabel", "get_label"):
        if hasattr(K, meth):
            v = getattr(K, meth)
            if callable(v):
                try:
                    labs = v(s)
                    return [str(x) for x in list(labs)]
                except TypeError:
                    pass
    # Dict-like
    d = kripke_labeling_dict(K)
    if d is not None:
        return [str(x) for x in d.get(int(s), [])]
    return []

def _succ_map(K) -> Dict[int, List[int]]:
    succ: Dict[int, List[int]] = {int(s): [] for s in kripke_states(K)}
    for (a, b) in kripke_edges(K):
        succ.setdefault(int(a), []).append(int(b))
    return succ

def find_witness_A_safe_U_goal(K, init_states: Iterable[int]) -> Optional[Tuple[List[int], List[int]]]:
    """Return (prefix, cycle) witness for violation of A (safe U goal).

    Violation if there exists a path that:
      (1) reaches a 'fail' state before reaching 'goal', while avoiding goal, OR
      (2) stays within 'safe' forever while avoiding 'goal' (a safe SCC cycle).
    """
    succ = _succ_map(K)
    init = [int(s) for s in init_states]

    def labs(s: int) -> Set[str]:
        return set(kripke_labels(K, s))

    # (1) reach fail before goal (BFS)
    q = deque()
    parent: Dict[int, Optional[int]] = {}
    seen: Set[int] = set()
    for s in init:
        if "goal" in labs(s):
            continue
        q.append(s)
        parent[s] = None
        seen.add(s)

    fail_state = None
    while q:
        u = q.popleft()
        if "fail" in labs(u):
            fail_state = u
            break
        for v in succ.get(u, []):
            if v in seen:
                continue
            if "goal" in labs(v):
                continue
            seen.add(v)
            parent[v] = u
            q.append(v)

    if fail_state is not None:
        path: List[int] = []
        cur: Optional[int] = fail_state
        while cur is not None:
            path.append(cur)
            cur = parent.get(cur, None)
        path.reverse()
        return path, []

    # (2) safe SCC avoiding goal/fail
    safe_nodes = {
        s for s in succ.keys()
        if ("safe" in labs(s)) and ("goal" not in labs(s)) and ("fail" not in labs(s))
    }

    reachable: Set[int] = set()
    pred: Dict[int, Optional[int]] = {}
    q = deque()
    for s in init:
        if s in safe_nodes:
            reachable.add(s)
            pred[s] = None
            q.append(s)
    while q:
        u = q.popleft()
        for v in succ.get(u, []):
            if v not in safe_nodes:
                continue
            if v in reachable:
                continue
            reachable.add(v)
            pred[v] = u
            q.append(v)

    if not reachable:
        return None

    # Tarjan SCC on reachable safe subgraph
    index = 0
    stack: List[int] = []
    onstack: Set[int] = set()
    indices: Dict[int, int] = {}
    lowlink: Dict[int, int] = {}
    sccs: List[List[int]] = []

    def strongconnect(v: int):
        nonlocal index
        indices[v] = index
        lowlink[v] = index
        index += 1
        stack.append(v)
        onstack.add(v)

        for w in succ.get(v, []):
            if w not in reachable or w not in safe_nodes:
                continue
            if w not in indices:
                strongconnect(w)
                lowlink[v] = min(lowlink[v], lowlink[w])
            elif w in onstack:
                lowlink[v] = min(lowlink[v], indices[w])

        if lowlink[v] == indices[v]:
            comp: List[int] = []
            while True:
                w = stack.pop()
                onstack.remove(w)
                comp.append(w)
                if w == v:
                    break
            sccs.append(comp)

    for v in list(reachable):
        if v not in indices:
            strongconnect(v)

    def has_cycle(comp: List[int]) -> bool:
        if len(comp) > 1:
            return True
        v = comp[0]
        return v in succ.get(v, [])

    cycle_comp = next((c for c in sccs if has_cycle(c)), None)
    if cycle_comp is None:
        return None

    c0 = cycle_comp[0]
    pref: List[int] = []
    cur: Optional[int] = c0
    while cur is not None:
        pref.append(cur)
        cur = pred.get(cur, None)
    pref.reverse()

    sccset = set(cycle_comp)
    if len(cycle_comp) == 1:
        cyc = [c0]
    else:
        # find short cycle back to c0
        parent2: Dict[int, Optional[int]] = {c0: None}
        dq = deque([c0])
        back_from = None
        while dq and back_from is None:
            u = dq.popleft()
            for v in succ.get(u, []):
                if v not in sccset:
                    continue
                if v == c0 and u != c0:
                    back_from = u
                    break
                if v not in parent2:
                    parent2[v] = u
                    dq.append(v)
        if back_from is None:
            cyc = cycle_comp
        else:
            cyc_rev = [c0]
            cur = back_from
            while cur is not None and cur != c0:
                cyc_rev.append(cur)
                cur = parent2[cur]
            cyc = list(reversed(cyc_rev))

    return pref, cyc
